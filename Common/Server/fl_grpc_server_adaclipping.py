import threading
import numpy as np
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcServicer, add_FL_GrpcServicer_to_server
from Common import config
from concurrent import futures
import grpc
import time

con = threading.Condition()
num = 0
clipping_bound = 0

data_ori = {}
data_upd = []
data_b = []


class FlGrpcServer(FL_GrpcServicer):
    def __init__(self, config):
        super(FlGrpcServer, self).__init__()
        self.config = config

    # multi threads
    def process(self, dict_data, b, handler, clippingBound):
        global num, data_ori, data_upd, data_b, clipping_bound

        data_ori.update(dict_data)
        data_b += b

        con.acquire() # request the lock
        num += 1
        if num < self.config.num_workers:
            con.wait() # wait for the awake
        else:
            rst = [data_ori[k] for k in sorted(data_ori.keys())]
            rst = np.array(rst).flatten()
            data_upd, clipping_bound = handler(data_in=rst, b_in=data_b, S=clippingBound, gamma=self.config.gamma, blr=self.config.blr)
            data_ori = {}
            data_b = []
            num = 0
            con.notifyAll() # wake up all waiting threads

        con.release() # release the lock

        return data_upd, clipping_bound

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.num_workers), options=self.config.grpc_options)
        add_FL_GrpcServicer_to_server(self, server)

        target = self.address + ":" + str(self.port)
        #target = '192.168.126.12:5000'
        server.add_insecure_port(target)
        server.start()

        try:
            while True:
                time.sleep(60 * 60 * 24)
        except KeyboardInterrupt:
            server.stop(0)
