from Common.Server.fl_grpc_server_adaclipping import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_Clipping
from Common.Handler.handler import Handler

import Common.config as config

import numpy as np


class ClearDenseServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearDenseServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler
        self.clippingBound = config.initClippingBound

    # override UpdateGrad_float for server
    # receive gradients from clients, aggregate, and give them back
    def UpdateGrad_Clipping(self, request, context):
        data_dict = {request.id: request.grad_ori}
        b_list = [request.b]
        print("have received:", data_dict.keys(), b_list, 'clip_b:', np.round(self.clippingBound,4))
        rst, self.clippingBound = super().process(dict_data=data_dict, b=b_list, handler=self.handler.computation, clippingBound=self.clippingBound)
        return GradResponse_Clipping(b=self.clippingBound, grad_upd=rst)


class AvgGradientHandler(Handler):
    def __init__(self, num_workers):
        super(AvgGradientHandler, self).__init__()
        self.num_workers = num_workers

    def computation(self, data_in, b_in:list, S, gamma, blr):
        # calculating adaptive noise
        grad_noise = (config.z_multiplier**(-2) - (2*config.b_noise)**(-2))**(-0.5) * S
        # average aggregator
        grad_in = np.array(data_in).reshape((self.num_workers, -1))
        grad_in += np.random.normal(0, grad_noise, size=grad_in.shape)
        grad_in = grad_in.mean(axis=0)
        # new bound computation
        b_avg = (np.sum(b_in) + np.random.normal(0,config.b_noise)) / config.num_workers
        S *= np.exp(-blr*(b_avg-gamma))

        return grad_in.tolist(), S


if __name__ == "__main__":
    gradient_handler = AvgGradientHandler(num_workers=config.num_workers)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    clear_server.start()
