from scipy.spatial.kdtree import distance_matrix
from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_float
from Common.Handler.handler import Handler
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import Common.config as config
import numpy as np


class ClearDenseServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearDenseServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

    # override UpdateGrad_float for server
    # receive gradients from clients, aggregate, and give them back
    def UpdateGrad_float(self, request, context):
        data_dict = {request.id: request.grad_ori}
        print("have received:", data_dict.keys())
        rst = super().process(dict_data=data_dict, handler=self.handler.computation)
        return GradResponse_float(grad_upd=rst)


class AvgGradientHandler(Handler):
    def __init__(self, num_workers):
        super(AvgGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.cluster = hdbscan.HDBSCAN(min_cluster_size=2, metric='precomputed')
        
    def computation(self, data_in):
        gradients_in = np.array(data_in).reshape((self.num_workers, -1))
       
        distance_matrix = pairwise_distances(gradients_in, metric='cosine')
        self.cluster.fit(distance_matrix)
        label = self.cluster.labels_
        if (label==-1).all():
            bengin_id = np.arange(self.num_workers).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            # label = -1 are discarded, as they cannot be assigned to any clusters
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label==majority)[0].tolist()
        print("used id:", bengin_id)
        # average aggregator
        return gradients_in[bengin_id].mean(axis=0).tolist()


if __name__ == "__main__":
    gradient_handler = AvgGradientHandler(num_workers=config.num_workers)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    clear_server.start()
