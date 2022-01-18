from Common.Server._old.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_float
from Common.Handler.handler import Handler
import Common.config as config
import torch
import hdbscan
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats
import numpy as np
from Common.Model.LeNet import LeNet
from crab.metrics.pairwise import adjusted_cosine
from mpc_clustering.mpc_dbscan import *
import time

from crypten.mpc import MPCTensor
import crypten
import crypten.mpc as mpc


class ClearFLGuardServer(FlGrpcServer):
    def __init__(self, address, port, config, handler, num_model_params):
        super(ClearFLGuardServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler
        self.num_model_params = num_model_params

    def UpdateGrad_float(self, request, context): # using flguard
        data_dict = {request.id: request.grad_ori}
        #print("have received:", data_dict.keys())
        #rst,#id = super().process(dict_data=data_dict, handler=self.handler.computation)
        #if
        rst, split_label = super().process(dict_data=data_dict, handler=self.handler.computation)
        print(rst)
        print(split_label)
        num_groups = len(split_label)
        for i in range(num_groups):
            if request.id in split_label[i]:
               return GradResponse_float(grad_upd=rst[i*self.num_model_params:(i+1)*self.num_model_params])


class FLGuardGradientHandler(Handler):
    def __init__(self, num_workers, f, weights):
        super(FLGuardGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.f = f
        self.weights = weights
        self.lambdaa = 0.001
        #self.cluster = hdbscan.HDBSCAN(metric='l2', min_cluster_size=2, allow_single_cluster=True, min_samples=1, cluster_selection_epsilon=0.1)

    # @mpc.run_multiprocess(world_size=1)
    # def computation(self, data_in):
    #     weights_in = np.array(data_in).reshape((self.num_workers, -1))
    #     weights_in_mpc = torch.from_numpy(np.divide(weights_in, 10))
    #
    #     nc = weights_in_mpc.shape[0]
    #     data_enc = MPCTensor(weights_in_mpc, ptype=crypten.mpc.arithmetic)
    #     distance_matrix = torch.ones((nc, nc))
    #     distance_enc = MPCTensor(distance_matrix, ptype=crypten.mpc.arithmetic)
    #     cos = crypten.nn.CosineSimilarity(dim=0, eps=1e-6)
    #     for i in range(nc - 1):
    #         for j in range(i + 1, nc):
    #             distance_enc[i, j] = cos(data_enc[i], data_enc[j])
    #             distance_enc[j, i] = distance_enc[i, j]
    #
    #     label = dbscan(distance_enc, eps=0.2, min_points=2)
    #
    #     label = np.array(label).squeeze()
    #
    #     split_label = []
    #     if (label == -1).all():
    #         split_label = [[i for i in range(self.num_workers)]]
    #     else:
    #         label_elements = np.unique(label)
    #         for i in label_elements.tolist():
    #             split_label.append(np.where(label == i)[0].tolist())
    #         # b = np.where(label == 0)[0].tolist()
    #     # euclidean distance between self.weights and clients' weights
    #     weight_agg = []
    #     weights_in_mpc = torch.from_numpy(weights_in)
    #     for b in split_label:
    #         weights_enc = MPCTensor(weights_in_mpc[b], ptype=crypten.mpc.arithmetic)
    #         weight_agg.append(weights_enc.mean(dim=0).get_plain_text().numpy())
    #     #self.weights = weight_agg
    #
    #     weight_agg = np.array(weight_agg).flatten()
    #     print(split_label)
    #     print(weight_agg)
    #     return weight_agg

    def computation(self, data_in):
        # cluster
        weights_in = np.array(data_in).reshape((self.num_workers, -1))
        weights_in_mpc = torch.from_numpy(np.divide(weights_in, 10))
        # weights_in_average = np.mean(weights_in,axis=0)
        # distance_matrix = 1 - adjusted_cosine(weights_in,weights_in,weights_in_average)

        # distance_matrix = pairwise_distances(weights_in, metric='cosine')
        # distance_matrix = np.round(distance_matrix)
        # self.cluster.fit(distance_matrix)
        # label = self.cluster.labels_
        #
        # split_label = []
        # if (label == -1).all():
        #     split_label = [[i for i in range(self.num_workers)]]
        # else:
        #     label_elements = np.unique(label)
        #     for i in label_elements.tolist():
        #         split_label.append(np.where(label == i)[0].tolist())
        #     #b = np.where(label == 0)[0].tolist()
        # # euclidean distance between self.weights and clients' weights
        # weight_agg = []
        #
        # for b in split_label:
        #      weight_agg.append(np.mean(weights_in[b], axis=0))
        #
        # #self.weights = weight_agg
        #
        # weight_agg = np.array(weight_agg).flatten()
        weight_agg, split_label = mpc_dbscan(weights_in_mpc)
        print(weight_agg, split_label)
        return weight_agg, split_label

if __name__ == "__main__":
    PATH = './Model/LeNet'
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    model.load_state_dict(torch.load(PATH))
    weights = []
    for param in model.parameters():
        weights += param.data.view(-1).numpy().tolist()
    gradient_handler = FLGuardGradientHandler(num_workers=config.num_workers, f = config.f, weights=np.array(weights))

    num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    flguard_server = ClearFLGuardServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler, num_model_params=num_model_params)
    flguard_server.start()
