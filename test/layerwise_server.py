from Common.Server.fl_grpc_server import FlGrpcServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_float
from Common.Handler.handler import Handler
import Common.config as config
import hdbscan
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances
from scipy import stats
import numpy as np
import torch
from Common.Model.LeNet import LeNet
import torchvision.models as models


class ClearFLGuardServer(FlGrpcServer):
    def __init__(self, address, port, config, handler):
        super(ClearFLGuardServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

    def UpdateGrad_float(self, request, context):  # using flguard
        data_dict = {request.id: request.grad_ori}
        #print("have received:", data_dict.keys())
        rst = super().process(dict_data=data_dict, handler=self.handler.computation)
        return GradResponse_float(grad_upd=rst)


class FLGuardGradientHandler(Handler):
    def __init__(self, num_workers, f, weights):
        super(FLGuardGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.f = f
        self.weights = weights
        self.lambdaa = 0.001
        self.cluster = hdbscan.HDBSCAN(metric='precomputed')

    def computation(self, data_in):
        # cluster
        weights_in = np.array(data_in).reshape((self.num_workers, -1))
        ####################################################################################################
        layerslen=[]
        for p in model.parameters():
            if p.requires_grad:
                layerslen.append(p.numel())
        layerslen = [sum(layerslen[:i + 1]) for i in range(len(layerslen))]
        layerslen = layerslen[:-1]
        weights_in = np.hsplit(weights_in, layerslen)
        self.weights = np.hsplit(self.weights, layerslen)
        weight_agg = []
        count = 0
        for layer_weights in weights_in:
        ####################################################################################################

            distance_matrix = pairwise_distances(layer_weights, metric='cosine')
            self.cluster.fit(distance_matrix)
            label = self.cluster.labels_
            b = []
            if (label == -1).all():
                b = [i for i in range(self.num_workers)]
            else:
                bucket = np.zeros(label.shape)
                for value in label:
                    if value != -1:
                        bucket[value] += 1
                majority = np.argmax(bucket)
                b = np.where(label == majority).tolist()
            # euclidean distance between self.weights and clients' weights
            edis = []
            for i in range(self.num_workers):
                dist = np.linalg.norm(self.weights[count] - layer_weights[i])
                edis.append(dist)
            St = np.median(np.array(edis))
            for i in range(len(b)):
                layer_weights[b[i]] = layer_weights[b[i]] * min(1, St / edis[b[i]])

            weightstar = np.sum(layer_weights[b], axis=0) / len(b)
            #delta = self.lambdaa * St
            #weight_agg.append(weightstar + np.random.normal(0, delta, weightstar.shape))
            weight_agg.append(weightstar)
            count += 1
        weight_agg = np.hstack((z for z in weight_agg))
        self.weights = weight_agg
        return weight_agg


if __name__ == "__main__":
    PATH = './Model/LeNet'
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = LeNet().to(device)
    #model = models.resnet18().to(device)
    model.load_state_dict(torch.load(PATH))
    weights = []
    for param in model.parameters():
        weights += param.data.view(-1).numpy().tolist()
    gradient_handler = FLGuardGradientHandler(num_workers=config.num_workers, f=config.f, weights=np.array(weights))

    flguard_server = ClearFLGuardServer(address=config.server1_address, port=config.port1, config=config,
                                        handler=gradient_handler)
    flguard_server.start()
