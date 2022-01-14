from Common.Server.fl_grpc_server_adaclipping import FlGrpcServer as FLGrpcClipServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_Clipping
from Common.Handler.handler import Handler

import numpy as np
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances

import Common.config as config
from Common.Utils.gradients_recorder import detect_GAN_raw, save_distance_matrix

class ClearDenseServer(FLGrpcClipServer):
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
        print(data_dict.keys(), np.round(b_list,4), 'clip_b:', np.round(self.clippingBound,4))
        rst, self.clippingBound = super().process(dict_data=data_dict, b=b_list, handler=self.handler.computation, clippingBound=self.clippingBound)
        return GradResponse_Clipping(b=self.clippingBound, grad_upd=rst)


class AvgGradientHandler(Handler):
    def __init__(self, num_workers):
        super(AvgGradientHandler, self).__init__()
        self.num_workers = num_workers
        self.cluster = hdbscan.HDBSCAN(
            metric='l2', 
            min_cluster_size=2, # the smallest size grouping that you wish to consider a cluster
            allow_single_cluster=True, #
            min_samples=2, # how conservative you want you clustering to be
            cluster_selection_epsilon=0.1,
        )
        # self.npy_num = 0

    def computation(self, data_in, b_in:list, S, gamma, blr):
        # calculating adaptive noise
        # grad_noise = (config.z_multiplier**(-2) - (2*config.b_noise)**(-2))**(-0.5) * S

        # average aggregator
        grad_in = np.array(data_in).reshape((self.num_workers, -1))
        # detect_GAN_raw("Eva/gradients/GANDetection/raw_data.txt", grad_in)

        # --- HDBScan Start --- #
        distance_matrix = pairwise_distances(grad_in-grad_in.mean(axis=0), metric='cosine')
        # save_distance_matrix("Eva/distance_matrix/FedAVG_flipping.txt", distance_matrix)
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
        # if 6 in bengin_id or 7 in bengin_id or 8 in bengin_id or 9 in bengin_id:
        # if len(bengin_id) < 8:
        #     np.save('../temp/grads_'+str(self.npy_num)+'npy', grad_in)
        #     self.npy_num += 1
        print("used id:", bengin_id)
        # --- HDBScan End --- #


        # add noise to gradients from the server side
        # grad_in += np.random.normal(0, grad_noise, size=grad_in.shape)
        grad_in = grad_in[bengin_id].mean(axis=0)

        # add noise to indicators from the server side
        # b_avg = (np.sum(b_in) + np.random.normal(0,config.b_noise)) / config.num_workers

        b_in = list(map(lambda x: max(0,x), b_in))
        b_avg = np.sum(b_in) / config.num_workers
        S *= np.exp(-blr*(min(b_avg,1)-gamma))

        return grad_in.tolist(), S

        # --- TEST --- #
        grad_in = np.array(data_in).reshape((self.num_workers, -1)).mean(axis=0)
        return grad_in.tolist(), 1


if __name__ == "__main__":
    gradient_handler = AvgGradientHandler(num_workers=config.num_workers)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    print('lambda:', config.coef, 'b_noise:', config.b_noise, 'gamma:', config.gamma, 'z:', config.z_multiplier)
    clear_server.start()
