from Common.Server.fl_grpc_server_adaclipping import FlGrpcServer as FLGrpcClipServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_Clipping
from Common.Handler.handler import Handler

import numpy as np
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances

import Common.config as config
from Common.Utils.gradients_recorder import detect_GAN_raw, save_distance_matrix
from Common.Utils.gaussian_moments_account import AutoDP_epsilon, acc_track_eps

import warnings 
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

class ClearDenseServer(FLGrpcClipServer):
    def __init__(self, address, port, config, handler):
        super(ClearDenseServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

        self.clippingBound = self.config.initClippingBound

    # override UpdateGrad_float for server
    # receive gradients from clients, aggregate, and give them back
    def UpdateGrad_Clipping(self, request, context):
        data_dict = {request.id: request.grad_ori}
        b_list = [request.b]
        # print(data_dict.keys(), np.round(b_list,4), 'clip_b:', np.round(self.clippingBound,4))
        rst, self.clippingBound = super().process(dict_data=data_dict, b=b_list, handler=self.handler.computation, clippingBound=self.clippingBound)
        
        return GradResponse_Clipping(b=self.clippingBound, grad_upd=rst)

class AvgGradientHandler(Handler):
    def __init__(self, config):
        super(AvgGradientHandler, self).__init__()
        self.config = config
        self.total_number = config.total_number_clients
        self.dpoff = self.config._dpoff
        self.grad_noise_sigma = self.config.grad_noise_sigma
        self.b_noise_std = self.config.b_noise_std
        self.delta = self.config.delta
        self.clients_per_round = self.config.num_workers
        self.account_method = self.config.account_method
        self.acc_params = []

        self.grad_prev = np.array([])

        self.cluster = hdbscan.HDBSCAN(
            metric='l2', 
            min_cluster_size=2, # the smallest size grouping that you wish to consider a cluster
            allow_single_cluster=True, #
            min_samples=2, # how conservative you want you clustering to be
            cluster_selection_epsilon=0.1,
        )
        
        # moments_account:
        if self.grad_noise_sigma:
            self.sigma = (self.grad_noise_sigma**(-2) + (2*self.b_noise_std)**(-2))**(-0.5)
        else:
            self.sigma = None
        self.track_eps = AutoDP_epsilon(self.delta)

    def computation(self, data_in, b_in:list, S, gamma, blr):
        # calculating adaptive noise
        # grad_noise = (config.z_multiplier**(-2) - (2*config.b_noise)**(-2))**(-0.5) * S

        # average aggregator
        grad_in = np.array(data_in).reshape((self.clients_per_round, -1))
        # detect_GAN_raw("Eva/gradients/GANDetection/raw_data.txt", grad_in)

        # --- HDBScan Start --- #
        distance_matrix = pairwise_distances(grad_in-grad_in.mean(axis=0), metric='cosine')
        # save_distance_matrix("Eva/distance_matrix/FedAVG_flipping.txt", distance_matrix)
        self.cluster.fit(distance_matrix)
        label = self.cluster.labels_
        if (label==-1).all():
            bengin_id = np.arange(self.clients_per_round).tolist()
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
        # --- HDBScan End --- #
        if self.dpoff:
            grad_in = grad_in[bengin_id].mean(axis=0)
            S = np.inf
            print("used id: ", bengin_id)
        else:
            extra_grad_noise_std = self.grad_noise_sigma * S * np.sqrt(1-len(bengin_id)/self.clients_per_round)
            extra_b_noise_std = self.b_noise_std * np.sqrt(1-len(bengin_id)/self.clients_per_round)
            noise_compensatory_grad = np.random.normal(0, extra_grad_noise_std, size=grad_in.shape[1])
            noise_compensatory_b = np.random.normal(0, extra_b_noise_std)
            
            if self.sigma != None:
                if self.account_method == "googleTF": # Gaussian Moments Accountant
                    self.acc_params.append((len(bengin_id)/self.total_number, self.sigma, 1))
                    cur_eps, cur_delta = acc_track_eps(self.acc_params, delta=config.delta)
                elif self.account_method == "autodp": # Renyi Differential Privacy
                    self.track_eps.update_mech(len(bengin_id)/self.total_number, self.sigma, 1)
                    cur_eps, cur_delta = self.track_eps.get_epsilon(), config.delta
                print("epsilon: %.2f | delta: %.6f | "%(cur_eps, cur_delta), end="")
            print("clip_bound: %.3f | used id: "%S, bengin_id)
            
            # gradients average
            grad_in = (grad_in[bengin_id].sum(axis=0) + noise_compensatory_grad) / len(bengin_id)
            # post-processing
            grad_in_l2_norm = np.linalg.norm(grad_in)
            if grad_in_l2_norm > S: # clipping averaged gradients
                grad_in *= S/grad_in_l2_norm

            # adaptive clipping
            b_in = list(map(lambda x: max(0,x), b_in))
            b_avg = (np.sum(b_in) + noise_compensatory_b) / self.clients_per_round
            S *= np.exp(-blr*(min(b_avg,1)-gamma))

        return grad_in.tolist(), S

        # --- TEST --- #
        grad_in = np.array(data_in).reshape((self.num_workers, -1)).mean(axis=0)
        return grad_in.tolist(), 1


if __name__ == "__main__":
    gradient_handler = AvgGradientHandler(config=config)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    print('lambda:', config.coef, '| dpoff:', config._dpoff, '| b_noise_std:', config.b_noise_std, 
        '| clip_ratio:', config.gamma, '| grad_noise_sigma:', config.grad_noise_sigma, '| dp_in:', config._dpin)
    clear_server.start()
