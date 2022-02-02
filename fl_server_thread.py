# GRPC
from Common.Server.fl_grpc_server_adaclipping import FlGrpcServer as FLGrpcClipServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_Clipping
from Common.Handler.handler import Handler

# Utils
from Common.Utils.gaussian_moments_account import AutoDP_epsilon, acc_track_eps

# Settings
import Common.config as config

# Other Libs
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import numpy as np
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
        self.dpcompen = self.config._dpcompen
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
        """the averaging process of the aggregator

        Args:
            data_in (list): collected gradients 
            b_in (list): collected indicators
            S (float): clipping boundary
            gamma (float): clipping ratio
            blr (float): adaptive clipping learning rate

        Returns:
            [list, float]: averaged gradients and next round clipping boundary
        """
        grad_in = np.array(data_in).reshape((self.clients_per_round, -1))
        
        # cosine distance filtering
        bengin_id = self.cosine_distance_filter(grad_in)

        if self.dpoff:
            # naive gradient average
            grad_in = grad_in[bengin_id].mean(axis=0)
            S = 0
            print("clip_bound: inf | used id: ", bengin_id)
        else:
            # noise compensation
            if self.dpcompen:
                noise_compensatory_grad, noise_compensatory_b = self.dp_noise_compensator(
                    g_std=self.grad_noise_sigma * S,
                    g_shape=grad_in.shape[1],
                    b_std=self.b_noise_std,
                    num_used=len(bengin_id)
                )
                sigma = self.sigma
            else:
                noise_compensatory_grad, noise_compensatory_b = 0, 0
                sigma = self.sigma * np.sqrt(len(bengin_id)/self.clients_per_round)

            # moment accountant
            if self.sigma != None:
                cur_eps, cur_delta = self.dp_budget_trace(
                    q=len(bengin_id)/self.total_number, 
                    sigma=sigma, 
                    account_method=self.account_method)
                print("epsilon: %.2f | delta: %.6f | "%(cur_eps, cur_delta), end="")
            print("clip_bound: %.3f | used id: "%S, bengin_id)
            
            # gradients average
            grad_in = (grad_in[bengin_id].sum(axis=0) + noise_compensatory_grad) / len(bengin_id)

            # post-processing
            grad_in = self.post_clipping(grad_in, S)

            # adjustment of adaptive clipping
            b_in = np.array(b_in)[bengin_id].tolist()
            S = self.adaptive_clipping(b_in, S, gamma, blr, noise_compensatory_b)

        return grad_in.tolist(), S


    def cosine_distance_filter(self, grad_in):
        """The HDBSCAN filter based on cosine distance

        Args:
            grad_in (list/np.ndarray): the raw input gradients/weight_diffs
        """
        distance_matrix = pairwise_distances(grad_in-grad_in.mean(axis=0), metric='cosine')
        self.cluster.fit(distance_matrix)
        label = self.cluster.labels_
        if (label==-1).all():
            bengin_id = np.arange(self.clients_per_round).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label==majority)[0].tolist()

        return bengin_id

    def dp_noise_compensator(self, g_std, g_shape, b_std, num_used):
        """to generate the compensatory Gaussian noise when the number of used clients 
        is less than the selected number 

        Args:
            g_std(float): the standard deviation of compensatroy gradient noise
            g_shape (list/tuple): the shape of raw input gradients/weight_diffs
            b_std(float): the standard deviation of compensatroy indicator noise
            num_used (int): the number of used/benign clients
        """

        extra_grad_noise_std = g_std * np.sqrt(1-num_used/self.clients_per_round)
        noise_compensatory_grad = np.random.normal(0, extra_grad_noise_std, size=g_shape)

        extra_b_noise_std = b_std * np.sqrt(1-num_used/self.clients_per_round)
        noise_compensatory_b = np.random.normal(0, extra_b_noise_std)

        return noise_compensatory_grad, noise_compensatory_b
    
    def dp_budget_trace(self, q, sigma, account_method):
        """to monitor the accountant for differential privacy budget  

        Args:
            q (float): the fraction of random selections used for this round
            sigma (float): the gradient noise multiplier (std=sigma*clipping_boundary)
            account_method (str): the method of accountant
        """

        if account_method == "googleTF": # Gaussian Moments Accountant
            self.acc_params.append((q, sigma, 1))
            cur_eps, cur_delta = acc_track_eps(self.acc_params, delta=config.delta)
        elif account_method == "autodp": # Renyi Differential Privacy
            self.track_eps.update_mech(q, sigma, 1)
            cur_eps, cur_delta = self.track_eps.get_epsilon(), config.delta

        return cur_eps, cur_delta

    def post_clipping(self, grad_in, clip_bound):
        """post processing clipping for averaged grad_in

        Args:
            grad_in (np.ndarray): the averaged grad_in
            clip_bound (float): the current clipping boundary
        """

        grad_in_l2_norm = np.linalg.norm(grad_in)
        if grad_in_l2_norm > clip_bound:
            grad_in *= clip_bound/grad_in_l2_norm

        return grad_in

    def adaptive_clipping(self, b_in, clip_bound, clip_ratio, lr, noise_compensatory_b=0.):
        """to adjust the clipping boundary for the next round according to the indicators

        Args:
            b_in (list): the list of indicators
            clip_bound (float): the current clipping boundary
            clip_ratio (float): the clipping ratio (a number between [0,1])
            lr (float): the adaptive clipping learning rate
            noise_compensatory_b (float, optional): noise compensation. Defaults to 0.
        """
        
        b_in = list(map(lambda x: max(0,x), b_in))
        b_avg = (np.sum(b_in) + noise_compensatory_b) / len(b_in)
        clip_bound *= np.exp(-lr*(min(b_avg,1)-clip_ratio))

        return clip_bound


if __name__ == "__main__":
    gradient_handler = AvgGradientHandler(config=config)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    print('ratio %d/%d:'%(config.num_workers, config.total_number_clients), '| dpoff:', config._dpoff, ' | dpcompen:', config._dpcompen,
    '| b_noise_std:', config.b_noise_std, '| clip_ratio:', config.gamma, '| grad_noise_sigma:', config.grad_noise_sigma)
    clear_server.start()
