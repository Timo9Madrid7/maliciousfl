# GRPC
from Common.Handler.handler import Handler

# Utils
from Common.Utils.gaussian_moments_account import AutoDP_epsilon, acc_track_eps
from Common.Utils.evaluate import evaluate_accuracy

# Other Libs
import torch
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import time
import numpy as np
from copy import deepcopy
import warnings 
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

# Settings
from Common.Utils.configSel import load_dict
import sys
config_path = load_dict("config_path")["config_path"]
sys.path.append(config_path)
import offline_config as config

class AvgGradientHandler(Handler):
    def __init__(self, config, model, device, test_iter):
        super(AvgGradientHandler, self).__init__()
        self.config = config
        self.model = model 
        self._level_length = [0]
        for _, param in self.model.state_dict().items():
            self._level_length.append(param.data.numel() + self._level_length[-1])
        self.device = device
        self.test_iter = test_iter

        self.total_number = self.config.total_number_clients
        self.dpoff = self.config._dpoff
        self.clip_bound =  None if self.dpoff else self.config.initClippingBound
        self.grad_noise_sigma = self.config.grad_noise_sigma
        self.b_noise_std = self.config.b_noise_std
        self.delta = self.config.delta
        self.clients_per_round = self.config.num_workers
        self.account_method = self.config.account_method
        self.weight_index = self.config.weight_index
        self.bias_index = self.config.bias_index
        self.log_moment = []

        self.cluster_base = hdbscan.HDBSCAN(
            metric='l2', 
            min_cluster_size=2, # the smallest size grouping that you wish to consider a cluster
            allow_single_cluster=True, # False performs better in terms of Backdoor Attack
            min_samples=2, # how conservative you want you clustering to be
            cluster_selection_epsilon=0.1,
        )
        self.cluster_lastLayer = hdbscan.HDBSCAN(
            metric='l2', 
            min_cluster_size=2,
            allow_single_cluster=True,
            min_samples=1,
        )
        
        # moments_account:
        if self.grad_noise_sigma:
            # self.sigma = (self.grad_noise_sigma**(-2) + (2*self.b_noise_std)**(-2))**(-0.5)
            self.sigma = self.grad_noise_sigma
        else:
            self.sigma = None
        self.track_eps = AutoDP_epsilon(self.delta)

        # history recording
        self.counter = 0
        self.accuracy_history = []
        self.epsilon_history = []

    def cosine_distance_filter(self, distance_matrix, cluster_sel=0):
        return self.hdbscan_filter(distance_matrix, cluster_sel=cluster_sel)

    def globalmodel_update(self, grad_in):
        self.upgrade(grad_in, self.model)
        test_accuracy = None
        if self.test_iter != None:
            test_accuracy = evaluate_accuracy(self.test_iter, self.model)
            self.accuracy_history.append(test_accuracy)
        return test_accuracy

    def get_clipBound(self):
        return self.clip_bound

    def update_clipBound(self, bs_avg, verbose=False):
        if not self.dpoff:
             self.clip_bound = (self.clip_bound * np.exp(-self.config.blr*(min(bs_avg,1)-self.config.gamma))).item()
        if verbose:
            print("next round clipping boundary: %.2f"%self.clip_bound)

    def hdbscan_filter(self, inputs, cluster_sel=0):
        if cluster_sel == 0:
            cluster = self.cluster_base
        elif cluster_sel == 1:
            cluster = self.cluster_lastLayer
        cluster.fit(inputs)
        label = cluster.labels_
        if (label==-1).all():
            bengin_id = np.arange(len(inputs)).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label==majority)[0].tolist()

        return bengin_id
    
    def upgrade(self, grad_in:list, model):
        layer = 0
        for _, param in model.state_dict().items():
            layer_diff = grad_in[self._level_length[layer]:self._level_length[layer + 1]]
            param.data += torch.tensor(layer_diff, device=self.device).view(param.data.size())
            layer += 1

    def add_dpNoise(self, grads_sum, bs_sum, num_used, verbose=True):
        if not self.dpoff:
            grads_sum += torch.normal(mean=0, std=self.grad_noise_sigma*self.clip_bound, size=grads_sum.shape)
            # bs_sum += np.random.normal(0, self.b_noise_std)
            if self.sigma != None:
                cur_eps, cur_delta = self.dp_budget_trace(num_used/self.total_number, self.sigma, account_method=self.account_method)
                if verbose:
                    print("epsilon: %.2f | delta: %.6f | clipB: %.2f"%(cur_eps, cur_delta, self.clip_bound))
                self.epsilon_history.append(cur_eps)
                grads_avg = grads_sum/num_used
            if torch.norm(grads_avg) > self.clip_bound: # post-processing
                grads_avg *= self.clip_bound/torch.norm(grads_avg)
            return grads_avg, bs_sum/num_used
        return grads_sum/num_used, bs_sum/num_used

    def dp_budget_trace(self, q, sigma, account_method):
        if account_method == "googleTF": # Gaussian Moments Accountant
            self.log_moment.append((q, sigma, 1))
            cur_eps, cur_delta = acc_track_eps(self.log_moment, delta=config.delta)
        elif account_method == "autodp": # Renyi Differential Privacy
            self.track_eps.update_mech(q, sigma, 1)
            cur_eps, cur_delta = self.track_eps.get_epsilon(), config.delta
        return cur_eps, cur_delta

    def save_dpHist(self):
        with open(config_path+"DP_history.txt", 'a') as f:
            f.write(''.join(str(i)+' ' for i in self.epsilon_history)+'\n')