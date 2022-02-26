# GRPC
from Common.Handler.handler import Handler

# Utils
from Common.Utils.gaussian_moments_account import AutoDP_epsilon, acc_track_eps
from Common.Utils.evaluate import evaluate_accuracy

# Settings
import OfflinePack.offline_config as config

# Other Libs
import torch
from sklearn.metrics.pairwise import pairwise_distances
import hdbscan
import time
import numpy as np
from copy import deepcopy
import warnings 
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

class AvgGradientHandler(Handler):
    def __init__(self, config, model, device, test_iter):
        super(AvgGradientHandler, self).__init__()
        self.config = config
        self.model = model 
        self._level_length = [0]
        for param in self.model.parameters():
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
            self.sigma = (self.grad_noise_sigma**(-2) + (2*self.b_noise_std)**(-2))**(-0.5)
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
            bengin_id = np.arange(self.clients_per_round).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label==majority)[0].tolist()

        return bengin_id
    
    def upgrade(self, grad_in:list, model):
        layer = 0
        for param in model.parameters():
            layer_diff = grad_in[self._level_length[layer]:self._level_length[layer + 1]]
            param.data += torch.tensor(layer_diff, device=self.device).view(param.data.size())
            layer += 1