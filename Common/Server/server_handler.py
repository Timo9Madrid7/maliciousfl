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

        self.total_number = config.total_number_clients
        self.dpoff = self.config._dpoff
        self.dpcompen = self.config._dpcompen
        self.grad_noise_sigma = self.config.grad_noise_sigma
        self.b_noise_std = self.config.b_noise_std
        self.delta = self.config.delta
        self.clients_per_round = self.config.num_workers
        self.account_method = self.config.account_method
        self.weight_index = self.config.weight_index
        self.bias_index = self.config.bias_index
        self.log_moment = []

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

        # history recording
        self.counter = 0
        self.accuracy_history = []
        self.epsilon_history = []

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

        self.counter += 1
        if self.counter == self.config.num_epochs and self.config.recording:
            np.savetxt("./Eva/accuracy/acc_"+self.config.Model+'_'+self.config.surffix+".txt", self.accuracy_history)
            if self.epsilon_history != []:
                np.savetxt("./Eva/dpbudget/eps_"+self.config.Model+'_'+self.config.surffix+".txt", self.epsilon_history)

        grad_in = np.array(data_in).reshape((self.clients_per_round, -1))

        # cosine distance filtering
        if self.config.naive_aggregation:
            bengin_id = list(range(self.clients_per_round))
        else:
            bengin_id = self.cosine_distance_filter(grad_in)

        # TODO: deepsight implementation
        # with open("./Eva/deepsight/grad_ly.txt", 'a') as f:
        #     np.savetxt(f, grad_in[:,-850::])
        # neups = self.neups_metric(grad_in=grad_in)
        # with open("./Eva/deepsight/neups.txt", "a") as f:
        #     np.savetxt(f, neups)
        # ddifs = self.ddifs_metric(grad_in=grad_in, samples_size=5000)
        # with open("./Eva/deepsight/ddifs.txt", "a") as f:
        #     for ddifs_i in ddifs:
        #         np.savetxt(f, np.array(ddifs_i))

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
            self.epsilon_history.append(cur_eps)
            
            # gradients average
            grad_in = (grad_in[bengin_id].sum(axis=0) + noise_compensatory_grad) / len(bengin_id)

            # post-processing
            grad_in = self.post_clipping(grad_in, S)

            # adjustment of adaptive clipping
            b_in = np.array(b_in)[bengin_id].tolist()
            S = self.adaptive_clipping(b_in, S, gamma, blr, noise_compensatory_b)

        grad_in = grad_in.tolist()
        self.upgrade(grad_in, self.model)
        if self.test_iter != None:
            test_accuracy = evaluate_accuracy(self.test_iter, self.model)
            print("current global model accuracy: %.3f"%test_accuracy)
            self.accuracy_history.append(test_accuracy)
        return grad_in, S


    def cosine_distance_filter(self, grad_in):
        """The HDBSCAN filter based on cosine distance

        Args:
            grad_in (list/np.ndarray): the raw input weight_diffs
        """
        distance_matrix = pairwise_distances(grad_in-grad_in.mean(axis=0), metric='cosine')
        return self.hdbscan_filter(distance_matrix)

    def neups_filter(self, grad_in):
        """The HDBSCAN filter based on NEUPS

        Args:
            grad_in (list/np.ndarray): the raw input weight_diffs
        """

        neups = self.neups_metric(grad_in-grad_in.mean(axis=0))
        return self.hdbscan_filter(neups)

    def dp_noise_compensator(self, g_std, g_shape, b_std, num_used):
        """to generate the compensatory Gaussian noise when the number of used clients 
        is less than the selected number 

        Args:
            g_std(float): the standard deviation of compensatroy gradient noise
            g_shape (list/tuple): the shape of raw input weight_diffs
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
            self.log_moment.append((q, sigma, 1))
            cur_eps, cur_delta = acc_track_eps(self.log_moment, delta=config.delta)
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

    def ddifs_metric(self, grad_in:np.ndarray, samples_size=20000):
        """DDifs measures the difference of predicted scores between local update model 
        and global model as they provide information about distribution of the training 
        labels of the respective client

        Args:
            grad_in (np.ndarray): the raw input weight_diffs
            samples_size (int, optional): the number of random samples. Defaults to 20000.

        Returns:
            [list]: the DDifs for 3 different seeds as a list of 3 lists 
        """
        
        ddifs = []
        for _ in range(3):
            ddifs_i = []
            random_samples = torch.randn(samples_size, 1, 28, 28)
            for client_grad in grad_in:
                temp_model = deepcopy(self.model)
                self.upgrade(client_grad.tolist(), temp_model)
                temp_output = temp_model.forward(random_samples).detach().cpu().data
                model_output = self.model.forward(random_samples).detach().cpu().data
                neuron_diff = torch.div(temp_output, model_output).sum(axis=0)/samples_size
                ddifs_i.append(neuron_diff.numpy().tolist())
            ddifs.append(ddifs_i)
    
        return ddifs

    def neups_metric(self, grad_in:np.ndarray):
        """NEUPs measures the magnitude changes of neurons in the last layer 
        and use them to provide a rough estimation of the output labels for 
        the training data of the individual client

        Args:
            grad_in (np.ndarray): the raw input weight_diffs

        Returns:
            [np.ndarray]: 2-dimession NormalizEd Energies UPdate for clients
        """

        neups = []
        for client_grad in grad_in:
            energy_weights = client_grad[-self.weight_index:-self.bias_index].reshape((self.bias_index,-1))
            energy_bias = client_grad[-self.bias_index::]
            energy_neuron = np.abs(energy_weights).sum(axis=1) + np.abs(energy_bias)
            energy_neuron /= energy_neuron.sum()
            neups.append(energy_neuron.tolist())

        return np.array(neups)
    
    def tes_metric(self, neups):
        """TEs analyzes the parameter updates of the output layer for a model
        to measure the homogeneity of its training data

        Args:
            neups ([np.ndarray]): NormalizEd Energies UPdate

        Returns:
            [np.ndarray]: the number of threshold exceedings
        """

        tes = []
        for client_neups in neups:
            threshold = max(0.01,1/self.bias_index)*client_neups.max()
            tes.append((client_neups>threshold).sum())

        return tes

    def hdbscan_filter(self, inputs):
        self.cluster.fit(inputs)
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
    
    def upgrade(self, grad_in:list, model):
        layer = 0
        for param in model.parameters():
            layer_diff = grad_in[self._level_length[layer]:self._level_length[layer + 1]]
            param.data += torch.tensor(layer_diff, device=self.device).view(param.data.size())
            layer += 1