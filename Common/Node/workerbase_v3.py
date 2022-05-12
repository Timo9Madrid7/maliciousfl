# Utils
from Common.Utils.evaluate import evaluate_accuracy

# Other Libs 
from sklearn.metrics.pairwise import pairwise_distances
import torch
import numpy as np
import time
import copy
from abc import ABCMeta, abstractmethod
import logging
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local weights' differences.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(
        self, 
        client_id, train_iter, eval_iter,
        model, loss_func, optimizer,
        local_model, local_loss_func, local_optimizer,
        config, device):

        # input data:
        self.client_id = client_id
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.device = device
        self.config = config
    
        # global model parameters:
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self._gradients = None                                  # not really gradients, but weights difference in one epoch
        self._grad_weight_prev, self._weight_prev = None, None # weights before this epoch
        self._weight_cur = None                                 # weights after this epoch
        self._grad_weight_local = None                          # weights of the local model
        self.global_lambda = self.config.global_lambda

        # local model parameters:
        self.local_model = local_model
        self.local_loss_func = local_loss_func
        self.local_optimizer = local_optimizer
        self._weight_local = None # weight paramters used for global model L2-norm
        self.local_minlambda = config.minLambda
        self.local_maxlambda = config.maxLambda
        self.local_lambda = self.local_minlambda # similarity: local to global

        # common parameters:
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
        
    def get_gradients(self):
        """ getting gradients """
        return self._gradients
    
    def set_gradients(self, gradients):
        """ setting gradients """
        self._gradients = gradients

    def get_grad_weights(self, model="global"):
        """ getting weights per layer"""
        if model == "local":
            model_paramters = self.local_model.parameters()
        else:
            model_paramters = self.model.parameters()
        weights = []
        for param in model_paramters:
            weights.append(param.data)
        return copy.deepcopy(weights)

    def get_all_weights(self, model="global"):
        """ getting weights per layer"""
        if model == "local":
            model_dict = self.local_model.state_dict()
        else:
            model_dict = self.model.state_dict()
        weights = []
        for _, param in model_dict.items():
            weights.append(param.data)
        return copy.deepcopy(weights)

    def calculate_weights_difference(self):
        """ calculating the epoch weights difference which will be uploaded to the server"""
        assert len(self._weight_prev) == len(self._weight_cur)
        self._gradients = []
        for i in range(len(self._weight_prev)):
            self._gradients += (self._weight_cur[i] - self._weight_prev[i]).data.view(-1).cpu().numpy().tolist()

    def upgrade(self):
        """ Use the processed gradient to update the weights """
        layer = 0
        for param in self.model.parameters():
            tmp = self._gradients[self._level_length[layer]:self._level_length[layer + 1]]
            diff = torch.tensor(tmp, device=self.device).view(param.data.size())
            param.data = self._weight_prev[layer] + diff
            layer += 1

    def adaptive_ditto(self, return_acc, local_acc, threshold=0.05, lr=1):
        self.local_lambda = min(
            max(self.local_minlambda, self.local_lambda + lr * (return_acc - local_acc - threshold)), 
            self.local_maxlambda
        )
    
    def train_step(self, x_global, y_global, x_local=None, y_local=None):
        """ one mini_batch training step """
        # global inputs:
        x = x_global.to(self.device)
        y = y_global.to(self.device)

        # global model training:
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad() 
        loss.backward()
        layer = 0
        for param in self.model.parameters():
            param.grad += self.global_lambda * (param - self._grad_weight_local[layer])
            layer += 1 
        self.optimizer.step()

        # local inputs:
        if x_local != None and y_local != None:
            x = x_local.to(self.device)
            y = y_local.to(self.device)
        else:
            x = x.detach()
            y = y.detach()

        # local model training:
        y_hat_local = self.local_model(x)
        loss_local = self.local_loss_func(y_hat_local, y)
        self.local_optimizer.zero_grad()
        loss_local.backward()
        layer = 0
        for param in self.local_model.parameters():
            param.grad += self.local_lambda * (param - self._grad_weight_prev[layer])
            layer += 1
        self.local_optimizer.step()

    def fl_train(self, local_epoch=1, verbose=False):
        # retrieve weights from the global model
        self._grad_weight_prev = self.get_grad_weights(model="global")
        self._weight_prev = self.get_all_weights(model="global")
        # ditto reference accuracy
        return_acc = evaluate_accuracy(self.eval_iter, self.model)
        lambda_list = [self.local_lambda]

        batch_count, start = 0, time.time()
        for epoch in range(local_epoch): # number of local epochs 
            # retrieve local weights from the local model
            self._grad_weight_local = self.get_grad_weights(model="local")

            # global & local models training
            for X, y in self.train_iter:
                batch_count += 1
                # training
                self.train_step(X, y)
                # adjusting ditto lambda
                local_test_acc = evaluate_accuracy(self.eval_iter, self.local_model)
                self.adaptive_ditto(return_acc, local_test_acc)
                lambda_list.append(self.local_lambda)

        if verbose:
            print(
                "client_id: %s | local_acc: %.3f | prior: %.3f, post: %.3f (local_test_set) | Ditto: L %.3f, G %.3f | time: %.2f"
                %(self.client_id, local_test_acc, return_acc, evaluate_accuracy(self.eval_iter, self.model), np.mean(lambda_list), self.global_lambda, time.time() - start)
            )
            
        self._weight_cur = self.get_all_weights(model="global")
        self.calculate_weights_difference()
        self.upgrade_local()
        self.model.zero_grad(), self.local_model.zero_grad()
        return self._gradients

        clip_bound = self.update()
        self.upgrade()

        return clip_bound

    def show_similarity(self):
        """this function is only used for debugging"""
        global_weights = []
        local_weights = []
        for global_param, local_param in zip(self.model.parameters(), self.local_model.parameters()):
            global_weights += global_param.data.view(-1).cpu().numpy().tolist()
            local_weights += local_param.data.view(-1).cpu().numpy().tolist()

        l2 = pairwise_distances([global_weights], [local_weights], metric="euclidean")
        cos = pairwise_distances([global_weights], [local_weights], metric="cosine")
        return l2.item(), cos.item()

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def upgrade_local(self):
        pass

    @abstractmethod
    def evaluation(self):
        pass