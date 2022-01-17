import logging
from logging import config
import random
import torch
import time
import copy
import itertools
from abc import ABCMeta, abstractmethod

from torch.optim import optimizer

from Common.Utils.evaluate import evaluate_accuracy
from Common.Utils.data_loader import load_data_noniid_mnist
#uploading gradients
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local gradients.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, test_iter, config, device, optimizer):
        # input data:
        self.train_iter = train_iter
        assert self.train_iter == None
        self.test_iter = test_iter

        # training client
        self.clients_index = []
        for i in itertools.combinations(range(0,10),7):
            self.clients_index.append(''.join(str(j) for j in i))
    
        # global model parameters:
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self._gradients = None          # not really gradients, but weights difference in one epoch
        self._weight_prev = None        # weights before this epoch
        self._weight_cur = None         # weights after this epoch
        
        # local model parameters:
        self.local_model = copy.deepcopy(self.model)
        self.local_loss_func = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), config.llr)
        self.local_minlambda = config.minLambda
        self.local_maxlambda = config.maxLambda
        self.local_lambda = self.local_minlambda

        # common parameters:
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])

        # other setting:
        self.config = config
        self.device = device
        
        # records:
        self.acc_record = [0]

    def get_gradients(self):
        """ getting gradients """
        return self._gradients
    
    def set_gradients(self, gradients):
        """ setting gradients """
        self._gradients = gradients

    def get_weights(self):
        """ getting weights/layer"""
        weights = []
        for param in self.model.parameters():
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
        idx = 0
        for param in self.model.parameters():
            tmp = self._gradients[self._level_length[idx]:self._level_length[idx + 1]]
            diff = torch.tensor(tmp, device=self.device).view(param.data.size())
            param.data = self._weight_prev[idx] + diff
            idx += 1

    def adaptive_ditto(self, return_acc, local_acc):
        self.local_lambda = min(
            max(self.local_minlambda, self.local_lambda + (return_acc - local_acc - self.local_minlambda)), 
            self.local_maxlambda
        )
    
    def train_step(self, x, y):
        """ one mini_batch training step """
        # inputs:
        x = x.to(self.device)
        y = y.to(self.device)

        # global model training:
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        # local model training:
        y_hat_local = self.local_model(x)
        loss_local = self.local_loss_func(y_hat_local, y)
        self.local_optimizer.zero_grad()
        loss_local.backward()
        layer = 0
        for param in self.local_model.parameters():
            param.grad += self.local_lambda * (param - self._weight_prev[layer])
            layer += 1
        self.local_optimizer.step()

    def fl_train(self, times):
        self.acc_record = [0]
        for epoch in range(self.config.num_epochs):
            
            if self.train_iter == None:
                _client = self.clients_index[random.randint(0, 119)]
                self.train_iter = load_data_noniid_mnist(_client, batch=128)

            self._weight_prev, batch_count, start = self.get_weights(), 0, time.time()
            if self.test_iter != None:
                return_acc = evaluate_accuracy(self.test_iter, self.model)
            for X, y in self.train_iter:
                batch_count += 1
                self.train_step(X, y)
                # if self.test_iter != None and (batch_count%10 == 0 or batch_count == len(self.train_iter)):
                if self.test_iter != None and batch_count == len(self.train_iter):
                    global_test_acc = evaluate_accuracy(self.test_iter, self.model)
                    test_acc = evaluate_accuracy(self.test_iter, self.local_model)
                    self.acc_record += [test_acc]
                    print(
                        "epoch: %d | test_acc: local: %.3f | global: [%.3f, %.3f] | Ditto: %.3f | time: %.2f | client: %s"
                        %(epoch, test_acc, global_test_acc, return_acc, self.local_lambda, time.time() - start, _client)
                    )
                    self.adaptive_ditto(return_acc, test_acc)
            self._weight_cur = self.get_weights()
            self.calculate_weights_difference()
            self.update()
            self.upgrade()

    def write_acc_record(self, fpath, info):
        s = ""
        for i in self.acc_record:
            s += str(i) + " "
        s += '\n'
        with open(fpath, 'a+') as f:
            f.write(info + '\n')
            f.write(s)
            f.write("" * 20)

    @abstractmethod
    def update(self):
        pass

