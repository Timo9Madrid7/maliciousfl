import logging
from logging import config
import random
from select import select
import torch
import numpy as np
import time
import copy
import itertools
from abc import ABCMeta, abstractmethod

from torch.optim import optimizer

from Common.Utils.evaluate import evaluate_accuracy
from Common.config import _dpin, _dpclient, _dprecord, _noniid
#uploading gradients
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local gradients.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, thread_id, test_iter, train_iter_loader, dittoEval_loader, different_client_loader, model, loss_func, config, device, optimizer):
        # input data:
        self.thread_id = thread_id
        self.train_iter_loader = train_iter_loader
        self.train_iter = None
        self.test_iter = test_iter
        self.dittoEval_loader = dittoEval_loader
        self.local_test_iter = None
        self.different_client_loader = different_client_loader
        self.client = ""

        # training client
        if _noniid:
            self.clients_index = []
            for i in itertools.combinations(range(0,10),7):
                self.clients_index.append(''.join(str(j) for j in i))
        else:
            self.clients_index = list(str(i) for i in range(config.total_number_clients))

        # dp test acc record
        self.acc_dp = []
        self.dpclient_iter = self.different_client_loader(_dpclient, noniid=_noniid)
    
        # global model parameters:
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer
        self._gradients = None          # not really gradients, but weights difference in one epoch
        self._weight_prev = None        # weights before this epoch
        self._weight_cur = None         # weights after this epoch
        self._gradients_list = []
        self.global_lambda = 0 # similarity: global to local

        # local model parameters:
        self.local_model = copy.deepcopy(self.model)
        self.local_loss_func = torch.nn.CrossEntropyLoss()
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), config.llr)
        self._weight_local = None # weight paramters used for global model L2-norm
        self.local_minlambda = config.minLambda
        self.local_maxlambda = config.maxLambda
        self.local_lambda = self.local_minlambda # similarity: local to global

        # common parameters:
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])

        # other setting:
        self.config = config
        self.device = device
        
    def get_gradients(self):
        """ getting gradients """
        return self._gradients
    
    def set_gradients(self, gradients):
        """ setting gradients """
        self._gradients = gradients

    def get_weights(self):
        """ getting weights per layer"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.data)
        return copy.deepcopy(weights)

    def get_local_weights(self):
        """ getting local weights per layer"""
        weights = []
        for param in self.local_model.parameters():
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
            param.grad += self.global_lambda * (param - self._weight_local[layer])
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
            param.grad += self.local_lambda * (param - self._weight_prev[layer])
            layer += 1
        self.local_optimizer.step()

    def select_client(self):
        id = random.randint(0, self.config.total_number_clients-1)
        self.local_model.load_state_dict(torch.load("./Model/Local_Models/LeNet_"+str(id))) 
        _client = self.clients_index[id]
        if _dpin and _client == _dpclient:
            self.train_iter = self.different_client_loader(_client, noniid=_noniid)
        else:
            self.train_iter = self.train_iter_loader(_client, noniid=_noniid)
        self.local_test_iter = self.dittoEval_loader(_client, noniid=_noniid)
    
        return _client, id

    def upgrade_local(self, id):
        torch.save(self.local_model.state_dict(), "./Model/Local_Models/LeNet_"+str(id))

    def fl_train(self):
        for epoch in range(self.config.num_epochs): # number of total epochs 
            # randomly pick up a client from non iid data set
            # if _dpin is true, client [may] select that dpclient
            self.client, model_id = self.select_client()
            
            # retrieve weights from the global model
            self._weight_prev, batch_count, start = self.get_weights(), 0, time.time()
            # retrieve local weights from the local model
            self._weight_local = self.get_local_weights()
            if epoch > int(self.config.num_epochs/2): #(!issue)
            # global model only learns from the local when local accuracy is acceptable
                self.global_lambda = self.config.global_lambda

            if self.thread_id == 0: # thread_0 responds for recording
                self.acc_dp.append(evaluate_accuracy(self.dpclient_iter, self.model))

            # ditto reference accuracy
            return_acc = evaluate_accuracy(self.local_test_iter, self.model)
            lambda_list = [self.local_lambda]

            # global & local models training
            for X, y in self.train_iter:
                batch_count += 1
                
                # training
                self.train_step(X, y)
                # evaluation
                local_test_acc = evaluate_accuracy(self.local_test_iter, self.local_model)
                # adjust the local-global distance
                self.adaptive_ditto(return_acc, local_test_acc)
                # record the lambda
                lambda_list.append(self.local_lambda)

                # verbose at last, thread_0 responds for recording
                if batch_count == len(self.train_iter) and self.thread_id == 0:
                    test_acc = evaluate_accuracy(self.test_iter, self.model)
                    print(
                        "epoch: %d | local_acc: %.3f | global_acc: %.3f | Ditto: L %.3f, G %.3f | time: %.2f | client: %s"
                        %(epoch, local_test_acc, test_acc, np.mean(lambda_list), self.global_lambda, time.time() - start, self.client)
                    )
            
            # global model update
            self._weight_cur = self.get_weights()
            self.calculate_weights_difference()
            self.update(model_id=model_id)
            self.upgrade()
            # local model update
            self.upgrade_local(id=model_id)
            
        if _dprecord and self.acc_dp != []:
            self.write_dp_acc_record()

    def write_dp_acc_record(self):
        np.savetxt("./Eva/dp_test_acc/dpclient_"+_dpclient+".txt", self.acc_dp)

    @abstractmethod
    def update(self, model_id):
        pass

