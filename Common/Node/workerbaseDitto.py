import logging
from logging import config
import torch
import time
import copy
from abc import ABCMeta, abstractmethod

from Common.Utils.evaluate import evaluate_accuracy
#uploading gradients
logger = logging.getLogger('client.workerbase')

'''
This is the worker for sharing the local gradients.
'''
class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, test_iter, config, device, optimizer):
        self.model = model
        self.local_model = copy.deepcopy(self.model) # local model
        self.loss_func = loss_func

        self.train_iter = train_iter
        self.test_iter = test_iter

        self.config = config
        self.optimizer = optimizer
        self.local_optimizer = torch.optim.Adam(self.local_model.parameters(), config.llr) # local optimizer

        self.acc_record = [0]

        self.device = device
        self._level_length = None
        self._grad_len = 0
        self._gradients = None
        self._local_gradients = None # local gradients
        
    def get_gradients(self):
        """ getting gradients """
        return self._gradients
    
    def set_gradients(self, gradients):
        """ setting gradients """
        self._gradients = gradients
    

    def train_step(self, x, y):
        """ Find the update gradient of each step in collaborative learning """
        x = x.to(self.device)
        y = y.to(self.device)
        
        y_hat = self.model(x) # forward
        loss = self.loss_func(y_hat, y) # loss calculation
        # every epoch uses the new calculated gradients, 
        # rather than the accumulated ones
        self.optimizer.zero_grad() 
        loss.backward() # backward

        self._gradients = []
        self._level_length = [0]
        

        # to store the gradients layer by layer,
        # so both shapes and values should be stored
        for param in self.model.parameters():
            self._level_length.append(param.grad.numel() + self._level_length[-1])
            # update the gradients after the backward propagation
            # these gradients will be uploaded before the next epoch
            self._gradients += param.grad.view(-1).cpu().numpy().tolist()

        self._grad_len = len(self._gradients)
            
    def upgrade(self):
        """ Use the processed gradient to update the gradient """
        # to update(replace) the gradients layer by layer
        idx = 0
        for param in self.model.parameters():
            # the gradients are flattened, so they should be restored sequentially
            tmp = self._gradients[self._level_length[idx]:self._level_length[idx + 1]]
            grad_re = torch.tensor(tmp, device=self.device)
            grad_re = grad_re.view(param.grad.size())

            param.grad = grad_re
            idx += 1

        # update all parameters
        self.optimizer.step()

    def local_train(self, x, y):
        # extract global weights
        global_weight = []
        for param in self.model.parameters():
            global_weight.append(param)
        
        # train the local model
        x = x.to(self.device)
        y = y.to(self.device)
        y_hat = self.local_model(x) # deepcopy from global model
        loss = self.loss_func(y_hat, y) # identical loss_func
        self.local_optimizer.zero_grad() # identical optimizer
        loss.backward()

        # update the local model
        ly = 0
        for param in self.local_model.parameters():
            param.grad += self.config.coef * (param - global_weight[ly])
            ly += 1
        self.local_optimizer.step()

        # return the local evaluation
        return loss.cpu().item(), y_hat


    def fl_train(self, times):
        self.acc_record = [0]
        counts = 0
        for epoch in range(self.config.num_epochs):
            train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
            for X, y in self.train_iter:
                counts += 1

                # times = 1, so this part will not execute under the default setting
                if (counts % times) != 0:
                    X = X.to(self.device)
                    y = y.to(self.device)
                    y_hat = self.model(X)
                    l = self.loss_func(y_hat, y)
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    train_l_sum += l.cpu().item()
                    train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                    n += y.shape[0]
                    batch_count += 1
                    continue

                self.train_step(X, y) # forward&backward propagation
                self.update() # upload&download the gradients
                self.upgrade() # push the global gradients into the model
                loss, y_hat = self.local_train(X, y) # update local model
                train_l_sum += loss
                train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
                n += y.shape[0]
                batch_count += 1

                if self.test_iter != None:
                    # evaluation
                    test_acc = evaluate_accuracy(self.test_iter, self.local_model) # test by local model
                    self.acc_record += [test_acc]
                  #   print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
                  # % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
                    print(test_acc)

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

