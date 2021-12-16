from copy import deepcopy
import logging
import torch
import time
import copy
from abc import ABCMeta, abstractmethod

from Common.Model.Generator import Generator
from scipy import special

logger = logging.getLogger('client.workerbase')

class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, train_iter, test_iter, config, device, optimizer):
        # input data:
        self.train_iter = train_iter
        self.test_iter = test_iter
    
        # global model parameters:
        self.model = model
        # self.loss_func = loss_func
        self.loss_func = loss_func
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self._gradients = None          # not really gradients, but weights difference in one epoch
        self._weight_prev = None        # weights before this epoch
        self._weight_cur = None         # weights after this epoch
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
        
        # Generator model parameters:
        self.G_model = Generator().to(torch.device('cpu' if torch.cuda.is_available() else 'cpu'))
        self.G_optimizer = torch.optim.Adam(self.G_model.parameters(), lr=0.05)
        # Discriminator model parameters:
        self.D_model = copy.deepcopy(model)
        self.D_optimizer = torch.optim.Adam(self.D_model.parameters(), lr=0.01)
        self.infer_loss_func = torch.nn.BCEWithLogitsLoss()

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
            weights.append(param)
        return weights

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
    
    def train_step(self, x, y, epoch):
        """ one mini_batch training step """
        # inputs:
        x = x.to(self.device)
        y = y.to(self.device)
        train_batch_size = 64
        generate_batch_size = 16
        # remove all belonging to target class (pretended inference)
        # x = x[torch.where(y!=0)]
        # y = y[torch.where(y!=0)]
        # Tanh transform
        trans = torch.nn.Tanh()
        x = trans(x)

        if epoch < 2:
            # global model training:
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()
            G_loss = 0.
        else:
            # inference model training:
            # --- 1. Replicate the model as D
            self.D_model = copy.deepcopy(self.model)
            for _ in range(10):
                # --- 2. Run Generator G on D targeting class
                noise_init = torch.randn(train_batch_size,100,1,1)    # random noise
                generated_data = self.G_model(noise_init).to(self.device) # fake images
                class_label = 0 * torch.ones(train_batch_size, dtype=torch.long).to(self.device) # inference class
                # class_label = torch.ones(train_batch_size, dtype=torch.float).to(self.device) # inference class
                # --- 3. Update G based on the answer from D
                y_G_hat = self.D_model(generated_data) # fool Discriminator
                G_loss = self.loss_func(y_G_hat, class_label) # ->1:indistinguishable, ->0:distinguishable
                # G_loss = self.infer_loss_func(y_G_hat[:,1], class_label)  
                self.G_optimizer.zero_grad()
                G_loss.backward()                               
                self.G_optimizer.step()
            
            # y_valid_hat = special.softmax(
            #     self.model(self.G_model(torch.randn(1,100,1,1))).view(-1).detach().numpy()  
            # )
                  
            # --- 4. Get n-samples of class generated by G
            noise_init = torch.randn(generate_batch_size,100,1,1)    # random noise
            x_ = self.G_model(noise_init).to(self.device) # fake images
            # --- 5. Assign (fake label) to generated samples of the class
            y_ = torch.randint(1,10,(generate_batch_size,),dtype=torch.long) * torch.ones(generate_batch_size,dtype=torch.long).to(self.device)
            # --- 6. Merge the generated data with the local dataset
            x = torch.cat((x,x_))
            y = torch.cat((y,y_))

            # global model training:
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            self.optimizer.zero_grad() 
            loss.backward()
            self.optimizer.step()

        return G_loss, loss


    def fl_train(self, times):
        for epoch in range(self.config.num_epochs):
            self._weight_prev, batch_count = self.get_weights(), 0
            for X, y in self.train_iter:
                batch_count += 1
                G_loss, loss = self.train_step(X, y, epoch)
                print("epoch: %d | G_loss: %.3f | D_loss: %.3f"%(epoch, G_loss, loss))
            self._weight_cur = self.get_weights()
            self.calculate_weights_difference()
            self.update()
            self.upgrade()
            # print("epoch: %d | G_loss: %.3f | D_loss: %.3f"%(epoch, G_loss, loss))
            torch.save(self.G_model.state_dict(), './Model/InferModelG')
            torch.save(self.D_model.state_dict(), './Model/InferModelD')

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

