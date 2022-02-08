# Utils
from Common.Model.Generator import Generator

# Other Libs
import numpy as np
import torch
import copy
from scipy import special
from abc import ABCMeta, abstractmethod
import logging
logger = logging.getLogger('client.workerbase')

class WorkerBase(metaclass=ABCMeta):
    def __init__(self, model, loss_func, optimizer, train_iter, device='cpu', target=0):
        # input data:
        self.train_iter = train_iter
    
        # global model parameters:
        self.model = model.to(device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self._gradients = None          # not really gradients, but weights difference in one epoch
        self._weight_prev = None        # weights before this epoch
        self._weight_cur = None         # weights after this epoch
        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
        
        # inference target
        self.target = target
        # Generator model parameters:
        self.G_model = Generator().to(device)
        self.G_optimizer = torch.optim.SGD(self.G_model.parameters(), lr=0.05)
        # Discriminator model parameters:
        self.D_model = copy.deepcopy(model).to(device)
        self.D_optimizer = torch.optim.Adam(self.D_model.parameters(), lr=0.01)

        # other setting:
        self.device = device

    def get_weights(self):
        weights = []
        for param in self.model.parameters():
            weights.append(param.data)
        return copy.deepcopy(weights)

    def calculate_weights_difference(self):
        assert len(self._weight_prev) == len(self._weight_cur)
        self._gradients = []
        for i in range(len(self._weight_prev)):
            self._gradients += (self._weight_cur[i] - self._weight_prev[i]).data.view(-1).cpu().numpy().tolist()
    
    def train_step(self, x, y):
        # inputs:
        x = x.to(self.device)
        y = y.to(self.device)

        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.optimizer.zero_grad() 
        loss.backward()
        self.optimizer.step()

        return loss

    def genTrain(self, fake_label=1, train_batch_size=64, generate_batch_size=64, num_iter=8):
        # --- 1. Replicate the model as D
        self.D_model = copy.deepcopy(self.model).to(self.device)
        for _ in range(num_iter):
            # --- 2. Run Generator G on D targeting class
            noise_init = torch.randn(train_batch_size,100,1,1).to(self.device)    # random noise
            generated_data = self.G_model(noise_init).to(self.device) # fake images
            class_label = self.target * torch.ones(train_batch_size, dtype=torch.long).to(self.device) # inference class
            # --- 3. Update G based on the answer from D
            y_G_hat = self.D_model(generated_data) # fool Discriminator
            G_loss = self.loss_func(y_G_hat, class_label) # ->1:indistinguishable, ->0:distinguishable
            self.G_optimizer.zero_grad()
            G_loss.backward()                               
            self.G_optimizer.step()

        # y_valid_hat = special.softmax(
        #         self.D_model(self.G_model(torch.randn(1,100,1,1))).view(-1).detach().numpy()  
        #     )
        # x_, y_ = None, None
        # if y_valid_hat[0] > y_valid_hat[1:].mean():
        # --- 4. Get n-samples of class generated by G
        noise_init = torch.randn(generate_batch_size,100,1,1).to(self.device)    # random noise
        x_ = self.G_model(noise_init) # fake images
        # --- 5. Assign (fake label) to generated samples of the class
        y_ = torch.randint(1,10,(generate_batch_size,),dtype=torch.long) * torch.ones(generate_batch_size,dtype=torch.long)
        # y_ = fake_label * torch.ones(generate_batch_size,dtype=torch.long).to(self.device)
        # --- 6. Merge the generated data with the local dataset
        
        return x_.detach_(), y_.detach_(), G_loss

    def fl_train(self):
        self.G_model.load_state_dict(torch.load("./Model/LeNet/InferModelG"))
        self._weight_prev, batch_count = self.get_weights(), 0
        slot = np.random.choice(range(len(self.train_iter)), 8, replace=False)
        G_loss, poison = 0, []
        for X, y in self.train_iter:
            
            if batch_count in slot:
                # remove all belonging to target class (pretended inference)
                X = X[torch.where(y!=self.target)]
                y = y[torch.where(y!=self.target)]

                # Tanh transform
                trans = torch.nn.Tanh()
                X = trans(X)                    
                
                x_gen, y_gen, G_loss = self.genTrain()
                y_valid_hat = special.softmax(
                    self.model(self.G_model(torch.randn(1,100,1,1,device=self.device))).view(-1).detach().cpu().numpy()  
                )
                if y_valid_hat[self.target] > np.delete(y_valid_hat, self.target).mean():
                    X = torch.cat((X.to(self.device), x_gen))
                    y = torch.cat((y, y_gen))
                    poison += [1]
                # elif np.sum(y_valid_hat[self.target] < np.delete(y_valid_hat, self.target)) > 4:
                #     X = torch.cat((X.to(self.device), x_gen))
                #     y = torch.cat((y, (self.target*torch.ones(len(x_gen), dtype=torch.long)).to(self.device)))
                #     poison += [2]
                else:
                    poison += [0]

                D_loss = self.train_step(X, y)

            batch_count += 1

        print("infer_client: | G_loss: %.3f | D_loss: %.3f | P?:"%(G_loss, D_loss), poison)
        torch.save(self.G_model.state_dict(), './Model/LeNet/InferModelG')
        torch.save(self.D_model.state_dict(), './Model/LeNet/InferModelD')

        self._weight_cur = self.get_weights()
        self.calculate_weights_difference()

        return self._gradients

    @abstractmethod
    def update(self):
        pass
