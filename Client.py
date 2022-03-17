from LeNet import LeNet
from Common.Utils.data_loader import load_data, load_test
from Common.Utils.evaluate import evaluate_accuracy
from config import locally_encode_layer
import torch 
import copy

class Client():
    def __init__(self, client_id, device):
        self.client_id = str(client_id)
        self.device = device 
        # client 0 - 4: have 0s in traning sets, client 5 - 9: don't have 0s in traning sets
        self.train_iter = load_data(self.client_id)
        # only client_0 uses all digits testing set, other clients use all 0s testing set
        self.test_iter = load_test() if self.client_id == "0" else load_data('0', path="./Data/digit_")
        self.model = LeNet().to(self.device)
        self.model.load_state_dict(torch.load("./Models/LeNet_"+self.client_id))
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self._level_length = [0]
        for param in self.model.parameters():
            self._level_length.append(param.data.numel() + self._level_length[-1])
        self.start_index = self._level_length[locally_encode_layer]

    def get_weights(self, model):
        weights = []
        for param in model.parameters():
            weights += param.data.view(-1).cpu().numpy().tolist()
        return copy.deepcopy(weights[self.start_index:])

    def upgrade(self, global_model_path="./Models/LeNet"):
        global_model = LeNet().to(self.device)
        global_model.load_state_dict(torch.load(global_model_path))
        global_layers_parameters =  self.get_weights(global_model)
        layer = 0
        for param in self.model.parameters():
            if layer >= locally_encode_layer:
                param_avg = global_layers_parameters[self._level_length[layer]-self.start_index:self._level_length[layer + 1]-self.start_index]
                param.data = torch.tensor(param_avg, device=self.device).view(param.data.size())
            layer += 1    

    def store_local(self, local_models_path="./Models/LeNet_"):
        torch.save(self.model.state_dict(), local_models_path+self.client_id)

    def fl_train(self, local_epoch=1, verbose=True):
        self.upgrade() # update the common layers parameters from the server
        
        for epoch in range(local_epoch):
            for X, y in self.train_iter:
                
                X,y = X.to(self.device), y.to(self.device)
                y_hat = self.model(X)
                loss = self.loss_func(y_hat, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        if verbose:
            print("client_id: %s | loss: %.2f |"%(self.client_id, loss), end=" ")
            if self.test_iter != None:
                accuracy = evaluate_accuracy(self.test_iter, self.model, device=self.device)
                print("acc: %.2f"%accuracy, end=" ")
            print()

        self.store_local()
        return self.get_weights(self.model)