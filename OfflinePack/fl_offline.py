# Utils
from Common.Node.workerbase_v3 import WorkerBase as WorkerBaseDitto
from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_noniid_mnist, load_data_dittoEval_mnist, load_all_test_mnist
from Common.Utils.data_loader import load_data_noniid_cifar10, load_data_dittoEval_cifar10
from Common.Utils.evaluate import evaluate_accuracy

# Settings
import Common.offline_config as config

# Other Libs
import torch
import numpy as np 
import random

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class OfflineClient(WorkerBaseDitto):
    def __init__(
        self, client_id, train_iter, eval_iter, 
        model, loss_func, optimizer, 
        local_model, local_loss_func, local_optimizer,
        config, device, 
        clippingBound):
        super().__init__(client_id, train_iter, eval_iter, model, loss_func, optimizer, local_model, local_loss_func, local_optimizer, config, device)

        self.dpoff = self.config._dpoff
        self.clippingBound = clippingBound # initial clipping bound for a client
        self.grad_noise_sigma = self.config.grad_noise_sigma
        self.b_noise_std = self.config.b_noise_std
        self.clients_per_round = self.config.num_workers

    def adaptiveClipping(self, input_gradients):
        '''
        To clip the input gradient, if its norm is larger than the setting
        '''
        if self.dpoff: # don't clip
            return np.array(input_gradients).tolist(), 1
        # else: do clipping+noising
        gradients = np.array(input_gradients)
        # --- adaptive noise calculation ---
        if self.grad_noise_sigma==0:
            grad_noise = 0
            b_noise = 0
        else: 
            grad_noise_std = self.grad_noise_sigma * self.clippingBound # deviation for gradients
            b_noise = np.random.normal(0, self.b_noise_std/np.sqrt(self.clients_per_round))
            grad_noise = np.random.normal(0, grad_noise_std/np.sqrt(self.clients_per_round), size=gradients.shape)
        # --- adaptive noise calculation ---
        norm = np.linalg.norm(gradients)
        if norm > self.clippingBound:
            return (gradients * self.clippingBound/np.linalg.norm(gradients) + grad_noise).tolist(), 0 + b_noise
        else:
            return (gradients + grad_noise).tolist(), 1 + b_noise

    def update(self):
        pass

    def evaluation(self):
        if self.thread_id == 0:
           return evaluate_accuracy(self.debug_test_iter, self.model)
    
    def upgrade_local(self):
        torch.save(self.local_model.state_dict(), self.config.local_models_path+self.client_id)
    
    def malicious_random_upload(self, model="LeNet"):
        pass


def upgrade(self):
        """ Use the processed gradient to update the weights """
        layer = 0
        for param in self.model.parameters():
            tmp = self._gradients[self._level_length[layer]:self._level_length[layer + 1]]
            diff = torch.tensor(tmp, device=self.device).view(param.data.size())
            param.data = self._weight_prev[layer] + diff
            layer += 1

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'gpu')
    global_model = LeNet.to(device)
    global_optimizer = torch.optim.Adam(global_model.parameters(), lr=0.01)
    global_loss_func = torch.nn.CrossEntropyLoss()
    initClippingBound = config.initClippingBound

    for epoch in range(config.num_epochs):
        print("epoch %d started, %d out of %d clients selected"
            %(epoch, config.num_workers, config.total_number_clients))

        client_ids_ = random.sample(range(120), config.num_workers)

        b_list_ = []
        grads_list_ = []

        for client_id in client_ids_:
            train_iter = load_data_noniid_mnist(client_id, noniid=config._noniid)
            eval_iter = load_data_dittoEval_mnist(client_id, noniid=config._noniid)
            local_model = LeNet().to(device)
            local_model.load_state_dict(config.local_models_path+client_id)
            local_optimizer = torch.optim.Adam(local_model.parameters(), config.llr)
            local_loss_func = torch.nn.CrossEntropyLoss()

            client = OfflineClient(
                client_id=client_id,
                train_iter=train_iter,
                eval_iter=eval_iter,
                model=global_model,
                loss_func=global_loss_func,
                optimizer=global_optimizer,
                local_model=local_model,
                local_loss_func=local_loss_func,
                local_optimizer=local_optimizer,
                config=config,
                device=device,
                grad_stub=grad_stub,
                clippingBound=clippingBound,
                debug_test_iter=debug_test_iter
            )



    