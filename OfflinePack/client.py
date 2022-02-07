# Utils
from Common.Node.workerbase_v3 import WorkerBase as WorkerBaseDitto
from Common.Utils.evaluate import evaluate_accuracy

# Other Libs
import torch
import numpy as np 

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
        pass

    def upgrade_local(self):
        torch.save(self.local_model.state_dict(), self.config.local_models_path+self.client_id)
    
    def malicious_random_upload(self, model="LeNet"):
        pass