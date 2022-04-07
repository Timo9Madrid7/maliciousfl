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
        config, device):
        super().__init__(client_id, train_iter, eval_iter, model, loss_func, optimizer, local_model, local_loss_func, local_optimizer, config, device)

    def update(self):
        pass

    def evaluation(self):
        pass

    def upgrade_local(self):
        torch.save(self.local_model.state_dict(), self.config.local_models_path+self.client_id)
    
    def malicious_random_upload(self):
        num_params = sum([params.numel() for params in self.model.state_dict().values()])
        return np.random.normal(self.clippingBound, 1, size=(num_params,))