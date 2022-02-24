# Utils 
from Common.Node.workerbase_InferForv3 import WorkerBase as WorkerBaseInfer

# Other Libs
import numpy as np

class InferClient(WorkerBaseInfer):
    def __init__(self, model, loss_func, optimizer, train_iter, config, device='cpu', target=0):
        super().__init__(model, loss_func, optimizer, train_iter, device, target)
        
        self.config = config

    def update(self):
        pass