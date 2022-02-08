# Utils 
from Common.Node.workerbase_InferForv3 import WorkerBase as WorkerBaseInfer

# Other Libs
import numpy as np

class InferClient(WorkerBaseInfer):
    def __init__(self, model, loss_func, optimizer, train_iter, config, device='cpu', target=0):
        super().__init__(model, loss_func, optimizer, train_iter, device, target)
        
        self.config = config

    def adaptiveClipping(self, input_gradients):
        '''
        To clip the input gradient, if its norm is larger than the setting
        '''
        if self.config._dpoff: # don't clip
            return input_gradients, 1
        gradients = np.array(input_gradients) # list to numpy.ndarray
        norm = np.linalg.norm(gradients)        
        if norm > self.config.initClippingBound:
            return gradients * self.config.initClippingBound/np.linalg.norm(gradients), 0
        else:
            return gradients.tolist(), 1

    def update(self):
        pass