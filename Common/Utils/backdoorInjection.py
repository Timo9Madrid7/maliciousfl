from OfflinePack.offline_config import backdoor_pdr, backdoor_target

import numpy as np
import torch

injection = torch.zeros((28,28))
injection[1:6, 1:6] = 2
def backdoor_mnist(data:list, pdr=backdoor_pdr, target=backdoor_target):
    if np.random.rand() < pdr:
        return data[0]+injection, target 
    else:
        return data[0], data[1]

def backdoor_mnist_test(data:list, target=backdoor_target):
    return data[0]+injection, target
    