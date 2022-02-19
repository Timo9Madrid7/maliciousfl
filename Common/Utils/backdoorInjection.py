from OfflinePack.offline_config import backdoor_pdr, backdoor_target, flipping_pdr

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

def flipping_mnist(data:list, pdr=flipping_pdr):
    if np.random.rand() < pdr:
        return data[0], 9-data[1]
    else:
        return data[0], data[1]

def flipping_mnist_test(data:list):
    return data[0], 9-data[1]