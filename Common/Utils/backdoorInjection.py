from OfflinePack.offline_config import backdoor_pdr, backdoor_target, flipping_pdr
from OfflinePack.offline_config import semantic_feature, num_inserted

import numpy as np
import torch
import torchvision
import random

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

def backdoor_cifar10(semantic_feature=semantic_feature, num_inserted=num_inserted, target=backdoor_target):
    path = "./Data/CIFAR10/backdoor/cars_"+semantic_feature+"_train.npy"
    semantic_images = random.choices(np.load(path, allow_pickle=True), k=num_inserted)

    transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4), 
            torchvision.transforms.RandomHorizontalFlip(), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    return list(zip([transforms_train(image) for image in semantic_images], [target]*num_inserted))

def backdoor_cifar10_test(semantic_feature=semantic_feature, target=backdoor_target, test_only=False):
    path = "./Data/CIFAR10/backdoor/cars_"+semantic_feature+"_test.npy"
    semantic_images = np.load(path, allow_pickle=True).tolist()
    if not test_only:
        path = "./Data/CIFAR10/backdoor/cars_"+semantic_feature+"_train.npy"
        semantic_images += np.load(path, allow_pickle=True).tolist()

    transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    return list(zip([transforms_test(image) for image in semantic_images], [target]*len(semantic_images)))

def flipping_cifar10(data:list, pdr=flipping_pdr):
    if np.random.rand() < pdr:
        return data[0], 9-data[1]
    else:
        return data[0], data[1]

def flipping_cifar10_test(data:list):
    return data[0], 9-data[1]

def backdoor_emnist(data:list, pdr=backdoor_pdr, target=backdoor_target):
    if np.random.rand() < pdr:
        return data[0]+injection, target 
    else:
        return data[0], data[1]

def backdoor_emnist_test(data:list, target=backdoor_target):
    return data[0]+injection, target

def flipping_emnist(data:list, pdr=flipping_pdr):
    if np.random.rand() < pdr:
        if data[1] < 10:
            flipping_label = 9 - data[1]
        elif data[1] < 36:
            flipping_label = 10 + (35-data[1])
        else: # data[1] < 62:
            flipping_label = 36 + (61-data[1])
        return data[0], flipping_label
    else:
        return data[0], data[1]

def flipping_emnist_test(data:list):
    if data[1] < 10:
        flipping_label = 9 - data[1]
    elif data[1] < 36:
        flipping_label = 10 + (35-data[1])
    else: # data[1] < 62:
        flipping_label = 36 + (61-data[1])
    return data[0], flipping_label