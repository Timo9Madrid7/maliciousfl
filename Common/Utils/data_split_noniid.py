import torch
import torchvision

import itertools
import random

import matplotlib.pyplot as plt

def load_trian_mnist(root="./Data/MNIST"):
    transforms = torchvision.transforms
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    
    return mnist_train

def load_test_mnist(root='./Data/MNIST'):
    transforms = torchvision.transforms
    mnist_test = torchvision.datasets.MNIST(root=root, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    
    return mnist_test

def create_noniid_clients(number_classes_per_client=7, number_samples_per_class=300, save_path="./Data/MNIST/noniid/client_"):
    data = load_trian_mnist()
    collection = [[] for _ in range(10)]
    for sample in data:
        collection[sample[1]].append(sample[0])
        
    clients_classes = itertools.combinations(range(0,10), number_classes_per_client)
    for client_classes in clients_classes:
        temp = []
        path = save_path
        for client_class in client_classes:
            temp += list(zip(random.sample(collection[client_class], number_samples_per_class), [client_class]*number_samples_per_class))
            path += str(client_class)
        assert len(temp) == number_samples_per_class * number_classes_per_client
        path += ".pt"
        torch.save(temp, path)
        
def create_dptest_client(client_classes="0123456", number_samples_per_class=300, save_path="./Data/MNIST/noniid/test/dpclient_"):
    data = load_test_mnist()
    collection = [[] for _ in range(10)]
    for sample in data:
        collection[sample[1]].append(sample[0])
    
    temp = []
    for per_class in client_classes:
        temp += list(zip(random.sample(collection[int(per_class)], number_samples_per_class), [int(per_class)]*number_samples_per_class))
    assert len(temp) == len(client_classes) * number_samples_per_class
    
    random.shuffle(temp)
    path = save_path + client_classes + ".pt"
    torch.save(temp, path)

def create_noniid_eval_clients(number_classes_per_client=7, number_samples_per_class=50, save_path="./Data/MNIST/noniid/client_eval_"):
    data = load_test_mnist()
    collection = [[] for _ in range(10)]
    for sample in data:
        collection[sample[1]].append(sample[0])
        
    clients_classes = itertools.combinations(range(0,10), number_classes_per_client)
    for client_classes in clients_classes:
        temp = []
        path = save_path
        for client_class in client_classes:
            temp += list(zip(random.sample(collection[client_class], number_samples_per_class), [client_class]*number_samples_per_class))
            path += str(client_class)
        assert len(temp) == number_samples_per_class * number_classes_per_client
        path += ".pt"
        torch.save(temp, path)
  
if __name__ == "__main__":
    create_noniid_clients()
    # create_noniid_eval_clients()
    create_dptest_client(client_classes="0123456")
    
    # test_client = torch.load("./Data/MNIST/noniid/client_678.pt")
    # print(len(test_client), len(test_client[0]))
    # # plt.imshow(test_client[0][0].reshape(28,28))
    # print(test_client[0][1])
    # # plt.imshow(test_client[200][0].reshape(28,28))   
    # print(test_client[200][1])                                  
    # # plt.imshow(test_client[400][0].reshape(28,28))
    # print(test_client[400][1])
    # plt.show()
    
    # import numpy as np
    # test = torch.load("./Data/MNIST/noniid/test/dpclient_0123456.pt")
    # labels = []
    # for sample in test:
    #     labels.append(sample[1])
        
    # print(np.unique(labels, return_counts=True))



# %%
