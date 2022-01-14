import torch
import torchvision

import itertools
import random

import matplotlib.pyplot as plt

def load_trans_mnist(root="./Data/MNIST"):
    transforms = torchvision.transforms
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    
    return mnist_train

mnist_train = load_trans_mnist()

def create_noniid_clients(data=mnist_train, number_classes_per_client=3, number_samples_per_class=200, save_path="./Data/MNIST/noniid/client_"):
    collection = [[] for _ in range(10)]
    for sample in mnist_train:
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
    # create_noniid_clients()
    
    test_client = torch.load("./Data/MNIST/noniid/client_678.pt")
    print(len(test_client), len(test_client[0]))
    # plt.imshow(test_client[0][0].reshape(28,28))
    print(test_client[0][1])
    # plt.imshow(test_client[200][0].reshape(28,28))   
    print(test_client[200][1])                                  
    # plt.imshow(test_client[400][0].reshape(28,28))
    print(test_client[400][1])
    plt.show()



# %%
