import torch
import torchvision
import pdb

def poison_data_mnist(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'mnist_train_'+str(id)+'_.pt')
        poison_data = []
        for i in range(len(data)):
            poison_data.append((data[i][0], 9 - data[i][1]))
        torch.save(poison_data, path +'/' + 'mnist_train_poisoned' + str(id) +'_.pt')
    print("mnist poisoned end")


def poison_data_cifar10(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'cifar10_train_'+str(id)+'_.pt')
        poison_data = []
        for i in range(len(data)):
            poison_data.append((data[i][0], 9 - data[i][1]))
        torch.save(poison_data, path +'/' + 'cifar10_train_poisoned' + str(id) +'_.pt')
    print("poisoned end")

def poison_data_fmnist(path=None):
    for id in range(10):
        data = torch.load(path+'/'+'fmnist_train_'+str(id)+'_.pt')
        poison_data = []
        for i in range(len(data)):
            poison_data.append((data[i][0], 9 - data[i][1]))
        torch.save(poison_data, path +'/' + 'fmnist_train_poisoned' + str(id) +'_.pt')
    print("fmnist poisoned end")



if __name__ == '__main__':
    path = './Data/MNIST'
    poison_data_mnist(path=path)