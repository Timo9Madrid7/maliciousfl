import torch
import torchvision
import pdb

def load_data_mnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'mnist_train_'+str(id%10)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0: # eval 只对 id=0 做
        transforms = torchvision.transforms
        # train=False: load test.pt
        # transforms.Compose(): chain various tranforms together
        # transforms.Totensor(): covert PIL Image or numpy.ndarray to tensor
        # transforms.Normalize(): Normalize a float tensor image with mean and standard deviation
        test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_all_test_mnist(batch=128, path="./Data/MNIST/"):
    transforms = torchvision.transforms
    test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    return test_iter

def load_data_posioned_mnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'mnist_train_posioned'+str(id%10)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_noniid_mnist(client_index, batch=128, noniid=True):
    if noniid:
        path = "./Data/MNIST/noniid"
    else:
        path = "./Data/MNIST/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    return train_iter

def load_data_dpclient_mnist(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/MNIST/noniid/client_dp/client_"
    else:
        path="./Data/MNIST/iid/client_dp/client_"
    data = torch.load(path+client_index+".pt")
    dp_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    return dp_iter

def load_data_dittoEval_mnist(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/MNIST/noniid"
    else:
        path="./Data/MNIST/iid"
    data = torch.load(path+'/'+'client_eval_'+client_index+'.pt')
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_data_usps(id, batch=None, path=None):
    data = torch.load(path+'/'+'usps_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        test = torchvision.datasets.USPS(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_fmnist(id, batch=None, path=None):
    data = torch.load(path+'/'+'fmnist_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        test = torchvision.datasets.FashionMNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_cifar10(id, batch=None, path=None):
    data = torch.load(path+'/'+'cifar10_train_'+str(id)+'_.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    if id == 0:
        transforms = torchvision.transforms
        trans_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        test = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=trans_aug)
        test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
        return train_iter, test_iter

    return train_iter

def load_data_noniid_cifar10(client_index, batch=128, noniid=True):
    if noniid:
        path = "./Data/CIFAR10/noniid"
    else:
        path = "./Data/CIFAR10/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    train_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    return train_iter

def load_data_dpclient_cifar10(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/CIFAR10/noniid/client_dp/client_"
    else:
        path="./Data/CIFAR10/iid/client_dp/client_"
    data = torch.load(path+client_index+".pt")
    dp_iter = torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    return dp_iter

def load_data_dittoEval_cifar10(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/CIFAR10/noniid"
    else:
        path="./Data/CIFAR10/iid"
    data = torch.load(path+'/'+'client_eval_'+client_index+'.pt')
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_all_test_cifar10(batch=128, path="./Data/CIFAR10/"):
    transforms = torchvision.transforms
    trans_aug = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=trans_aug)
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    return test_iter
    
