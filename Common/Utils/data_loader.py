from Common.Utils.backdoorInjection import backdoor_mnist, backdoor_mnist_test, flipping_mnist, flipping_mnist_test
from Common.Utils.backdoorInjection import backdoor_cifar10, backdoor_cifar10_test, flipping_cifar10, flipping_cifar10_test

import torch
import torchvision
import random
import pickle
from PIL import Image

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

def load_data_backdoor_mnist(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/MNIST/noniid"
    else:
        path="./Data/MNIST/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    data = list(map(backdoor_mnist, data))
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    
def load_data_backdoor_mnist_test(batch=128, path="./Data/MNIST/"):
    transforms = torchvision.transforms
    test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test = list(map(backdoor_mnist_test, test))
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    return test_iter

def load_data_flipping_mnist(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/MNIST/noniid"
    else:
        path="./Data/MNIST/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    data = list(map(flipping_mnist, data))
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_data_flipping_mnist_test(batch=128, path="./Data/MNIST/"):
    transforms = torchvision.transforms
    test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test = list(map(flipping_mnist_test, test))
    return torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)

def load_data_edge_case_mnist(client_index, num_edge_case=60, batch=128, noniid=True):
    if noniid:
        path="./Data/MNIST/noniid"
    else:
        path="./Data/MNIST/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    edge_case = random.sample(torch.load('./Data/Ardis_IV/edge_case_train_7.pt'), k=num_edge_case)
    data += edge_case
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_data_edge_case_mnist_test(batch=128):
    fake_labels_data = torch.load('./Data/Ardis_IV/edge_case_test_7.pt')
    true_labels_data = torch.load('./Data/Ardis_IV/edge_case_test_true_7.pt')
    return torch.utils.data.DataLoader(fake_labels_data, batch_size=batch, shuffle=True, num_workers=0), torch.utils.data.DataLoader(true_labels_data, batch_size=batch, shuffle=True, num_workers=0)

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
    trans_aug = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=trans_aug)
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    return test_iter

def load_data_backdoor_cifar10(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/CIFAR10/noniid"
    else:
        path="./Data/CIFAR10/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    data += backdoor_cifar10()
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)
    
def load_data_backdoor_cifar10_test(batch=128):
    return torch.utils.data.DataLoader(backdoor_cifar10_test(), batch_size=batch, shuffle=True, num_workers=0)

def load_data_flipping_cifar10(client_index, batch=128, noniid=True):
    if noniid:
        path="./Data/CIFAR10/noniid"
    else:
        path="./Data/CIFAR10/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')
    data = list(map(flipping_cifar10, data))
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_data_flipping_cifar10_test(batch=128, path="./Data/CIFAR10/"):
    transforms = torchvision.transforms
    trans_aug = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    test = torchvision.datasets.CIFAR10(root=path, train=False, download=False, transform=trans_aug)
    test = list(map(flipping_cifar10_test, test))
    return torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)

def load_data_edge_case_cifar10(client_index, num_edge_case=60, batch=128, noniid=True):
    if noniid:
        path="./Data/CIFAR10/noniid"
    else:
        path="./Data/CIFAR10/iid"
    data = torch.load(path+'/'+'client_'+client_index+'.pt')

    with open("./Data/SouthwestAirline/edge_case_train.pkl", 'rb') as f:
        edge_case = pickle.load(f)
    edge_case = edge_case[random.sample(range(len(edge_case)), k=num_edge_case)]

    transforms_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4), 
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    edge_data = []
    for case in edge_case:
        edge_data.append([transforms_train(Image.fromarray(case)), 9])

    data += edge_data
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_data_edge_case_cifar10_test(batch=128):
    with open("./Data/SouthwestAirline/edge_case_test.pkl", 'rb') as f:
        edge_case = pickle.load(f)

    transforms_test = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    fake_labels_data = []
    true_labels_data = []
    for case in edge_case:
        fake_labels_data.append([transforms_test(Image.fromarray(case)), 9])
        true_labels_data.append([transforms_test(Image.fromarray(case)), 0])

    return torch.utils.data.DataLoader(fake_labels_data, batch_size=batch, shuffle=True, num_workers=0), torch.utils.data.DataLoader(true_labels_data, batch_size=batch, shuffle=True, num_workers=0)

def load_dataset(client_index, dataset="MNIST", test=False, batch=128, noniid=False):
    if noniid:
        path = "./Data/" + dataset + "/noniid/client_"
    else:
        path = "./Data/" + dataset + "/iid/client_"
    path += "eval_" if test else ""
    data = torch.load(path+client_index+'.pt')
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_testset(dataset="MNIST", batch=128):
    if dataset == "MNIST":
        return load_all_test_mnist(batch=batch)
    elif dataset == "CIFAR10":
        return load_all_test_cifar10(batch=batch)

def load_backdoor(client_index, dataset="MNIST", batch=128, noniid=True):
    if dataset == "MNIST":
        return load_data_backdoor_mnist(client_index, batch=batch, noniid=noniid)
    elif dataset == "CIFAR10":
        return load_data_backdoor_cifar10(client_index, batch=batch, noniid=noniid)

def load_backdoor_test(dataset="MNIST", batch=128):
    if dataset == "MNIST":
        return load_data_backdoor_mnist_test(batch=batch)
    elif dataset == "CIFAR10":
        return load_data_backdoor_cifar10_test(batch=batch)

def load_flipping(client_index, dataset="MNIST", batch=128, noniid=True):
    if dataset == "MNIST":
        return load_data_flipping_mnist(client_index, batch=batch, noniid=noniid)
    elif dataset == "CIFAR10":
        return load_data_flipping_cifar10(client_index, batch=batch, noniid=noniid)

def load_flipping_test(dataset="MNIST", batch=128):
    if dataset == "MNIST":
        return load_data_flipping_mnist_test(batch=batch)
    elif dataset == "CIFAR10":
        return load_data_flipping_cifar10_test(batch=batch)

def load_edgecase(client_index, dataset="MNIST", num_edge_case=60, batch=128, noniid=True):
    if dataset == "MNIST":
        return load_data_edge_case_mnist(client_index, num_edge_case, batch, noniid)
    elif dataset == "CIFAR10":
        return load_data_edge_case_cifar10(client_index, num_edge_case, batch, noniid)

def load_edgecase_test(dataset="MNIST", batch=128):
    if dataset == "MNIST":
        return load_data_edge_case_mnist_test(batch)
    elif dataset == "CIFAR10":
        return load_data_edge_case_cifar10_test(batch)