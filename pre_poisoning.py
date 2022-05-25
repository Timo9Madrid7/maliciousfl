from Common.Utils.backdoorInjection import backdoor_mnist, flipping_mnist
from Common.Utils.backdoorInjection import backdoor_cifar10, flipping_cifar10
from Common.Utils.backdoorInjection import backdoor_emnist, flipping_emnist

import torch
import torchvision
import random
import pickle
import argparse
from PIL import Image

from Common.Utils.configSel import load_dict
import sys
config_path = load_dict("config_path")["config_path"]
sys.path.append(config_path)
import offline_config as config

class PPoison():
    def __init__(self, n:int, frac:float, dataset:str, noniid:bool, type:str):
        
        self.n = n 
        self.frac = frac
        self.dataset = dataset
        self.noniid = noniid
        self.type = type
        self.edge_case_num = config.edge_case_num
        self.pidx = random.sample(range(n), k=int(n*self.frac))

        if self.dataset == "MNIST":
            self.path = "./Data/MNIST/noniid" if self.noniid else "./Data/MNIST/iid"
        elif self.dataset == "CIFAR10":
            self.path = "./Data/CIFAR10/noniid" if self.noniid else "./Data/CIFAR10/iid"
        elif self.dataset == "EMNIST":
            self.path = "./Data/EMNIST/noniid" if self.noniid else "./Data/EMNIST/iid"

    def _backdoor_poison(self, client_index):
        data = torch.load(self.path+'/client_'+str(client_index)+'.pt')
        if self.dataset == "MNIST":
            data = list(map(backdoor_mnist, data))
        elif self.dataset == "CIFAR10":
            data += backdoor_cifar10()
        elif self.dataset == "EMNIST":
            data = list(map(backdoor_emnist, data))
        return data
        
    def _flipping_poison(self, client_index):
        data = torch.load(self.path+'/client_'+str(client_index)+'.pt')
        if self.dataset == "MNIST":
            data = list(map(flipping_mnist, data))
        elif self.dataset == "CIFAR10":
            data = list(map(flipping_cifar10, data))
        elif self.dataset == "EMNIST":
            data = list(map(flipping_emnist, data))
        return data
    
    def _edgecase_poison(self, client_index):
        data = torch.load(self.path+'/client_'+str(client_index)+'.pt')
        if self.dataset == "MNIST" or self.dataset == "EMNIST":
            edge_case = random.sample(torch.load('./Data/Ardis_IV/edge_case_train_7.pt'), k=self.edge_case_num)
        elif self.dataset == "CIFAR10":
            with open("./Data/SouthwestAirline/edge_case_train.pkl", 'rb') as f:
                edge_data = pickle.load(f)
            edge_data = edge_data[random.sample(range(len(edge_data)), k=self.edge_case_num)]
            transforms_train = torchvision.transforms.Compose([
                torchvision.transforms.RandomCrop(32, padding=4), 
                torchvision.transforms.RandomHorizontalFlip(), 
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            edge_case = []
            for case in edge_data:
                edge_case.append([transforms_train(Image.fromarray(case)), 9])   
        
        data += edge_case
        return data

    def poison(self):
        if self.type == "backdoor":
            pmethod = self._backdoor_poison
        elif self.type == "flipping":
            pmethod = self._flipping_poison
        elif self.type == "edge":
            pmethod = self._edgecase_poison

        for client_index in self.pidx:
            pdata = pmethod(client_index)
            torch.save(pdata, self.path+'/client_'+str(client_index)+'.pt')

        self.pidx.sort()
        return self.pidx
    
def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=config.total_number_clients, help="total number of clients")
    parser.add_argument("--frac", type=float, default=0.4, help="poisoned model ratio")
    parser.add_argument("--dataset", type=str, default=config.DATASET, help="name of the dataset")
    parser.add_argument("--noniid", type=bool, default=config._noniid, help="whether noniid")
    parser.add_argument("--type", type=str, default="backdoor", help="type of poison")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()

    poisoner = PPoison(args.n, args.frac, args.dataset, args.noniid, args.type)
    poison_indices = poisoner.poison()

    print("Finished poisoning!")
    print("poisoned indices:", poison_indices)
