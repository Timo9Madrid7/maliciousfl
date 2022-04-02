import torch 
import torchvision
import random
import numpy as np
import argparse

class DataSplitter():
    def __init__(self, path: str, dataset="MNIST") -> None:
        """
        path: the path_dir to the dataset
        """
        self.path = path 
        self.num_class = 10
        self.data_train, self.data_test = None, None
        if dataset == "MNIST":
            self.data_train, self.data_test = self.load_trainTest_mnist()
        elif dataset == "CIFAR10":
            self.data_train, self.data_test = self.load_trainTest_cifar10()
    
    def load_trainTest_mnist(self):
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])

        mnist_train = torchvision.datasets.MNIST(root=self.path, train=True, download=False, transform=transforms)
        mnist_test = torchvision.datasets.MNIST(root=self.path, train=False, download=False, transform=transforms)
        
        return mnist_train, mnist_test

    def load_trainTest_cifar10(self):
        transforms_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4), 
            torchvision.transforms.RandomHorizontalFlip(), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transforms_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.cifar10_train = torchvision.datasets.CIFAR10(root=self.path, train=True, download=False, transform=transforms_train)
        self.cifar10_test = torchvision.datasets.CIFAR10(root=self.path, train=False, download=False, transform=transforms_test)

        return self.cifar10_train, self.cifar10_test

    def random_samples(self, m:int, main_index:int, p_main:float, p_others:float):
        classes = np.array(range(self.num_class))
        p = [p_others] * self.num_class
        p[main_index] = p_main
        samples_per_class = np.zeros(self.num_class, dtype=int)
        idx, num = np.unique(np.random.choice(classes,m,p=p), return_counts=True)
        samples_per_class[idx] = num
        return samples_per_class

    def split_train_data(self, n:int, m:int, q=0.1, save_path='/'):
        """
        n: total number of clients, multiple of 10;
        m: number of samples per client;
        q: degree of non-IID;
        save_path: './Data/[self.path]/' by default.
        """
        assert n%self.num_class == 0
        
        train_samples = [[] for _ in range(self.num_class)]
        for train_sample in self.data_train:
            train_samples[train_sample[1]].append(train_sample)

        for client in range(n):      
            samples_per_class = self.random_samples(m, client%self.num_class, q, (1-q)/(self.num_class-1))
            client_train = []
            for i in range(self.num_class):
                client_train += random.sample(train_samples[i], samples_per_class[i])
            random.shuffle(client_train)
            torch.save(client_train, self.path+save_path+"client_"+str(client)+".pt")
    
    def split_unique_train_data(self, n:int, q=0.1, save_path='/'):
        """split the training data according to the given degree of non-iid 
        such that each sample will be merely assigned to an unique client

        Args:
            n (int): total number of clients, multiple of 10.
            q (float, optional): degree of non-IID.
            save_path (str, optional): './Data/[self.path]/' by default.
        """
        assert n%self.num_class == 0

        train_samples = [[] for _ in range(self.num_class)]
        for train_sample in self.data_train:
            train_samples[train_sample[1]].append(train_sample)
        
        for client in range(n):      
            samples_per_class = self.random_samples(int(len(self.data_train)/n), client%self.num_class, q, (1-q)/(self.num_class-1))
            client_train = []
            for i in range(self.num_class):
                if samples_per_class[i] <= len(train_samples[i]):
                    for _ in range(samples_per_class[i]):
                        client_train.append(train_samples[i].pop(random.randint(0,len(train_samples[i])-1)))
                else:
                    client_train += train_samples[i]
            random.shuffle(client_train)
            torch.save(client_train, self.path+save_path+"client_"+str(client)+".pt")

    def split_eval_data(self, n:int, m:int, q=0.1, save_path='/'):
        """
        n: total number of clients, multiple of 10;
        m: number of samples per client;
        q: degree of non-IID;
        save_path: './Data/[self.path]/' by default.
        """
        assert n%self.num_class == 0
        
        eval_samples = [[] for _ in range(self.num_class)]
        for eval_sample in self.data_test:
            eval_samples[eval_sample[1]].append(eval_sample)

        for client in range(n):      
            samples_per_class = self.random_samples(m, client%self.num_class, q, (1-q)/(self.num_class-1))
            client_eval = []
            for i in range(self.num_class):
                client_eval += random.sample(eval_samples[i], samples_per_class[i])
            random.shuffle(client_eval)
            torch.save(client_eval, self.path+save_path+"client_eval_"+str(client)+".pt")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="MNIST", help="select from [MNIST, CIFAR10, ...], default MNIST")
    parser.add_argument("--n", type=int, default=100, help="total number of clients, default 100")
    parser.add_argument("--m1", type=int, default=600, help="number of training samples per client, default 600")
    parser.add_argument("--m2", type=int, default=100, help="number of testing samples per client, default 100")
    parser.add_argument("--unique", type=bool, default=False, help="client holds unique data? If true, m is useless, default False")
    parser.add_argument("--q", type=float, default=0.1, help="the non-iid ratio, 0.1 represents iid, default 0.1")
    parser.add_argument("--path", type=str, default="iid", help="the path to save the datasets, default iid")
    args = parser.parse_args()
    return args

if __name__ == "__main__":  
    args = args_parser()

    dataset = args.dataset
    path = "./Data/" + dataset
    mysplitter = DataSplitter(path=path, dataset=dataset)

    if args.unique:
        mysplitter.split_unique_train_data(n=args.n, q=args.q, save_path="/"+args.path+'/')
    else:
        mysplitter.split_train_data(n=args.n, m=args.m1, q=args.q, save_path="/"+args.path+'/')
    mysplitter.split_eval_data(n=args.n, m=args.m2, q=args.q, save_path="/"+args.path+'/')

    print("Finished Splition")




        


