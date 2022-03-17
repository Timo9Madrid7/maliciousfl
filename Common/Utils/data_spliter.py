import torch 
import torchvision
import random
import numpy as np

def load_trainTest_mnist():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    mnist_train = torchvision.datasets.MNIST(root="./Data/", train=True, download=False, transform=transforms)
    mnist_test = torchvision.datasets.MNIST(root="./Data/", train=False, download=False, transform=transforms)
    
    return mnist_train, mnist_test

def split_train_test_data():
    data_train, data_test = load_trainTest_mnist()

    train_samples = [[] for _ in range(10)]
    for train_sample in data_train:
        train_samples[train_sample[1]].append(train_sample)
    
    eval_samples = [[] for _ in range(10)]
    for eval_sample in data_test:
        eval_samples[eval_sample[1]].append(eval_sample)

    for client in range(10):
        client_train = []
        if client < 5:
            client_train += random.sample(train_samples[0], 1200)
        for i in range(1, 10):
            client_train += random.sample(train_samples[i], 600)
        random.shuffle(client_train)
        torch.save(client_train, "./Data/"+"client_"+str(client)+".pt")

    torch.save(eval_samples[0], "./Data/"+"digit_0.pt")

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
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4), 
            torchvision.transforms.RandomHorizontalFlip(), 
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        self.cifar10_train = torchvision.datasets.CIFAR10(root=self.path, train=True, download=False, transform=transforms)
        self.cifar10_test = torchvision.datasets.CIFAR10(root=self.path, train=False, download=False, transform=transforms)

        return self.cifar10_train, self.cifar10_test

    def random_samples(self, m:int, main_index:int, p_main:float, p_others:float):
        classes = np.array(range(self.num_class))
        p = [p_others] * self.num_class
        p[main_index] = p_main
        return np.unique(np.random.choice(classes,m,p=p), return_counts=True)[1]

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

if __name__ == "__main__":
    split_train_test_data()

    for client in range(10):
        client_data = torch.load("./Data/client_"+str(client)+".pt")
        labels_distribution = [client_data[i][1] for i in range(len(client_data))]
        print("client %d:"%client, np.unique(labels_distribution, return_counts=True), end="\n\n")

    all_0_test_set = torch.load("./Data/digit_0.pt")
    labels_distribution = [all_0_test_set[i][1] for i in range(len(all_0_test_set))]
    print("test set", np.unique(labels_distribution, return_counts=True))


    # path = "./Data/"
    # dataset = "MNIST"
    # mysplitter = DataSplitter(path=path, dataset=dataset)
    # mysplitter.split_train_data(n=10, m=6000, q=0.5, save_path='noniid/')

    # for client in range(10):
    #     client_data = torch.load("./Data/noniid/client_"+str(client)+".pt")
    #     labels_distribution = [client_data[i][1] for i in range(len(client_data))]
    #     print("client %d:"%client, np.unique(labels_distribution, return_counts=True), end="\n\n")