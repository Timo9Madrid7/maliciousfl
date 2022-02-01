from dataclasses import replace
import torch 
import torchvision
import random
import numpy as np

# TODO: class Splitter should be the parent class for children MNIST, FMNIST and CIFAR spliters.
class MNISTSplitter():
    def __init__(self, path: str) -> None:
        """
        path: the path_dir to MNIST Data
        """
        self.path = path 
        self.num_class = 10
        self.mnist_train, self.mnist_test = self.load_trianTest_mnist()
    
    def load_trianTest_mnist(self):
        transforms = torchvision.transforms

        mnist_train = torchvision.datasets.MNIST(
                root=self.path, train=True, download=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(), 
                    transforms.Normalize((0.1307,), (0.3081,)) 
            ]))

        mnist_test = torchvision.datasets.MNIST(
                root=self.path, train=False, download=False, 
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
            ]))
        
        return mnist_train, mnist_test

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
        save_path: './Data/MNIST/' by default.
        """
        assert n%10 == 0
        
        train_samples = [[] for _ in range(self.num_class)]
        for train_sample in self.mnist_train:
            train_samples[train_sample[1]].append(train_sample)

        for client in range(n):      
            samples_per_class = self.random_samples(m, client%10, q, (1-q)/(self.num_class-1))
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
        save_path: './Data/MNIST/' by default.
        """
        assert n%10 == 0
        
        eval_samples = [[] for _ in range(self.num_class)]
        for eval_sample in self.mnist_test:
            eval_samples[eval_sample[1]].append(eval_sample)

        for client in range(n):      
            samples_per_class = self.random_samples(m, client%10, q, (1-q)/(self.num_class-1))
            client_eval = []
            for i in range(self.num_class):
                client_eval += random.sample(eval_samples[i], samples_per_class[i])
            random.shuffle(client_eval)
            torch.save(client_eval, self.path+save_path+"client_eval_"+str(client)+".pt")

if __name__ == "__main__":
    
    path = "./Data/MNIST"
    mysplitter = MNISTSplitter(path)

    # MNIST IID data splition
    mysplitter.split_train_data(n=200, m=600, q=0.1, save_path='/iid/')
    mysplitter.split_eval_data(n=200, m=600, q=0.1, save_path='/iid/')

    # MNIST non-IID data splition
    mysplitter.split_train_data(n=200, m=600, q=0.5, save_path='/noniid/')
    mysplitter.split_eval_data(n=200, m=600, q=0.5, save_path='/noniid/')
    print("Finished Splition")

        


        


