import torch 
import torchvision

def load_data(client_index, batch=128, path="./Data/client_"):
    data = torch.load(path+client_index+".pt")
    return torch.utils.data.DataLoader(data, batch_size=batch, shuffle=True, num_workers=0)

def load_test(batch=128, path="./Data/"):
    transforms = torchvision.transforms
    test = torchvision.datasets.MNIST(root=path, train=False, download=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]))
    test_iter = torch.utils.data.DataLoader(test, batch_size=batch, shuffle=True, num_workers=0)
    return test_iter