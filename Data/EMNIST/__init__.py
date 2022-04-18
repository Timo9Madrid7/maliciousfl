if __name__ == "__main__":
    import torchvision
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])

    emnist_train = torchvision.datasets.EMNIST(root="./", split="byclass", train=True, download=True, transform=transforms)
    emnist_test = torchvision.datasets.EMNIST(root="./", split="byclass", train=False, download=True, transform=transforms)