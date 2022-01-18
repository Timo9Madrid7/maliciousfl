import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=1, help="rounds of training")
    # parser.add_argument('--num_workers', type=int, default=10, help="number of users: N")
    # parser.add_argument('--E', type=int, default=1, help="the number of local updates: E")
    parser.add_argument('--batch_size', type=int, default=128, help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")
    parser.add_argument('--id', type=int, default=0, help='thread id (default: 0)')
    parser.add_argument('--path', type=str, default='./Data/MNIST/', help="dataset")
    #parser.add_argument('--path', type=str, default='./Data/FMNIST/', help="dataset")
    #parser.add_argument('--path', type=str, default='./Data/CIFAR10/', help="dataset")
    args = parser.parse_args()
    return args
