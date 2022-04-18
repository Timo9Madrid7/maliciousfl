"""
Download EMNIST dataset, and splits it among clients
"""
import os
import argparse
import pickle
import torch

from torchvision.datasets import EMNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import ConcatDataset

from sklearn.model_selection import train_test_split

from utils import split_dataset_by_labels, pathological_non_iid_split


# TODO: remove this after new release of torchvision
EMNIST.url = "https://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip"

N_CLASSES = 62
RAW_DATA_PATH = "raw_data/"
PATH = "noniid/"


def save_data(l, path_):
    with open(path_, 'wb') as f:
        pickle.dump(l, f)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_tasks',
        help='number of tasks/clients;',
        type=int,
        required=True
    )
    parser.add_argument(
        '--pathological_split',
        help='if selected, the dataset will be split as in'
             '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument(
        '--n_shards',
        help='number of shards given to each clients/task; ignored if `--pathological_split` is not used;'
             'default is 2',
        type=int,
        default=2
    )
    parser.add_argument(
        '--n_components',
        help='number of components/clusters; default is -1',
        type=int,
        default=-1
    )
    parser.add_argument(
        '--alpha',
        help='parameter controlling tasks dissimilarity, the smaller alpha is the more tasks are dissimilar; '
             'default is 0.2',
        type=float,
        default=0.2)
    parser.add_argument(
        '--s_frac',
        help='fraction of the dataset to be used; default: 0.2;',
        type=float,
        default=0.2
    )
    parser.add_argument(
        '--tr_frac',
        help='fraction in training set; default: 0.8;',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--seed',
        help='seed for the random processes; default is 12345',
        type=int,
        default=12345
    )

    return parser.parse_args()


def main():
    args = parse_args()

    transform = Compose(
        [ToTensor(),
         Normalize((0.1307,), (0.3081,))
         ]
    )

    dataset = ConcatDataset([
        EMNIST(
            root=RAW_DATA_PATH,
            split="byclass",
            download=True,
            train=True,
            transform=transform
        ),
        EMNIST(root=RAW_DATA_PATH,
               split="byclass",
               download=False,
               train=False,
               transform=transform)
    ])

    if args.pathological_split:
        clients_indices = \
            pathological_non_iid_split(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_classes_per_client=args.n_shards,
                frac=args.s_frac,
                seed=args.seed
            )
    else:
        clients_indices = \
            split_dataset_by_labels(
                dataset=dataset,
                n_classes=N_CLASSES,
                n_clients=args.n_tasks,
                n_clusters=args.n_components,
                alpha=args.alpha,
                frac=args.s_frac,
                seed=args.seed,
            )

    train_clients_indices, test_clients_indices = train_test_split(clients_indices, test_size=0.2, random_state=args.seed)

    idx = 0
    server_test_indices = []
    for indices in train_clients_indices+test_clients_indices:
        train_indices, test_indices = train_test_split(indices, train_size=args.tr_frac, random_state=args.seed)
        server_test_indices += test_indices
        client_train = [dataset[i] for i in train_indices]
        torch.save(client_train, PATH+"client_{}.pt".format(idx))
        client_eval = [dataset[i] for i in test_indices]
        torch.save(client_eval, PATH+"client_eval_{}.pt".format(idx))
        idx += 1
    server_test = [dataset[i] for i in server_test_indices]
    torch.save(server_test, "./server_test.pt".format(idx))

if __name__ == "__main__":
    main()