from Common.Node.workerbasev2 import WorkerBaseV2
from Common.Node.workerbase import WorkerBase
import torch
from torch import nn
from torch import device
import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
import copy
import time
from Common.Utils.evaluate import evaluate_accuracy
import numpy as np 

def server_robust_agg(grad): ## server aggregation
    grad_in = np.array(grad).reshape((config.num_workers, -1)).mean(axis=0)
    return grad_in.tolist()

class ClearDenseClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id
        self.grad_stub = None

    def update(self):
        pass
        #gradients = super().get_gradients()
        #return gradients
        #res_grad_upd = self.grad_stub.UpdateGrad_float.future(GradRequest_float(id=self.client_id, grad_ori=gradients))

        #super().set_gradients(gradients=res_grad_upd.result().grad_upd)


if __name__ == '__main__':

    args = args_parser()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #PATH = './Model/LeNet'
    PATH = './Model/ResNet20'
    #model = copy.deepcopy(LeNet()).to(device)
    model = copy.deepcopy(ResNet(BasicBlock, [3, 3, 3])).to(device)
    model.load_state_dict(torch.load(PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    client = []
    for i in range(config.num_workers):
        args.id = i
        if i == 0:
            train_iter, test_iter = load_data_cifar10(id=args.id, batch = args.batch_size, path = args.path) #load_data_cifar10(id=args.id, batch = args.batch_size, path = args.path)
        else:
            train_iter, test_iter = load_data_cifar10(id=args.id, batch = args.batch_size, path = args.path), None
        
        client.append(ClearDenseClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_iter,
                                  test_iter=test_iter, config=config, optimizer=optimizer, device=device, grad_stub=None))
    #import pdb
    #pdb.set_trace()
    acc_record = [0]
    counts = 0
    for epoch in range(config.num_epochs):
        print('epoch:',epoch)
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        weights = []
        for i in range(config.num_workers):
            client[i].train()

            weights.append(client[i].get_weights())

        result = server_robust_agg(weights)

        for i in range(config.num_workers):
            client[i].set_weights(weights=result)
            client[i].upgrade()