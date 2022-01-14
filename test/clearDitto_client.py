from Common.Node.workerbaseDitto import WorkerBase
from Common.Grpc.fl_grpc_pb2 import GradRequest_float
import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub

import numpy as np 

class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id # client id
        self.grad_stub = grad_stub # communication channel

    def update(self):
        if self.client_id < 6:
             gradients = super().get_gradients()
        else:
            # malicious update
             gradients = np.random.normal(0, 0.1, self._grad_len).tolist()

        # upload local gradients
        res_grad_upd = self.grad_stub.UpdateGrad_float.future(GradRequest_float(id=self.client_id, grad_ori=gradients))

        # receive global gradients
        super().set_gradients(gradients=res_grad_upd.result().grad_upd)

if __name__ == '__main__':
    
    args = args_parser() # load setting
    # only cpu used here
    if args.id <1:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)
    PATH = './Model/LeNet'
    model = LeNet().to(device)
    #model = ResNet(BasicBlock, [3,3,3]).to(device)
    model.load_state_dict(torch.load(PATH))
    if args.id == 0:
        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path)
    else:
        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path), None
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("connect success!")
        print("local constraint: lambda =", config.coef)
        print(args.id)
        grad_stub = FL_GrpcStub(grad_channel)
        print(device)

        client = ClearDenseClient(
            client_id=args.id, 
            model=model, 
            loss_func=loss_func, 
            train_iter=train_iter,
            test_iter=test_iter, 
            config=config, 
            optimizer=optimizer, 
            device=device, 
            grad_stub=grad_stub
        )

        client.fl_train(times=args.E)
        client.write_acc_record(fpath="Eva/clearDitto_avg_acc_test_mnist.txt", info="clear_avg_acc_worker_test")

