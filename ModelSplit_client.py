from Common.Node.workerbase import WorkerBase
from Common.Node.workerbasev2 import WorkerBaseV2
from Common.Grpc.fl_grpc_pb2 import GradRequest_float
import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist, load_data_cifar10, load_data_fmnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub
import numpy as np

class ClearFLGuardClient(WorkerBaseV2):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub, num_model_params):
        super(ClearFLGuardClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device, num_model_params=num_model_params)
        self.client_id = client_id
        self.grad_stub = grad_stub

    def update(self):
        if self.client_id < 5:
            weights = super().get_weights()

        else:
            weights = np.random.normal(0, 0.1, self._weights_len).tolist()

        res_grad_upd = self.grad_stub.UpdateGrad_float(GradRequest_float(id=self.client_id, grad_ori=weights))

        super().set_weights(weights=res_grad_upd.grad_upd)

if __name__ == '__main__':
    args = args_parser()
    if args.id < 1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    PATH = './Model/LeNet'
    model = LeNet().to(device)
    model.load_state_dict(torch.load(PATH))
    if (args.id == 0) or (args.id == 9):

        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path)
    else:
        train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path), None

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("connect success!")

        grad_stub = FL_GrpcStub(grad_channel)
        num_model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        client = ClearFLGuardClient(client_id=args.id, model=model, loss_func=loss_func, train_iter=train_iter,
                                    test_iter=test_iter, config=config, optimizer=optimizer, device=device, grad_stub=grad_stub, num_model_params=num_model_params)

        client.fl_train(times=args.E)
        client.write_acc_record(fpath="Eva/clear_flgurd_acc_mnist.txt", info="clear_flguard_acc_worker_mnist")
