from Common.Node.workerbase import WorkerBase
from Common.Grpc.fl_grpc_pb2 import GradRequest_Clipping
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

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ClearDenseClient(WorkerBase):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id # client id
        self.grad_stub = grad_stub # communication channel
        self.clippingBound = config.initClippingBound
        self.b_noise = config.b_noise

    def adaptiveClipping(self, input_gradients):
        '''
        To clip the input gradient, if its norm is larger than the setting
        '''

        # --- adaptive noise calculation ---
        b_noise = self.b_noise
        grad_noise = (config.z_multiplier**(-2) - (2*b_noise)**(-2))**(-0.5) * self.clippingBound
        # --- adaptive noise calculation ---

        gradients = np.array(input_gradients)
        norm = np.linalg.norm(gradients)        
        if norm > self.clippingBound:
            return gradients * self.clippingBound/np.linalg.norm(gradients) + np.random.normal(0, grad_noise, size=gradients.shape), 0 + np.random.normal(0,b_noise)
        else:
            return gradients + np.random.normal(0, grad_noise, size=gradients.shape), 1 + np.random.normal(0,b_noise)

    def update(self):
        if self.client_id < 10:
            # clipping gradients before upload to the server
             gradients, b = self.adaptiveClipping(super().get_gradients())
        else:
            # malicious client when id>=10
             gradients, b = np.random.normal(0, 0.1, self._grad_len).tolist(), 1

        # upload local gradients and clipping indicator
        res_grad_upd = self.grad_stub.UpdateGrad_Clipping.future(GradRequest_Clipping(id=self.client_id, b=b, grad_ori=gradients))

        # receive global gradients
        super().set_gradients(gradients=res_grad_upd.result().grad_upd)
        # update the clipping bound for the next round
        self.clippingBound = res_grad_upd.result().b


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
        client.write_acc_record(fpath="Eva/clear_client_clipping_avg_acc_test_mnist.txt", info="clear_avg_acc_worker_test")
