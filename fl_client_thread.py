# from Common.Node.workerbaseDitto_FedAvg import WorkerBase as WorkerBaseDitto
from Common.Node.workerbase_v3 import WorkerBase as WorkerBaseDitto
from Common.Grpc.fl_grpc_pb2 import GradRequest_Clipping
import torch
from torch import nn

import Common.config as config

from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist, load_data_noniid_mnist, load_data_dittoEval_mnist, load_data_dpclient_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub

import numpy as np 

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ClearDenseClient(WorkerBaseDitto):
    def __init__(self, thread_id, test_iter, train_iter_loader, dittoEval_loader, different_client_loader, model, loss_func, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(thread_id = thread_id, 
        test_iter=test_iter, train_iter_loader=train_iter_loader, dittoEval_loader=dittoEval_loader, different_client_loader=different_client_loader,
        model=model, loss_func=loss_func, config=config, optimizer=optimizer, device=device)
        self.grad_stub = grad_stub # communication channel
        
        self.dpoff = self.config._dpoff
        self.clippingBound = self.config.initClippingBound # initial clipping bound for a client
        self.grad_noise_sigma = self.config.grad_noise_sigma
        self.b_noise_std = self.config.b_noise_std
        self.clients_per_round = self.config.num_workers

    def adaptiveClipping(self, input_gradients):
        '''
        To clip the input gradient, if its norm is larger than the setting
        '''

        if self.dpoff: # don't clip
            return np.array(input_gradients), 1
        # else: do clipping+noising

        gradients = np.array(input_gradients)

        # --- adaptive noise calculation ---
        if self.grad_noise_sigma==0:
            grad_noise = 0
            b_noise = 0
        else: 
            grad_noise_std = self.grad_noise_sigma * self.clippingBound # deviation for gradients
            b_noise = np.random.normal(0, self.b_noise_std)/self.clients_per_round
            grad_noise = np.random.normal(0, grad_noise_std, size=gradients.shape)/self.clients_per_round
        # --- adaptive noise calculation ---
           
        norm = np.linalg.norm(gradients)        
        if norm > self.clippingBound:
            return gradients * self.clippingBound/np.linalg.norm(gradients) + grad_noise, 0 + b_noise
        else:
            return gradients + grad_noise, 1 + b_noise

    def update(self, model_id):
        # clipping gradients before upload to the server
        gradients, b = self.adaptiveClipping(super().get_gradients())

        # upload local gradients and clipping indicator
        res_grad_upd = self.grad_stub.UpdateGrad_Clipping.future(GradRequest_Clipping(id=self.thread_id, b=b, grad_ori=gradients))

        # receive global gradients
        super().set_gradients(gradients=res_grad_upd.result().grad_upd)
        # update the clipping bound for the next round
        self.clippingBound = res_grad_upd.result().b


if __name__ == '__main__':

    args = args_parser() # load setting

    # only cpu used here
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    yaml_path = 'Log/log.yaml'
    setup_logging(default_path=yaml_path)

    # model setttings
    PATH = './Model/LeNet'
    model = LeNet().to(device)
    model.load_state_dict(torch.load(PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    # server settings
    server_grad = config.server1_address + ":" + str(config.port1)

    # client settings
    if args.id == 0:
        _, test_iter = load_data_mnist(args.id, batch=128, path="./Data/MNIST/")
    else: 
        test_iter = None
    

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("thread [%d]-%s: connect success!"%(args.id, device))
        grad_stub = FL_GrpcStub(grad_channel)

        client = ClearDenseClient(
            thread_id=args.id, 
            test_iter = test_iter,
            train_iter_loader = load_data_noniid_mnist,
            dittoEval_loader = load_data_dittoEval_mnist,
            different_client_loader = load_data_dpclient_mnist,
            model=model, 
            loss_func=loss_func,  
            config=config, 
            optimizer=optimizer, 
            device=device, 
            grad_stub=grad_stub
        )

        client.fl_train()
        # client.write_acc_record(fpath="Eva/comb_test.txt", info="clear_round")
