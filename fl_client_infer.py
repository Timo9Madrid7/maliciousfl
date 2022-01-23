import torch
from torch import nn
import grpc
import numpy as np

from Common.Node.workerbase_InferForv3 import WorkerBase as WorkerBaseInfer
from Common.Grpc.fl_grpc_pb2 import GradRequest_Clipping
from Common.Model.LeNet import LeNet
from Common.Utils.data_loader import load_data_fmnist, load_data_cifar10, load_data_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub
import Common.config as config

# OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class ClearDenseClient(WorkerBaseInfer):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id # client id
        self.grad_stub = grad_stub # communication channel
        self.clippingBound = config.initClippingBound # initial clipping bound for a client

    def adaptiveClipping(self, input_gradients):
        '''
        To clip the input gradient, if its norm is larger than the setting
        '''

        if config.gamma == 1 or config._dpoff: # don't clip
            return np.array(input_gradients), 1
        # else: do clipping+noising

        gradients = np.array(input_gradients) # list to numpy.ndarray

        # --- adaptive noise calculation ---
        if config.z_multiplier==0:
            grad_noise = 0
            b_noise = 0
        else: 
            std_b_noise = config.b_noise
            std_g_noise = config.z_multiplier * self.clippingBound # deviation for gradients
            b_noise = np.random.normal(0, std_b_noise)/config.num_workers
            grad_noise = np.random.normal(0, std_g_noise, size=gradients.shape)/config.num_workers
        # --- adaptive noise calculation ---
        
        norm = np.linalg.norm(gradients)        
        if norm > self.clippingBound:
            return gradients * self.clippingBound/np.linalg.norm(gradients) + grad_noise, 0 + b_noise
        else:
            return gradients + grad_noise, 1 + b_noise

    def update(self):
        # clipping gradients before upload to the server
        gradients, b = self.adaptiveClipping(super().get_gradients())

        # upload local gradients and clipping indicator
        res_grad_upd = self.grad_stub.UpdateGrad_Clipping.future(GradRequest_Clipping(id=self.client_id, b=b, grad_ori=gradients))

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

    PATH = './Model/LeNet'
    model = LeNet().to(device)
    #model = ResNet(BasicBlock, [3,3,3]).to(device)
    model.load_state_dict(torch.load(PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()

    train_iter, test_iter = load_data_mnist(id=args.id, batch = args.batch_size, path = args.path), None

    server_grad = config.server1_address + ":" + str(config.port1)

    with grpc.insecure_channel(server_grad, options=config.grpc_options) as grad_channel:
        print("thread [%d]-%s: connect success!"%(args.id, device))
        grad_stub = FL_GrpcStub(grad_channel)

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

        client.fl_train()
        # client.write_acc_record(fpath="Eva/comb_test.txt", info="clear_round")
