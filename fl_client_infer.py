# GRPC
import grpc
from Common.Grpc.fl_grpc_pb2_grpc import FL_GrpcStub
from Common.Grpc.fl_grpc_pb2 import GradRequest_Clipping

# Utils 
from Common.Node.workerbase_InferForv3 import WorkerBase as WorkerBaseInfer
from Common.Model.LeNet import LeNet
from Common.Utils.data_loader import load_data_mnist
from Common.Utils.set_log import setup_logging
from Common.Utils.options import args_parser

# Settings
import Common.config as config

# Other Libs
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized

class ClearDenseClient(WorkerBaseInfer):
    def __init__(self, client_id, model, loss_func, train_iter, test_iter, config, optimizer, device, grad_stub):
        super(ClearDenseClient, self).__init__(model=model, loss_func=loss_func, train_iter=train_iter,
                                               test_iter=test_iter, config=config, optimizer=optimizer, device=device)
        self.client_id = client_id # client id
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
        gradients = np.array(input_gradients) # list to numpy.ndarray
        norm = np.linalg.norm(gradients)        
        if norm > self.clippingBound:
            return gradients * self.clippingBound/np.linalg.norm(gradients), 0
        else:
            return gradients, 1

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

    PATH = config.global_models_path
    model = LeNet().to(device)
    #model = ResNet(BasicBlock, [3,3,3]).to(device)
    model.load_state_dict(torch.load(PATH))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_func = torch.nn.CrossEntropyLoss()

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
