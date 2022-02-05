# GRPC
from Common.Server.fl_grpc_server_adaclipping import FlGrpcServer as FLGrpcClipServer
from Common.Grpc.fl_grpc_pb2 import GradResponse_Clipping

# Utils
from Common.Server.server_handler import AvgGradientHandler
from Common.Utils.data_loader import load_all_test_mnist
from Common.Model.LeNet import LeNet

# Settings
import Common.config as config

# Other Libs
import torch

class ClearDenseServer(FLGrpcClipServer):
    def __init__(self, address, port, config, handler):
        super(ClearDenseServer, self).__init__(config=config)
        self.address = address
        self.port = port
        self.config = config
        self.handler = handler

        self.clippingBound = self.config.initClippingBound

    # override UpdateGrad_float for server
    # receive gradients from clients, aggregate, and give them back
    def UpdateGrad_Clipping(self, request, context):
        data_dict = {request.id: request.grad_ori}
        b_list = [request.b]
        # print(data_dict.keys(), np.round(b_list,4), 'clip_b:', np.round(self.clippingBound,4))
        rst, self.clippingBound = super().process(dict_data=data_dict, b=b_list, handler=self.handler.computation, clippingBound=self.clippingBound)
        
        return GradResponse_Clipping(b=self.clippingBound, grad_upd=rst)

if __name__ == "__main__":

    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu') 
    model = LeNet().to(device)
    model.load_state_dict(torch.load(config.global_models_path))
    test_iter = load_all_test_mnist()

    gradient_handler = AvgGradientHandler(config=config, model=model, device=device, test_iter=test_iter)

    clear_server = ClearDenseServer(address=config.server1_address, port=config.port1, config=config,
                                    handler=gradient_handler)
    print('ratio %d/%d:'%(config.num_workers, config.total_number_clients), '| dpoff:', config._dpoff, ' | dpcompen:', config._dpcompen,
    '| b_noise_std:', config.b_noise_std, '| clip_ratio:', config.gamma, '| grad_noise_sigma:', config.grad_noise_sigma)
    clear_server.start()
