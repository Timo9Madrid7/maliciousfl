# Utils
from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_noniid_mnist, load_data_dittoEval_mnist, load_all_test_mnist
from Common.Utils.data_loader import load_data_noniid_cifar10, load_data_dittoEval_cifar10
from Common.Utils.evaluate import evaluate_accuracy
from OfflinePack.client import OfflineClient

# Settings
import OfflinePack.offline_config as config

# Other Libs
import torch
import random
from copy import deepcopy
import numpy as np

def upgrade(grad_in:list, model):
    layer = 0
    for param in model.parameters():
        layer_diff = grad_in[level_length[layer]:level_length[layer + 1]]
        param.data += torch.tensor(layer_diff, device=device).view(param.data.size())
        layer += 1    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    global_model = LeNet().to(device)
    level_length = [0]
    for param in global_model.parameters():
        level_length.append(param.data.numel() + level_length[-1])
    global_model.load_state_dict(torch.load(config.global_models_path))
    clippingBound = config.initClippingBound
    test_iter = load_all_test_mnist()

    for epoch in range(config.num_epochs):
        print("epoch %d started, %d out of %d clients selected"
            %(epoch, config.num_workers, config.total_number_clients))
        
        client_ids_ = random.sample(range(120), config.num_workers)

        b_list_ = []
        grads_list_ = []
        for client_id in client_ids_:
            client_id = str(client_id)
            train_iter = load_data_noniid_mnist(client_id, noniid=config._noniid)
            eval_iter = load_data_dittoEval_mnist(client_id, noniid=config._noniid)
            local_model = LeNet().to(device)
            local_model.load_state_dict(torch.load(config.local_models_path+client_id))
            local_optimizer = torch.optim.Adam(local_model.parameters(), config.llr)
            local_loss_func = torch.nn.CrossEntropyLoss()
            model = deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss_func = torch.nn.CrossEntropyLoss()

            client = OfflineClient(
                client_id=client_id,
                train_iter=train_iter,
                eval_iter=eval_iter,
                model=model,
                loss_func=loss_func,
                optimizer=optimizer,
                local_model=local_model,
                local_loss_func=local_loss_func,
                local_optimizer=local_optimizer,
                config=config,
                device=device,
                clippingBound=clippingBound)

            grads_list_.append(client.fl_train(local_epoch=config.local_epoch, verbose=True))
        
        print("testing accuracy %.3f"%evaluate_accuracy(test_iter, global_model, device=device))
        print()
        assert len(grads_list_) == config.num_workers

        grad_in = np.array(grads_list_).reshape((config.num_workers, -1))
        grad_out = grad_in.mean(axis=0)
        upgrade(grad_out.tolist(), global_model)
        clippingBound = None
    




    