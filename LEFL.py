from Client import Client
from LeNet import LeNet
from config import num_epoch, num_client, locally_encode_layer
import torch 
import numpy as np
import os

if __name__ == "__main__":

    model = LeNet()
    if not os.path.exists('./Models/'):
        os.makedirs('./Models/')
    torch.save(model.state_dict(), './Models/LeNet')
    for i in range(num_client):
        torch.save(model.state_dict(), './Models/LeNet_'+str(i))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = LeNet().to(device)
    global_model.load_state_dict(torch.load("./Models/LeNet"))
    level_length = [0]
    for param in global_model.parameters():
        level_length.append(param.data.numel() + level_length[-1])
    start_index = level_length[locally_encode_layer]

    print("First %d layer(s) are locally encoded layers, so there are %d local parameters:"%(locally_encode_layer, start_index))
    for epoch in range(num_epoch):
        print(">>> epoch %d started >>>"%epoch)
        params_list_ = []

        for client_id in range(num_client):
            client = Client(client_id=client_id, device=device)
            params_list_.append(client.fl_train(local_epoch=1, verbose=True))

        update_avg = np.mean(params_list_, axis=0).tolist()
        layer = 0
        for param in global_model.parameters():
            if layer >= locally_encode_layer:
                param_avg = update_avg[level_length[layer]-start_index:level_length[layer + 1]-start_index]
                param.data = torch.tensor(param_avg, device=device).view(param.data.size())
            layer += 1 
        torch.save(global_model.state_dict(), "./Models/LeNet")
        print()