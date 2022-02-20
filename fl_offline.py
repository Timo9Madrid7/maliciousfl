# Utils
from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist, load_data_noniid_mnist, load_data_dittoEval_mnist, load_all_test_mnist, load_data_dpclient_mnist
from Common.Utils.data_loader import load_data_backdoor_mnist, load_data_backdoor_mnist_test, load_data_flipping_mnist, load_data_flipping_mnist_test
from Common.Utils.data_loader import load_data_noniid_cifar10, load_data_dittoEval_cifar10
from Common.Utils.evaluate import evaluate_accuracy
from Common.Server.server_cryptoHandler import AvgGradientHandler
from Crypto.s2pc import S2PC

# Offline Packages
import OfflinePack.offline_config as config
from OfflinePack.client import OfflineClient
from OfflinePack.infer_client import InferClient

# Other Libs
import torch
import random
import numpy as np
from copy import deepcopy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    global_model = LeNet().to(device)
    level_length = [0]
    for param in global_model.parameters():
        level_length.append(param.data.numel() + level_length[-1])
    global_model.load_state_dict(torch.load(config.global_models_path))
    if config.dp_test:
        test_iter = load_data_dpclient_mnist(config.dp_client, noniid=config._noniid)
    else:
        test_iter = load_all_test_mnist()
    aggregator = AvgGradientHandler(config, global_model, device, test_iter)
    clippingBound = config.initClippingBound

    # secure two-party computation initialize
    s2pc = S2PC(num_clients=config.num_workers)

    print('model:', config.Model, '| dpoff:', config._dpoff, ' | dpcompen:', config._dpcompen,
    '| grad_noise_sigma:', config.grad_noise_sigma, '| b_noise_std:', config.b_noise_std, '| clip_ratio:', config.gamma,
    '| malicious clients:', len(config.malicious_clients), '| backdoor clients:', len(config.backdoor_clients), '| flipping clients:', len(config.flipping_clients),
    '\n')

    client_dp_counter = 0
    for epoch in range(config.num_epochs):
        print("epoch %d started, %d out of %d clients selected"
            %(epoch, config.num_workers, config.total_number_clients))
        
        # Clients
        client_id_counter = 0
        client_ids_ = random.sample(range(120), config.num_workers)
        b_list_ = []
        grads_list_ = []
        grads_ly_list_ = []
        norms_list_ = []
        for client_id in client_ids_:
            client_id = str(client_id)
            if config.dp_test and config.dp_in and client_id == config.dp_client:
                client_dp_counter += 1
                train_iter = load_data_dpclient_mnist(config.dp_client, noniid=config._noniid)
            elif client_id_counter in config.backdoor_clients:
                train_iter = load_data_backdoor_mnist(client_id, noniid=config._noniid)
            elif client_id_counter in config.flipping_clients:
                train_iter = load_data_flipping_mnist(client_id, noniid=config._noniid)
            else:
                train_iter = load_data_noniid_mnist(client_id, noniid=config._noniid)
            eval_iter = load_data_dittoEval_mnist(client_id, noniid=config._noniid)
            local_model = LeNet().to(device)
            local_model.load_state_dict(torch.load(config.local_models_path+client_id))
            local_optimizer = torch.optim.Adam(local_model.parameters(), config.llr)
            local_loss_func = torch.nn.CrossEntropyLoss()
            model = deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.glr)
            loss_func = torch.nn.CrossEntropyLoss()

            if config.reconstruct_inference and int(client_id) == client_ids_[-1]:
                train_iter, _ = load_data_mnist(0, 128, path='./Data/MNIST')
                infer_client = InferClient(model, loss_func, optimizer, train_iter, config, target=config.target, device=device)
                grads_raw = infer_client.fl_train()
                grads_dp, b_dp = infer_client.adaptiveClipping(grads_raw)
            else:
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

                if client_id_counter in config.malicious_clients:
                    grads_dp, b_dp = client.malicious_random_upload()
                else:
                    grads_raw = client.fl_train(local_epoch=config.local_epoch, verbose=True)
                    grads_dp, b_dp = client.adaptiveClipping(grads_raw)

            grads_norm = np.linalg.norm(grads_dp)
            grads_unit = list(map(lambda x:x/grads_norm, grads_dp))
            grads_norm_lastLayer = np.linalg.norm(grads_dp[-config.weight_index::])
            grads_unit_lastLayer = list(map(lambda x:x/grads_norm_lastLayer, grads_dp[-config.weight_index::]))
            norm_share = s2pc.secrete_share(secrete=grads_norm)
            grads_share = s2pc.secrete_share(secrete=grads_unit)
            grads_ly_share = s2pc.secrete_share(secrete=grads_unit_lastLayer)
            b_share = s2pc.secrete_share(secrete=b_dp)
            
            grads_list_.append(grads_share)
            grads_ly_list_.append(grads_ly_share)
            norms_list_.append(norm_share)
            b_list_.append(b_share)
            client_id_counter += 1
        
        # Server
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Server>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        cosinedist_hidden_1 = aggregator.cosinedist_s2pc(grads_list_)
        cosinedist_plain_1 = s2pc.distanceMatrix_reconstruct(cosinedist_hidden_1)
        filter1_id = aggregator.cosine_distance_filter(np.array(cosinedist_plain_1), cluster_sel=0)
        print("filter 1 id:", filter1_id)
        grads_ly_filtered = []
        for _id in filter1_id:
            grads_ly_filtered.append(grads_ly_list_[_id])
        cosinedist_hidden_2 = aggregator.cosinedist_s2pc(grads_ly_filtered)
        cosinedist_plain_2 = s2pc.distanceMatrix_reconstruct(cosinedist_hidden_2)
        filter2_id = aggregator.cosine_distance_filter(np.array(cosinedist_plain_2), cluster_sel=1)
        benign_id = []
        for _id in filter2_id:
            benign_id.append(filter1_id[_id])
        print("filter 2 id:", benign_id)
        gradsAvg_hidden, bAvg_hidden = aggregator.aggregation_s2pc(grads_list_, norms_list_, b_list_, benign_id)
        gradsAvg_plain, bAvg_plain = s2pc.avg_reconstruct(gradsAvg_hidden, bAvg_hidden)
        clippingBound = aggregator.adaptive_clipping(bAvg_plain, clippingBound, config.gamma, config.blr)
        test_accuracy = aggregator.globalmodel_update(gradsAvg_plain.cpu().numpy().tolist())
        if test_accuracy != None: 
            print("global accuracy:%.3f"%test_accuracy)
        print()
    
    if config.dp_test:
        print("client_%s showed %d times"%(config.dp_client, client_dp_counter))
    if config.backdoor_clients != []:
        backdoor_test_iter = load_data_backdoor_mnist_test()
        print("backdoor accuracy: %.3f"%evaluate_accuracy(backdoor_test_iter, global_model, device))
    if config.flipping_clients != []:
        flipping_test_iter = load_data_flipping_mnist_test()
        print("flipping accuracy: %.3f"%evaluate_accuracy(flipping_test_iter, global_model, device))

    torch.save(global_model.state_dict(), config.global_models_path)
    



    