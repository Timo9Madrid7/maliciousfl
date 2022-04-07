# Utils
from Common.Model.LeNet import LeNet
from Common.Model.ResNet import resnet20 as ResNet
from Common.Utils.data_loader import load_dataset, load_testset
from Common.Utils.data_loader import load_data_mnist, load_data_dpclient_mnist
from Common.Utils.data_loader import load_backdoor, load_backdoor_test, load_flipping, load_flipping_test, load_edgecase, load_edgecase_test
from Common.Utils.evaluate import evaluate_accuracy
from Common.Utils.attackStrategies import krumAttack, trimmedMeanAttack
from Common.Server.server_sympcHandler import AvgGradientHandler
from Crypto.s2pcSyMPC_v2 import S2PC

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

    if config.DATASET == "MNIST":
        global_model = LeNet().to(device)
    elif config.DATASET == "CIFAR10":
        global_model = ResNet().to(device)
    
    level_length = [0]
    for _, param in global_model.state_dict().items():
        level_length.append(param.data.numel() + level_length[-1])
    global_model.load_state_dict(torch.load(config.global_models_path))
    if config.dp_test:
        test_iter = load_data_dpclient_mnist(config.dp_client, noniid=config._noniid)
    else:
        test_iter = load_testset(dataset=config.DATASET, batch=128)
    aggregator = AvgGradientHandler(config, global_model, device, test_iter)

    # secure two-party computation initialize
    s2pc = S2PC(eps1=2., minNumPts1=3, eps2=3., minNumPts2=5, precision=16)

    print(
        'dataset:', config.DATASET, '| total rounds:', config.num_epochs, '| total clients:', config.total_number_clients, '| clients per round:', config.num_workers, '| distribution: %s'%('Non-IID' if config._noniid else 'IID'), 
        '\n',
        '| dpoff:', config._dpoff, '| grad_noise_sigma:', config.grad_noise_sigma, '| b_noise_std:', config.b_noise_std, '| clip_ratio:', config.gamma,
        '\n',
        'malicious clients:', len(config.malicious_clients), '| backdoor clients:', len(config.backdoor_clients), '| flipping clients:', len(config.flipping_clients),
        '\n',
        'krum clients:', len(config.krum_clients), '| trimmedMean clients:', len(config.trimmedMean_clients), '| edge-case clients:', len(config.edge_case_clinets),
        '\n'
    )

    client_dp_counter = 0
    grads_avg = None
    MA_history, BA_history = [], []
    for epoch in range(config.num_epochs):
        print("epoch %d started, %d out of %d clients selected"
            %(epoch, config.num_workers, config.total_number_clients))
        
        # Clients
        client_id_counter = 0
        client_ids_ = random.sample(range(config.total_number_clients), config.num_workers)
        grads_list_ = []
        grads_ly_list_ = []
        norms_list_ = []
        temp_grads_list_ = []
        for client_id in client_ids_:
            client_id = str(client_id)
            if config.dp_test and config.dp_in and client_id == config.dp_client:
                client_dp_counter += 1
                train_iter = load_data_dpclient_mnist(config.dp_client, noniid=config._noniid)
            elif client_id_counter in config.backdoor_clients:
                train_iter = load_backdoor(client_id, dataset=config.DATASET, noniid=config._noniid)
            elif client_id_counter in config.flipping_clients:
                train_iter = load_flipping(client_id, dataset=config.DATASET, noniid=config._noniid)
            elif client_id_counter in config.edge_case_clinets:
                train_iter = load_edgecase(client_id, dataset=config.DATASET, num_edge_case=config.edge_case_num, noniid=config._noniid)
            else:
                train_iter = load_dataset(client_id, dataset=config.DATASET, test=False, batch=128, noniid=config._noniid)
            eval_iter = load_dataset(client_id, dataset=config.DATASET, test=True, batch=128, noniid=config._noniid)
            if config.DATASET == "MNIST":
                local_model = LeNet().to(device)
            elif config.DATASET == "CIFAR10":
                local_model = ResNet().to(device)
            local_model.load_state_dict(torch.load(config.local_models_path+client_id))
            local_optimizer = torch.optim.Adam(local_model.parameters(), config.llr)
            local_loss_func = torch.nn.CrossEntropyLoss()
            model = deepcopy(global_model).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=config.glr)
            loss_func = torch.nn.CrossEntropyLoss()

            if config.reconstruct_inference and int(client_id) == client_ids_[-1]:
                train_iter, _ = load_data_mnist(0, 128, path='./Data/MNIST')
                infer_client = InferClient(model, loss_func, optimizer, train_iter, config, target=config.target, device=device)
                grads_upload = infer_client.fl_train()
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
                    device=device)

                if client_id_counter in config.malicious_clients:
                    grads_upload = client.malicious_random_upload()
                else:
                    grads_upload = client.fl_train(local_epoch=config.local_epoch, verbose=True)

            temp_grads_list_.append(grads_upload)
            client_id_counter += 1

        # Other attack strategies:
        if grads_avg != None: # they start from the second rounds at leaset
            if len(config.krum_clients) != 0: # Krum Attack
                temp_grads_list_ = krumAttack(temp_grads_list_, grads_avg.tolist(), verbose=True)
            if len(config.trimmedMean_clients) !=0: # Trimmed Mean Attack
                temp_grads_list_ = trimmedMeanAttack(temp_grads_list_, grads_avg.tolist())

        for grad in temp_grads_list_:
            grads_norm = np.linalg.norm(grad)
            norm_share = s2pc.secrete_share(secrete=grads_norm)
            norms_list_.append(norm_share)
            grads_unit = list(map(lambda x:x/grads_norm, grad))
            grads_share = s2pc.secrete_share(secrete=grads_unit)
            grads_list_.append(grads_share)
            grads_norm_lastLayer = np.linalg.norm(grad[-config.weight_index::])
            grads_unit_lastLayer = list(map(lambda x:x/grads_norm_lastLayer, grad[-config.weight_index::]))
            grads_ly_share = s2pc.secrete_share(secrete=grads_unit_lastLayer)
            grads_ly_list_.append(grads_ly_share)

        # Server
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Server>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        benign_id = s2pc.cosineFilter_s2pc(grads_list_, grads_ly_list_)
        grads_sum, bs_sum = s2pc.aggregation_s2pc(grads_list_, norms_list_, aggregator.get_clipBound(), benign_id)
        # grads_avg, bs_avg = grads_sum/len(benign_id), bs_sum/len(benign_id) 
        grads_avg, bs_avg = aggregator.add_dpNoise(grads_sum, bs_sum, len(benign_id), verbose=True)
        aggregator.update_clipBound(bs_avg)
        test_accuracy = aggregator.globalmodel_update(grads_avg.tolist())
        MA_history.append(test_accuracy)
        if test_accuracy != None: 
            print("global accuracy:%.3f | "%test_accuracy, end="")
            if aggregator.get_clipBound() != None:
                print("next clipping boundary:%.2f"%aggregator.get_clipBound())
            else:
                print("next clipping boundary:inf")
        if (epoch+1) % 25 == 0: # save global model and MA every 25 rounds
            torch.save(global_model.state_dict(), config.global_models_path)
            with open("./Eva/offline/MA_history.txt", 'a') as f:
                f.write(''.join(str(i)+' ' for i in MA_history)+'\n')
            MA_history = []
        print()
    
    if config.dp_test:
        print("client_%s showed %d times"%(config.dp_client, client_dp_counter))
    if config.backdoor_clients != []:
        backdoor_test_iter = load_backdoor_test(dataset=config.DATASET)
        print("backdoor accuracy: %.3f"%evaluate_accuracy(backdoor_test_iter, global_model, device))
    if config.flipping_clients != []:
        flipping_test_iter = load_flipping_test(dataset=config.DATASET)
        print("flipping accuracy: %.3f"%evaluate_accuracy(flipping_test_iter, global_model, device))
    if config.edge_case_clinets != [] or config.edge_case_test:
        edge_case_test_iter_true_fake, edge_case_test_iter_true = load_edgecase_test(dataset=config.DATASET)
        print("edge cases BA(true->fake): %.3f | MA(true->true): %.3f"%(evaluate_accuracy(edge_case_test_iter_true_fake, global_model, device), evaluate_accuracy(edge_case_test_iter_true, global_model, device)))

    torch.save(global_model.state_dict(), config.global_models_path)
    



    