from Common.Model.LeNet import LeNet
from Common.Model.ResNet import resnet20 as ResNet
from Common.Model.LeNet import EmnistCNN
from Common.Model.Generator import Generator
from Common.Utils.data_loader import load_data_mnist
from OfflinePack import offline_config as config
import torch
import torchvision.models as models
import os
from torchsummary import summary

if config.reconstruct_inference:
    G_model = Generator()
    if not os.path.exists('./Model/LeNet'):
        os.makedirs('./Model/LeNet')
    torch.save(G_model.state_dict(), './Model/LeNet/InferModelG')

if config.DATASET == "MNIST":
    model = LeNet()
    if not os.path.exists('./Model/LeNet'):
        os.makedirs('./Model/LeNet')
    torch.save(model.state_dict(), config.global_models_path)
    if not os.path.exists('./Model/LeNet/Local_Models'):
        os.makedirs('./Model/LeNet/Local_Models')
    for i in range(config.total_number_clients):
        torch.save(model.state_dict(), config.local_models_path+str(i))
    model_summary = summary(model, (1,28,28))

elif config.DATASET == "CIFAR10":
    model = ResNet()

    if config.pretrained:
        print("load model parameters from the pretrained net")
        # model_parameters = torch.load("./Pretrained_Models/resnet20-12fca82f.th")
        model_parameters = torch.load("./Pretrained_Models/cifar10_resnet20-4118986f.pt")
        layer_parameters = []
        # for key in model_parameters['state_dict'].keys():
        for key in model_parameters.keys():
            if "running" in key or "tracked" in key or "downsample" in key: # to abort centralized training records
                continue
            # layer_parameters.append(model_parameters['state_dict'][key])
            layer_parameters.append(model_parameters[key])
        layer = 0
        for parameters in model.parameters():
            parameters.data = layer_parameters[layer]
            layer += 1

    if not os.path.exists('./Model/ResNet'):
        os.makedirs('./Model/ResNet')
    torch.save(model.state_dict(), config.global_models_path)
    if not os.path.exists("./Model/ResNet/Local_Models"):
        os.makedirs("./Model/ResNet/Local_Models")
    for i in range(config.total_number_clients):
        torch.save(model.state_dict(), config.local_models_path+str(i))
    model_summary = summary(model, (3,32,32))

elif config.DATASET == "EMNIST":
    model = EmnistCNN()
    if not os.path.exists('./Model/EmnistCNN'):
        os.makedirs('./Model/EmnistCNN')
    torch.save(model.state_dict(), config.global_models_path)
    if not os.path.exists('./Model/EmnistCNN/Local_Models'):
        os.makedirs('./Model/EmnistCNN/Local_Models')
    for i in range(config.total_number_clients):
        torch.save(model.state_dict(), config.local_models_path+str(i))
    model_summary = summary(model, (1,28,28))

# # Model info
# model_load = LeNet()
# model_load = models.resnet18()
# model_load = ResNet(BasicBlock, [3,3,3])
# model_load.load_state_dict(torch.load(PATH))
# print(model_load)
# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)
# for name in model.state_dict():
#     print(name)
# layerslen = []
# for p in model.parameters():
#     if p.requires_grad:
#      layerslen.append(p.numel())
# layerslen = [sum(layerslen[:i + 1]) for i in range(len(layerslen))]
# layerslen = layerslen[:-1]
# print(layerslen)
# print(len(layerslen))
# print('parameters_count:',count_parameters(model))
