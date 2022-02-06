from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist
from Common import config
import torch
import torchvision.models as models
import os
from torchsummary import summary



PATH = config.global_models_path
#PATH = './Model/ResNet18'
#model = models.resnet18()
model = LeNet()
#model = ResNet(BasicBlock, [3,3,3])
torch.save(model.state_dict(), PATH)

if not os.path.exists('./Model/LeNet/Local_Models'):
    os.makedirs('./Model/LeNet/Local_Models')
for i in range(config.total_number_clients):
    torch.save(model.state_dict(), config.local_models_path+str(i))

data_shape = (1,28,28)
temp_model = summary(model, data_shape)

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
