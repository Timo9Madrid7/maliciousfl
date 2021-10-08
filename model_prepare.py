from Common.Model.LeNet import LeNet
from Common.Model.ResNet import ResNet, BasicBlock
from Common.Utils.data_loader import load_data_mnist
import torch
import torchvision

PATH = './Model/LeNet'
model = LeNet()
#model = ResNet(BasicBlock, [3,3,3])
torch.save(model.state_dict(), PATH)
model_load = LeNet()
#model_load = ResNet(BasicBlock, [3,3,3])
model_load.load_state_dict(torch.load(PATH))
print(model_load)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('parameters_count:',count_parameters(model))