#!/usr/bin/env python
# coding: utf-8

# In[6]:


import torch 
import torchvision
from Common.Model.ResNet import resnet20
from Common.Utils.evaluate import evaluate_accuracy


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = resnet20().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()


# In[5]:


transforms_train = torchvision.transforms.Compose([
   torchvision.transforms.RandomCrop(32, padding=4), 
   torchvision.transforms.RandomHorizontalFlip(), 
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transforms_test = torchvision.transforms.Compose([
   torchvision.transforms.ToTensor(),
   torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

cifar10_train = torchvision.datasets.CIFAR10(root='./Data/CIFAR10/', train=True, download=False, transform=transforms_train)
train_iter = torch.utils.data.DataLoader(cifar10_train, batch_size=64, shuffle=True, num_workers=0)
cifar10_test = torchvision.datasets.CIFAR10(root='./Data/CIFAR10/', train=False, download=False, transform=transforms_test)
test_iter = torch.utils.data.DataLoader(cifar10_test, batch_size=64, shuffle=True, num_workers=0)


# In[ ]:

num_rounds = 10
for epoch in range(num_rounds):
    for X,y in train_iter:
        loss = loss_func(model(X.to(device)), y.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    test_acc = evaluate_accuracy(test_iter, model, device=device)
    print("epoch: %d | loss: %.3f | test acc: %.3f"%(epoch, loss, test_acc))


# In[ ]:




