import torch
for cid in range(10):

    if cid < 1:
        print(1)
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    elif cid < 5:
        print(2)
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    else:
        print(3)
        device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')