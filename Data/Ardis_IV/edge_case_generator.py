import numpy as np 
import torch 
import torchvision.transforms as transforms

if __name__ == '__main__':
    x_train, y_train = np.loadtxt('./Data/Ardis_IV/ARDIS_train_2828.csv', dtype=np.float32), np.loadtxt('./Data/Ardis_IV/ARDIS_train_labels.csv').argmax(axis=1)
    x_test, y_test = np.loadtxt('./Data/Ardis_IV/ARDIS_test_2828.csv', dtype=np.float32), np.loadtxt('./Data/Ardis_IV/ARDIS_test_labels.csv').argmax(axis=1)

    x_train_7 = x_train[y_train==7]
    x_train_7 = x_train_7.reshape((x_train_7.shape[0], 28, 28, 1))
    x_test_7 = x_test[y_test==7]
    x_test_7 = x_test_7.reshape((x_test_7.shape[0], 28, 28, 1))

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    edge_case_train_7 = list(zip(map(transformer, x_train_7), [1]*x_train_7.shape[0]))
    edge_case_test_7 = list(zip(map(transformer, x_test_7), [1]*x_test_7.shape[0]))
    edge_case_test_true_7 = list(zip(map(transformer, x_test_7), [7]*x_test_7.shape[0]))
    torch.save(edge_case_train_7, './Data/Ardis_IV/edge_case_train_7.pt')
    torch.save(edge_case_test_7, './Data/Ardis_IV/edge_case_test_7.pt')
    torch.save(edge_case_test_true_7, './Data/Ardis_IV/edge_case_test_true_7.pt')
