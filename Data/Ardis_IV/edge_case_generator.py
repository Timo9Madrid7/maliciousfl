import numpy as np 
import torch 

if __name__ == '__main__':
    x_train, y_train = np.loadtxt('./Data/Ardis_IV/ARDIS_train_2828.csv'), np.loadtxt('./Data/Ardis_IV/ARDIS_train_labels.csv').argmax(axis=1)
    x_test, y_test = np.loadtxt('./Data/Ardis_IV/ARDIS_test_2828.csv'), np.loadtxt('./Data/Ardis_IV/ARDIS_test_labels.csv').argmax(axis=1)

    x_train_7 = x_train[y_train==7]
    x_train_7 = x_train_7.reshape((x_train_7.shape[0], 1, 28, 28))
    x_test_7 = x_test[y_test==7]
    x_test_7 = x_test_7.reshape((x_test_7.shape[0], 1, 28, 28))

    edge_case_train_7 = list(zip(torch.from_numpy(x_train_7), [1]*x_train_7.shape[0]))
    edge_case_test_7 = list(zip(torch.from_numpy(x_test_7), [1]*x_test_7.shape[0]))
    torch.save(edge_case_train_7, './Data/Ardis_IV/edge_case_train_7.pt')
    torch.save(edge_case_test_7, './Data/Ardis_IV/edge_case_test_7.pt')
