import csv 
import numpy as np

def write_to_csv(path:str, mode='a', data=[]):
    with open(path, mode, newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(data)

def write_to_txt(path:str, mode='a', data=[]):
    with open(path, mode) as f:
        f.write(''.join(str(i)+' ' for i in data)+'\n')

def detect_GAN_raw(path:str, data: np.ndarray, mode='a'):
    last_fl_params = data[:, -850::]
    with open(path, mode) as f:
        np.savetxt(f, last_fl_params)
        f.write('\n')

def save_distance_matrix(path:str, data: np.ndarray, mode='a'):
    with open(path, mode) as f:
        np.savetxt(f, data)