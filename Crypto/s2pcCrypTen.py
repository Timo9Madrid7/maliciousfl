import crypten
import torch
import crypten.mpc as mpc
import crypten.communicator as comm

class S2PC():

    def __init__(self):
        crypten.init()
        torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)
    def cosinedist_s2pc(self, grads_secrete:list):
        grad_share = crypten.cryptensor(grads_secrete, precision=30)
        distance_matrix = [[0 for _ in range(len(grads_secrete))] for _ in range(len(grads_secrete))]
        for i in range(len(grads_secrete)):
            for j in range(i+1, len(grads_secrete)):
                distance_matrix[i][j] = distance_matrix[j][i] = (grad_share[i]*grad_share[j]).sum().get_plain_text().item()
        return distance_matrix

    # !: runtime error
    @mpc.run_multiprocess(world_size=2)
    def aggregation_s2pc(self, grads_secrete:list, norms_secrete:list, bs_secrete:list, benign_id:list):
        grads_sum = 0
        bs_sum = 0
        for _id in benign_id:
            grads_sum += (grads_secrete[_id]*norms_secrete[_id])
            if bs_secrete[_id] > 0:
                bs_sum += bs_secrete[_id]
        return (grads_sum/len(benign_id)).get_plain_text(), (bs_sum/len(benign_id)).get_plain_text().item()
    

    