import crypten
import torch
import numpy as np
import crypten.mpc as mpc
import crypten.communicator as comm

class S2PC():

    def __init__(self):
        crypten.init()
        torch.set_num_threads(1)

    @mpc.run_multiprocess(world_size=2)
    def cosinedist_s2pc(self, grads_secrete:list, precision=24, correctness_check=False):
        grad_share = crypten.cryptensor(grads_secrete, precision=precision)
        grad_share_mean = grad_share.mean(axis=0)
        distance_matrix = [[0 for _ in range(len(grads_secrete))] for _ in range(len(grads_secrete))]
        for i in range(len(grads_secrete)):
            for j in range(i+1, len(grads_secrete)):
                distance_matrix[i][j] = distance_matrix[j][i] = torch.subtract(
                  torch.tensor([1.]), ((grad_share[i]-grad_share_mean).dot(grad_share[j]-grad_share_mean)).get_plain_text()).item()
        
        if correctness_check and comm.get().get_rank():
            distance_matrix_compare = self.cosinedist_correctness_check(grads_secrete)
            for i in range(len(grads_secrete)):
                print("| ", end="")
                for j in range(len(grads_secrete)):
                    print("%.6f | "%(np.abs(distance_matrix[i][j]-distance_matrix_compare[i][j])), end="")
                print()
        
        return distance_matrix

    def aggregation_s2pc(self, grads_secrete:list, norms_secrete:list, clip_bound:float or None, benign_id:list, precision=24):
        @mpc.run_multiprocess(world_size=2)
        def aggregation(grads_secrete:list, norms_secrete:list, clip_bound:float or None, benign_id:list, precision:int):
            grads_share = crypten.cryptensor(grads_secrete, precision=precision)
            norms_share = crypten.cryptensor(norms_secrete, precision=precision)
            grads_sum = 0
            for _id in benign_id:
                if clip_bound == None or (norms_share[_id]<=clip_bound).get_plain_text().item():
                    grads_sum += grads_share[_id].mul(norms_share[_id])
                else:
                    grads_sum += grads_share[_id].mul(torch.tensor(clip_bound))
            grads_sum = grads_sum.get_plain_text()
            torch.save(grads_sum.type(torch.float64), "./temp.pt")
            if clip_bound != None:
                return (crypten.cryptensor(norms_secrete)[benign_id] <= clip_bound).sum().get_plain_text().item()
            else:
                return len(benign_id)
        bs_sum, _ = aggregation(grads_secrete, norms_secrete, clip_bound, benign_id, precision)
        grads_sum = torch.load('./temp.pt')
        return grads_sum, bs_sum
    
    def cosinedist_correctness_check(self, grads_secrete:list):
        grads_secrete = np.array(grads_secrete)
        grads_secrete_mean = grads_secrete.mean(axis=0)
        distance_matrix = [[0 for _ in range(len(grads_secrete))] for _ in range(len(grads_secrete))]
        for i in range(len(grads_secrete)):
            for j in range(i+1, len(grads_secrete)):
                 distance_matrix[i][j] = distance_matrix[j][i] = 1 - ((grads_secrete[i]-grads_secrete_mean).dot(grads_secrete[j]-grads_secrete_mean))
        return distance_matrix