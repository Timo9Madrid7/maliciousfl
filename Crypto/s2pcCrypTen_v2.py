from Crypto.dbscan import EncDBSCAN, DBSCAN
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import torch
import numpy as np


class S2PC():

    def __init__(self, eps1=2., minNumPts1=3, eps2=3., minNumPts2=5):
        crypten.init()
        torch.set_num_threads(1)

        self.cluster_base = EncDBSCAN(eps1, minNumPts1, self)
        self.cluster_lastLayer = EncDBSCAN(eps2, minNumPts2, self)

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

    def notLargerThan_s2pc(self, a:mpc.mpc.MPCTensor, b:mpc.mpc.MPCTensor):
        @mpc.run_multiprocess(world_size=2)
        def notLargerThan(a, b):
            if (a<=b).get_plain_text():
                return True 
            return False
        return notLargerThan(a, b)[0]

    def comparisonReconstruct_s2pc(self, a:mpc.mpc.MPCTensor):
        @mpc.run_multiprocess(world_size=2)
        def comparisonReconstruct(a):
            return a.get_plain_text().numpy().astype(bool)
        return comparisonReconstruct(a)[0]
    
    def cosineFilter_s2pc(self, grads_list_:list, grads_ly_list_:list, precision=24, verbose=True):
        @mpc.run_multiprocess(world_size=2)
        def cosineFilter(grads_list_, grads_ly_list_, precision, verbose):
            grad_share = crypten.cryptensor(grads_list_, precision=precision)
            grad_share_mean = grad_share.mean(axis=0)
            # distance_matrix = crypten.cryptensor([[0. for _ in range(len(grads_list_))] for _ in range(len(grads_list_))], precision=precision)
            # for i in range(len(grads_list_)):
            #     for j in range(i+1, len(grads_list_)):
            #         distance_matrix[i][j] = distance_matrix[j][i] = 1. - ((grad_share[i]-grad_share_mean).dot(grad_share[j]-grad_share_mean))
            distance_matrix = 1. - (grad_share-grad_share_mean).matmul((grad_share-grad_share_mean).transpose(1,0))
            for i in range(len(distance_matrix)):
                distance_matrix[i,i] = 0
            labels = self.cluster_base.fit(distance_matrix).labels_
            filter1_id = self.get_ids(labels)
            grads_ly_filtered = []
            for _id in filter1_id:
                grads_ly_filtered.append(grads_ly_list_[_id])
            
            grad_share = crypten.cryptensor(grads_ly_filtered, precision=precision)
            grad_share_mean = grad_share.mean(axis=0)
            # distance_matrix = crypten.cryptensor([[0. for _ in range(len(grads_ly_filtered))] for _ in range(len(grads_ly_filtered))], precision=precision)
            # for i in range(len(grads_ly_filtered)):
            #     for j in range(i+1, len(grads_ly_filtered)):
            #         distance_matrix[i][j] = distance_matrix[j][i] = 1. - ((grad_share[i]-grad_share_mean).dot(grad_share[j]-grad_share_mean))
            distance_matrix = 1. - (grad_share-grad_share_mean).matmul((grad_share-grad_share_mean).transpose(1,0))
            for i in range(len(distance_matrix)):
                distance_matrix[i,i] = 0
            labels = self.cluster_lastLayer.fit(distance_matrix).labels_
            filter2_id = self.get_ids(labels)
            benign_id = []
            for _id in filter2_id:
                benign_id.append(filter1_id[_id])
            
            if verbose and comm.get().get_rank():
                print("filter 1 id:", filter1_id)
                print("filter 2 id:", benign_id)
            return benign_id

        return cosineFilter(grads_list_, grads_ly_list_, precision, verbose)[0]

    def get_ids(self, label):
        if (label==-1).all():
            bengin_id = np.arange(len(label)).tolist()
        else:
            label_class, label_count = np.unique(label, return_counts=True)
            if -1 in label_class:
                label_class, label_count = label_class[1:], label_count[1:]
            majority = label_class[np.argmax(label_count)]
            bengin_id = np.where(label==majority)[0].tolist()
        return bengin_id


    def cosinedist_correctness_check(self, grads_secrete:list):
        grads_secrete = np.array(grads_secrete)
        grads_secrete_mean = grads_secrete.mean(axis=0)
        distance_matrix = [[0 for _ in range(len(grads_secrete))] for _ in range(len(grads_secrete))]
        for i in range(len(grads_secrete)):
            for j in range(i+1, len(grads_secrete)):
                 distance_matrix[i][j] = distance_matrix[j][i] = 1 - ((grads_secrete[i]-grads_secrete_mean).dot(grads_secrete[j]-grads_secrete_mean))
        return distance_matrix

    def filters_parameters_tuning(self, grads_list_:list, grads_ly_list_:list, verbose=True):
        grad_share = torch.tensor(grads_list_)
        grad_share_mean = grad_share.mean(axis=0)
        distance_matrix = torch.tensor([[0. for _ in range(len(grads_list_))] for _ in range(len(grads_list_))])
        for i in range(len(grads_list_)):
            for j in range(i+1, len(grads_list_)):
                distance_matrix[i][j] = distance_matrix[j][i] = 1. - ((grad_share[i]-grad_share_mean).dot(grad_share[j]-grad_share_mean))

        labels = DBSCAN(0.1, 3).fit(distance_matrix).labels_
        filter1_id = self.get_ids(labels)
        grads_ly_filtered = []
        for _id in filter1_id:
            grads_ly_filtered.append(grads_ly_list_[_id])
        
        grad_share = torch.tensor(grads_ly_filtered)
        grad_share_mean = grad_share.mean(axis=0)
        distance_matrix = torch.tensor([[0. for _ in range(len(grads_ly_filtered))] for _ in range(len(grads_ly_filtered))])
        for i in range(len(grads_ly_filtered)):
            for j in range(i+1, len(grads_ly_filtered)):
                distance_matrix[i][j] = distance_matrix[j][i] = 1. - ((grad_share[i]-grad_share_mean).dot(grad_share[j]-grad_share_mean))

        labels = DBSCAN(0.01, 5).fit(distance_matrix).labels_
        filter2_id = self.get_ids(labels)
        benign_id = []
        for _id in filter2_id:
            benign_id.append(filter1_id[_id])
        
        if verbose:
            print("filter 1 id:", filter1_id)
            print("filter 2 id:", benign_id)
        return benign_id