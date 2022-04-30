import syft as sy
sy.logger.remove()
from sympc.session import Session, SessionManager
from sympc.config import Config
import sympc.protocol as Protocol
from sympc.protocol import ABY3
from sympc.tensor.static import cat
import torch 
import numpy as np 
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", "Temporarily disabling CUDA as FSS does not support it")

class S2PC():

    def __init__(self, eps1=2.5, minNumPts1=3, eps2=3., minNumPts2=5, precision=24, protocol='fss', level='semi-honest'):
        """running S2PC will override torch.Tensor

        Args:
            num_clients (int): number of clients per round
        """
        # SyMPC initialization 
        aggregator_vm = sy.VirtualMachine(name="aggregator")
        server_vm = sy.VirtualMachine(name='external_server')
        tp_vm = sy.VirtualMachine(name="third_party")
        aggregator = aggregator_vm.get_root_client()
        server = server_vm.get_root_client()
        tp = tp_vm.get_root_client()
        
        self.precision = precision
        cfg = Config(encoder_base=2, encoder_precision=self.precision)
        self.parties = [aggregator, server]
        
        self.level = level
        if protocol == "fss":
            share_protocol = Protocol.FSS('semi-honest')
        elif protocol == "falcon":
            share_protocol = Protocol.Falcon(self.level)
            self.parties.append(tp)
            self.aby3 = ABY3(self.level)
        self.protocol = protocol

        self.session = Session(parties=self.parties, protocol=share_protocol, config=cfg)
        SessionManager.setup_mpc(session=self.session)

        self.cluster_base = EncDBSCAN(eps1, minNumPts1, self)
        self.cluster_lastLayer = EncDBSCAN(eps2, minNumPts2, self)

    def secrete_share(self, secrete):
        if type(secrete) != torch.Tensor:
            secrete = torch.tensor(secrete)
        return secrete.share(session=self.session)

    def secrete_reconstruct(self, share):
        return share.reconstruct()

    def share_add(self, share1, share2):
        return share1 + share2
    
    def share_sub(self, share1, share2):
        return share1 - share2

    def share_mul(self, share1, share2):
        return share1 * share2

    def share_dot(self, share1, share2):
        return share1 @ share2

    def share_falcon_le(self, val1, val2):
        return not self.aby3.bit_decomposition_ttp(val2-val1, session=self.session)[-1].reconstruct(decode=False)[0]
      
    def to_distance_matrix(self, grads_share):
        grads_share_mean = sum(grads_share)*(1/len(grads_share)) # SyMPC supports sum()
        distance_matrix = [[self.secrete_share([0.]) for _ in range(len(grads_share))] for _ in range(len(grads_share))]
        cos_info = []
        for i in tqdm(range(len(grads_share))):
            for j in range(i+1, len(grads_share)):
                distance_matrix[i][j] = distance_matrix[j][i] = 1. - self.share_dot(grads_share[i]-grads_share_mean, grads_share[j]-grads_share_mean).view(-1)
            cos_info.append(cat(distance_matrix[i]))
        return cos_info

    def cosineFilter_s2pc(self, grads_list_:list, grads_ly_list_:list, verbose=True):
        grads_share = grads_list_
        print("calculating distance matrix...")
        distance_matrix = self.to_distance_matrix(grads_share)
        print("DBSCAN filtering...")
        labels = self.cluster_base.fit(distance_matrix).labels_
        filter1_id = self.get_ids(labels)
        grads_ly_filtered = []
        for _id in filter1_id:
            grads_ly_filtered.append(grads_ly_list_[_id])

        grads_share = grads_ly_filtered
        print("calculating distance matrix...")
        distance_matrix = self.to_distance_matrix(grads_share)
        print("DBSCAN filtering...")
        labels = self.cluster_lastLayer.fit(distance_matrix).labels_
        filter2_id = self.get_ids(labels)
        benign_id = []
        for _id in filter2_id:
            benign_id.append(filter1_id[_id])
        
        if verbose:
            print("filter 1 id: (%d)"%len(filter1_id), filter1_id)
            print("filter 2 id: (%d)"%len(benign_id), benign_id)
        return benign_id
        
    def aggregation_s2pc(self, grads_list_:list, norms_list_:list, clip_bound:float or None, benign_id:list):
        grads_share = grads_list_
        norms_share = norms_list_
        grads_sum = 0
        bs_sum = 0
        for _id in benign_id:
            if clip_bound == None or self.secrete_reconstruct(norms_share[_id]<=clip_bound).item():
                bs_sum += 1
                grads_sum += self.share_mul(grads_share[_id], norms_share[_id])
            else:
                grads_sum += self.share_mul(grads_share[_id], torch.tensor(clip_bound))
        return self.secrete_reconstruct(grads_sum), bs_sum

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

class EncDBSCAN:
    def __init__(self, radius:float, minPoints:int, s2pc=None):
        self.radius = radius 
        self.minPoints = minPoints
        self.s2pc = s2pc
        
        self.data = None
        self.numPoints = 0
        self._noise = -1
        self._unassigned = -1
        self._core = -2
        self._border = -3
        
        self.labels_, self.core_sample_indices_ = None, None

    def _neighbor_points(self):
        pointGroup = [[] for _ in range(self.numPoints)]
        for i in tqdm(range(self.numPoints)):
            for j in range(i+1, self.numPoints):
                distance = self.s2pc.share_mul(self.data[i]-self.data[j], self.data[i]-self.data[j]).sum().view(-1)
                if (self.s2pc.protocol=="fss" and self.s2pc.secrete_reconstruct(distance.le(self.radius))) or \
                    (self.s2pc.protocol=='falcon' and self.s2pc.share_falcon_le(distance, self.radius)):
                    pointGroup[i].append(j)
                    pointGroup[j].append(i)
        # for i in tqdm(range(self.numPoints)):
        #     tempGroup = []
        #     # if i < 10:
        #     #     print("%d : "%i, end="")
        #     # else:
        #     #     print("%d: "%i, end="")
        #     for j in range(self.numPoints):
        #         distance = sum([self.s2pc.share_mul(self.data[i][k]-self.data[j][k], self.data[i][k]-self.data[j][k]) for k in range(self.numPoints)])
        #         # print("%.3f %r|"%(distance.reconstruct(), self.s2pc.secrete_reconstruct(distance.le(self.radius)).type(torch.bool).item()), end=" ")
        #         if self.s2pc.protocol=="fss" and self.s2pc.secrete_reconstruct(distance.le(self.radius)):
        #             tempGroup.append(j)
        #         elif self.s2pc.protocol=='falcon' and self.s2pc.share_falcon_le(distance, self.radius):
        #             tempGroup.append(j)
        #     # print()
        #     pointGroup.append(tempGroup)
        return pointGroup

    def fit(self, data:list):
        self.data = data 
        self.numPoints = len(self.data)
        pointLabel = [self._unassigned] * self.numPoints
        pointGroup = self._neighbor_points()
        corePoint = []
        nonCorePoint = []
            
        for i in range(len(pointGroup)):
            if len(pointGroup[i]) >= self.minPoints:
                pointLabel[i] = self._core
                corePoint.append(i)
            else:
                nonCorePoint.append(i)
        
        for i in nonCorePoint:
            for j in pointGroup[i]:
                if j in corePoint: 
                    pointLabel[i] = self._border
                    break 
                            
        curClusterID = 0
        for i in range(len(pointLabel)):
            coreGroup = []
            if pointLabel[i] == self._core:
                pointLabel[i] = curClusterID
                for j in pointGroup[i]:
                    if pointLabel[j] == self._core:
                        coreGroup.append(j)
                    pointLabel[j] = curClusterID
                    
                while coreGroup != []:
                    neighbors = pointGroup[coreGroup.pop()]
                    for k in neighbors:
                        if pointLabel[k] == self._core:
                            coreGroup.append(k)
                        pointLabel[k] = curClusterID # It is being said that DBSCAN is not consistent on the border points and depends on which cluster it assigns the point to first.
            
                curClusterID += 1
            
        self.labels_ = np.array(pointLabel, dtype=int)
        self.core_sample_indices_ = np.array(coreGroup, dtype=int)
        return self