import syft as sy
sy.logger.remove()
import sympc
from sympc.session import Session, SessionManager
from sympc.config import Config
import sympc.protocol as Protocol
from sympc.protocol import ABY3
from sympc.tensor.static import cat
from sympc.encoder import FixedPointEncoder
import torch 
import numpy as np 
import random
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", "Temporarily disabling CUDA as FSS does not support it")

class S2PC():

    def __init__(self, eps1=2.5, minNumPts1=3, eps2=3., minNumPts2=5, precision=24, protocol='fss', level='semi-honest', is_auto_dbscan=True):
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

        self.fp_encoder = FixedPointEncoder(
            base=self.session.config.encoder_base,
            precision=self.session.config.encoder_precision    
        )

        self.cluster_base = EncDBSCAN(eps1, minNumPts1, self)
        self.cluster_lastLayer = EncDBSCAN(eps2, minNumPts2, self)
        self.is_auto_dbscan = is_auto_dbscan

    def secrete_share(self, secrete):
        if type(secrete) != torch.Tensor:
            secrete = torch.tensor(secrete)
        return secrete.share(session=self.session)

    def secrete_reconstruct(self, share, decode=True):
        return share.reconstruct(decode=decode)

    def share_add(self, share1, share2):
        return share1 + share2
    
    def share_sub(self, share1, share2):
        return share1 - share2

    def share_mul(self, share1, share2):
        return share1 * share2

    def share_dot(self, share1, share2):
        return share1 @ share2

    def public_compare(self, x, r):
        if self.protocol == 'fss':
            return self.secrete_reconstruct(x.le(r))
        elif self.protocol == 'falcon':
            r = self.fp_encoder.encode(r)
            x_b = ABY3.bit_decomposition_ttp(x, session=self.session)
            x_p = [ABY3.bit_injection(bit_share, self.session, sympc.tensor.PRIME_NUMBER) for bit_share in x_b]
            tensor_type = sympc.utils.get_type_from_ring(self.session.ring_size)
            result = Protocol.Falcon.private_compare(x_p, r.type(tensor_type))
            return ~self.secrete_reconstruct(result, decode=False)

    def to_distance_matrix(self, grads_share):
        num_grads = len(grads_share)
        grads_share_cat = cat(grads_share).reshape(num_grads,-1)
        
        cos_info = []
        for i in tqdm(range(num_grads)):
            zero_mask_i = torch.ones(num_grads)
            zero_mask_i[i] = zero_mask_i[i] * 0.
            distance = 1. - self.share_dot((grads_share[i]), grads_share_cat.transpose(1,0))
            distance = self.share_mul(distance, zero_mask_i)
            cos_info.append(distance)
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

    def auto_dbscan(self, distance_list:list):
        def quickMedian(nums:list, k:int):
            if len(nums) == 1: 
                return nums[0]

            ref = (cat(nums) - random.choice(nums)).reconstruct()
            lows = [nums[i] for i in torch.where(ref<0)[0]]
            highs = [nums[i] for i in torch.where(ref>0)[0]]
            pivots = [nums[i] for i in torch.where(ref==0)[0]]

            if k < len(lows):
                return quickMedian(lows, k)
            elif k < len(lows) + len(pivots):
                return pivots[0]
            else:
                return quickMedian(highs, k-len(lows)-len(pivots))

        eps = quickMedian(distance_list, len(distance_list)//2)

        return eps

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
        if not self.s2pc.is_auto_dbscan:
            pointGroup = [[i] for i in range(self.numPoints)]
            distances = []
            for i in tqdm(range(self.numPoints)):
                for j in range(i+1, self.numPoints):
                    distances.append(self.s2pc.share_mul(self.data[i]-self.data[j], self.data[i]-self.data[j]).sum().view(-1))
            distances = cat(distances)
            compare_result = self.s2pc.public_compare(distances, self.radius).tolist()

            for i in range(self.numPoints):
                for j in range(i+1, self.numPoints):
                    if compare_result.pop(0):
                        pointGroup[i].append(j)
                        pointGroup[j].append(i)   
            return pointGroup

        else:
            pointGroup = []
            distance_matrix = [[self.s2pc.secrete_share([0.]) for _ in range(self.numPoints)] for _ in range(self.numPoints)]
            eps_list = []
            for i in tqdm(range(self.numPoints)):
                for j in range(i+1, self.numPoints):
                    distance_matrix[i][j] = distance_matrix[j][i] = self.s2pc.share_mul(self.data[i]-self.data[j], self.data[i]-self.data[j]).sum().view(-1)
                eps_list.append(self.s2pc.auto_dbscan(distance_matrix[i]))
            self.radius = sum(eps_list)*(1/len(eps_list))
            
            for i in range(self.numPoints):
                indices = self.s2pc.secrete_reconstruct(cat(distance_matrix[i]) - self.radius) <= 0
                pointGroup.append(torch.where(indices)[0].tolist())
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