import syft as sy
sy.logger.remove()
from sympc.session import Session, SessionManager
import torch 
import time 

class S2PC():

    def __init__(self, num_clients:int):
        """running S2PC will override torch.Tensor

        Args:
            num_clients (int): number of clients per round
        """
        # SyMPC initialization 
        aggregator_vm = sy.VirtualMachine(name="aggregator")
        server_vm = sy.VirtualMachine(name='external_server')
        aggregator = aggregator_vm.get_root_client()
        server = server_vm.get_root_client()
        
        self.num_clients = num_clients
        self.parties = [aggregator, server]
        self.session = Session(parties=self.parties)
        SessionManager.setup_mpc(session=self.session)

    def secrete_share(self, secrete):
        if type(secrete) != torch.Tensor:
            secrete = torch.tensor(secrete)
        return secrete.share(session=self.session)

    def distanceMatrix_reconstruct(self, distance_matrix):
        start = time.time()
        matrix = [[0 for _ in range(self.num_clients)] for _ in range(self.num_clients)]
        for i in range(self.num_clients):
            print(",", end="")
            for j in range(i+1, self.num_clients):
                matrix[i][j] = matrix[j][i] = distance_matrix[i][j].reconstruct().item()
        print("s2pc cosine distance reconstructed %.1f"%(time.time()-start))
        return matrix

    def gradsAvg_reconstruct(self, grads_tensor):
        print(''.join([',']*self.num_clients), end='')
        start = time.time()
        grads_avg = grads_tensor.reconstruct()
        print("s2pc aggregation reconstructed %.1f"%(time.time()-start))
        return grads_avg

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
    
    def share_div(self, share1, share2):
        return share1 / share2