import crypten.mpc as mpc
from crypten.mpc import MPCTensor
import torch
import crypten
import numpy as np

UNCLASSIFIED = False
NOISE = -1
ws =2

def _dist(p, q):
    #return (p - q).square().sum().sqrt()
    return (p-q).norm(p=2, dim=None, keepdim=False)

def _eps_neighborhood(p,q,eps):
	return _dist(p,q) < eps

def _region_query(m, point_id, eps, n_points, distance_enc):
    seeds = [point_id]
    for i in range(n_points):
        if i == point_id:
            continue
        if (distance_enc[point_id, i] < eps).get_plain_text() == 1:
            seeds.append(i)
    return seeds

def _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points, n_points, distance_enc):

    seeds = _region_query(m, point_id, eps, n_points, distance_enc)
    if len(seeds) < min_points:
        classifications[point_id] = NOISE
        return False
    else:
        classifications[point_id] = cluster_id
        for seed_id in seeds:
            classifications[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            results = _region_query(m, current_point, eps, n_points, distance_enc)
            if len(results) >= min_points:
                for i in range(0, len(results)):
                    result_point = results[i]
                    if classifications[result_point] == UNCLASSIFIED or \
                            classifications[result_point] == NOISE:
                        if classifications[result_point] == UNCLASSIFIED:
                            seeds.append(result_point)
                        classifications[result_point] = cluster_id
            seeds = seeds[1:]
        return True

def dbscan(m, eps, min_points):

    cluster_id = 1
    n_points = m.shape[1]
    classifications = [UNCLASSIFIED] * n_points

    distance = torch.zeros((n_points,n_points))
    distance_enc = MPCTensor(distance, ptype=crypten.mpc.arithmetic)
    for i in range(n_points-1):
        for j in range(i + 1, n_points):
                distance_enc[i,j] = _dist(m[:, i], m[:, j])
                distance_enc[j,i] = distance_enc[i,j]

    for point_id in range(0, n_points):
        #point = m[:, point_id]
        if classifications[point_id] == UNCLASSIFIED:
            if _expand_cluster(m, classifications, point_id, cluster_id, eps, min_points, n_points, distance_enc):
                cluster_id = cluster_id + 1
    return classifications

@mpc.run_multiprocess(world_size=ws)
def mpc_dbscan(data):
    nc = data.shape[0]
    data_enc = MPCTensor(data, ptype=crypten.mpc.arithmetic)
    distance_matrix = torch.ones((nc, nc))
    distance_enc = MPCTensor(distance_matrix, ptype=crypten.mpc.arithmetic)
    cos = crypten.nn.CosineSimilarity(dim=0, eps=1e-6)

    for i in range(nc - 1):
        for j in range(i + 1, nc):
            # numerator_enc = (data_enc[i]*data_enc[j]).sum()
            # # numerator = numerator_enc.get_plain_text()
            # # print(numerator)
            # denominator_enc = data_enc[i].norm(p=2, dim=None, keepdim=False)*data_enc[j].norm(p=2, dim=None, keepdim=False)
            # # denominator = denominator_enc.get_plain_text()
            # # print(denominator)
            # distance_enc[i,j] = numerator_enc/denominator_enc
            distance_enc[i, j] = cos(data_enc[i],data_enc[j])
            distance_enc[j, i] = distance_enc[i, j]

    label = dbscan(distance_enc, eps=0.2, min_points=2)

    label = np.array(label).squeeze()

    split_label = []
    if (label == -1).all():
        split_label = [[i for i in range(nc)]]
    else:
        label_elements = np.unique(label)
        for i in label_elements.tolist():
            split_label.append(np.where(label == i)[0].tolist())
        # b = np.where(label == 0)[0].tolist()
    # euclidean distance between self.weights and clients' weights
    weight_agg = []
    weights_in_mpc = data
    for b in split_label:
        weights_enc = MPCTensor(weights_in_mpc[b], ptype=crypten.mpc.arithmetic)
        weight_agg.append(weights_enc.mean(dim=0).get_plain_text().numpy())
    # self.weights = weight_agg

    weight_aggg = np.array(weight_agg).flatten()

    print(weight_aggg)
    print(split_label)
    return weight_aggg

# @mpc.run_multiprocess(world_size=ws)
# def mpc_mean(data):
#     data_enc = MPCTensor(data, ptype=crypten.mpc.arithmetic)
#     data_avg = data_enc.mean(dim=0)
#     return data_avg.get_plain_text()