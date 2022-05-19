from OfflinePack import offline_config as config
import torch 

def krumAttackPartial(w_local:torch.Tensor, w_global:torch.Tensor, threshold=1e-5, eps=1e-3, verbose=True):
    """Attacking Krum with partial knowledge, reference to Local Model Poisoning Attacks to Byzantine-Robust Federated Learning

    Args:
        w_local (torch.Tensor): colluded clients' uploads.
        w_global (torch.Tensor): last round global parameters.
        threshold (_type_, optional): stoping searching threshold that will effect the minimum strength of the attack. Defaults to 1e-5.
        eps (_type_, optional): most distance variance of c compromised local models. Defaults to 1e-3.
        verbose (bool, optional): Defaults to True.

    Returns:
        _type_: Krum-crafted vector(s).
    """
    m, d = w_local.shape
    c = (m-2) // 2
    k = m - c - 2
    sign_vector = (w_local.mean(axis=0)-w_global).sign()
    
    dist = torch.cdist(w_local, w_local, p=2)
    kNNDist, kNN = torch.topk(dist, k=k+1, largest=False) # k+1 excludes self
    kNNDist_sum_min = kNNDist.sum(axis=1).min().item()
    local_global_max = torch.cdist(w_local, w_global.view(1,-1)).max().item()
    lambd_init = 1/((m-2*c-1)*d**0.5)*kNNDist_sum_min + 1/(d**0.5)*local_global_max
    
    inserted_num = 1
    lambd = lambd_init * 2
    w_craft = (w_global - lambd*sign_vector/2)
    while(lambd>=threshold):
        if verbose:
            print(".", end="")
        lambd /= 2
        w_craft = (w_global - lambd*sign_vector)
        w_crafts = torch.cat([w_craft.view(1,-1)]*inserted_num)
        w_local_craft = torch.cat((w_crafts, w_local))
        
        # krum selection
        dist = torch.cdist(w_local_craft, w_local_craft, p=2)
        kNNDist, kNN = torch.topk(dist, k=k+1, largest=False)
        
        if kNNDist.sum(axis=1).argmin() == 0 and lambd>=threshold:
            break
        elif lambd < threshold:
            inserted_num += 1
            lambd = lambd_init
    return [w_craft.tolist()] + [(w_craft+eps*torch.rand(w_craft.shape)/(w_craft.shape[0]**0.5)).tolist() for _ in range(m-1)]

def krumAttack(grads_list_, grads_avg, threshold=1e-5, eps=1e-3, verbose=True):
    if verbose:
        print("Krum uploads are crafting", end="")
    colluded_grads = torch.tensor([grads_list_[i] for i in config.krum_clients])
    krum_grads = krumAttackPartial(colluded_grads, torch.tensor(grads_avg), threshold, eps, verbose)
    for i in range(len(config.krum_clients)): # replace the benign uploads to Krum's
        grads_list_[config.krum_clients[i]] = krum_grads[i]
    if verbose:
        print(" succeeded")
    return grads_list_

def trimmedMeanAttackPartial(w_local:torch.Tensor, w_global:torch.Tensor):
    """Attacking Trimmed Mean with partial knowledge, reference to Local Model Poisoning Attacks to Byzantine-Robust Federated Learning.
    The random selection ranges are between [mu +3sigma, mu +4sigma] and [mu -4sigma, mu -3sigma].

    Args:
        w_local (torch.Tensor): colluded clients' uploads.
        w_global (torch.Tensor): last round global parameters.

    Returns:
        _type_: trimmedMean-crafted vector(s).
    """
    sign_vector = (w_local.mean(axis=0)-w_global).sign()
    params_mean, params_std = w_local.mean(axis=0), w_local.std(axis=0)
    return [(params_mean - sign_vector*(3*params_std+params_std*torch.rand(params_mean.shape))).tolist() for _ in range(w_local.shape[0])]

def trimmedMeanAttack(grads_list_, grads_avg):
    colluded_grads = torch.tensor([grads_list_[i] for i in config.trimmedMean_clients])
    trimmedMean_grads = trimmedMeanAttackPartial(colluded_grads, torch.tensor(grads_avg))
    for i in range(len(config.trimmedMean_clients)): # replace the benign uploads to trimmed-mean's
        grads_list_[config.trimmedMean_clients[i]] = trimmedMean_grads[i]
    return grads_list_