import numpy as np
from fastdtw import fastdtw
import torch 
from dtw import dtw

from ..training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def compute_dtw_fastdtw(x1, x2, distance_function = lambda a, b: abs(a - b), radius = 1):
    distance, _ = fastdtw(x1, x2, radius = radius, dist = distance_function)
    return distance

def compute_dtw_softDTWCuda(x1, x2, device = 'cpu'):

    x1 = torch.asarray(x1).unsqueeze(0).unsqueeze(0)
    x2 = torch.asarray(x2).unsqueeze(0).unsqueeze(0)

    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    distance = float(recon_loss_function(x1, x2).cpu())

    return distance

def compute_dtw_dtwpython(x1, x2):
    algiment = dtw(x1, x2, keep_internals=True)
    
    return algiment.distance

