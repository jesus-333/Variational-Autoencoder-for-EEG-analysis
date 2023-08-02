import numpy as np
from fastdtw import fastdtw
import torch 
from dtw import dtw

from ..training.soft_dtw_cuda import SoftDTW
from ..training.loss_function import compute_dtw_loss_along_channels 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def compute_dtw_fastdtw(x1, x2, distance_function = lambda a, b: abs(a - b), radius = 1):
    distance, _ = fastdtw(x1, x2, radius = radius, dist = distance_function)
    return distance

def compute_dtw_softDTWCuda(x1, x2, device = 'cpu'):
    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    if len(x1.shape) == 4: # Batch input
        distance = compute_dtw_loss_along_channels(x1, x2, recon_loss_function)
    else:

        x1 = torch.asarray(x1).unsqueeze(0).unsqueeze(0)
        x2 = torch.asarray(x2).unsqueeze(0).unsqueeze(0)

        distance = float(recon_loss_function(x1, x2).cpu())

    return distance

def compute_dtw_dtwpython(x1, x2):
    algiment = dtw(x1, x2, keep_internals=True)
    
    return algiment.distance

