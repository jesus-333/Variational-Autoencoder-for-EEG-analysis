import numpy as np
import fastdtw

from ..training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def compute_dtw_fastdtw(x1, x2, distance_function = None, radius = 1):
    distance, _ = fastdtw(x1, x2, radius = radius, dist = distance_function)
    return distance

def compute_dtw_softDTWCuda(x1, x2, device = 'cpu'):

    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)

    distance = recon_loss_function(x1, x2).cpu()
        

