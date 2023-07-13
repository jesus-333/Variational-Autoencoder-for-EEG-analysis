"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Minor support function used in the various script
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def split_dataset(full_dataset, percentage_split : float):
    """
    Split a dataset in 2 
    """

    size_1 = int(len(full_dataset) * percentage_split) 
    size_2 = len(full_dataset) - size_1
    dataset_1, dataset_2 = torch.utils.data.random_split(full_dataset, [size_1, size_2])
    
    return dataset_1, dataset_2

def get_sweep_path(sweep_id):
    """
    In this function, for each sweep I saved the path where the network weights were saved
    Note that in the path the file name is also included the file name is also included, apart from the era and extension
    E.g. if the weight are saved in 'saved_weights/model_weight_30.pth' where 30 is the epoch at the moment the weights were saved and pth is the extension of the file this function will return 'saved_weights/model_weight_'
    """

    weight_path = {
        'jesus_333/ICT4AWE_Extension/wjim0nwt' : 'TMP_Folder/model_',
    }

    return weight_path[sweep_id]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
