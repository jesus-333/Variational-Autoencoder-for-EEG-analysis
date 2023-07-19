"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Minor support function used in the various script
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_idx_to_split_data(n_elements : int, percentage_split : float, seed = -1):
    """
    Get to list of indices to split an array of data.
    """
    # Use of the seed for reproducibility
    if seed != -1: np.random.seed(42)
    
    # Create idx vector
    idx = np.random.permutation(n_elements)
    size_1 = int(n_elements * percentage_split) 
    
    return idx[0:size_1], idx[size_1:]


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
