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

def get_idx_to_split_data(n_elements : int, percentage_split : float, seed : int = None):
    """
    Given a number of elements (n_elements) create an array with number from 0 to n_elements - 1 and split it (randomly) in two lists.
    The size of the two list is determined by the percentage_split parameter. The first list will be have size x = int(percentage_split * n_elements) while the second will have size y = n_elements - x
    The procedure can be "deterministic" if the seed parameter is passed to the function.
    """
    
    # Check input parameter
    if n_elements <= 1 : raise ValueError("n_elements must be greater than 1. Current value is {}".format(n_elements))
    if percentage_split <= 0 or percentage_split >= 1 : raise ValueError("percentage_split must be between 0 and 1. Current value is {}".format(percentage_split))

    # Use of the seed for reproducibility
    if seed is not None : np.random.seed(seed)

    # Create idx vector
    idx = np.random.permutation(n_elements)
    size_1 = int(n_elements * percentage_split) 
    
    return idx[0:size_1], idx[size_1:]

# NOT USED
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
