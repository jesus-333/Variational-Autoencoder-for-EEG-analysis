"""
Scale and shift the signal of n std, with the std compute from the train set and evaluate the reconstruction error.
Since the shift and scale change the amplitude of signal, before the error computation both the original and the reconstructed signal are scaled between 0 and 1
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.dataset import dataset_time as ds_time
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj = 3

epoch = 80
use_test_set = False
repetition = 5

# How many the time use the std to scale/shift the signal
n_change_step = 1
n_change_list = (np.arange(10)) * n_change_step

use_dtw_divergence = True

batch_size = 72
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path matrix
path_avg_matrix = 'Saved Results/d2a_analysis/shift_and_scale_error/S{}/{}/'.format(subj, string_dataset)
