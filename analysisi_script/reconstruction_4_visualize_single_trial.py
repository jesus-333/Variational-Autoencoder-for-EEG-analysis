"""
Visualize (in time of frequency domain) the reconstruction of a single channel of a single eeg trial
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.config import config_dataset as cd
from library.config import config_model as cm
from library.dataset import preprocess as pp
from library.training import train_generic
from library.training.soft_dtw_cuda import SoftDTW

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

if len(sys.argv) > 1:
    tot_epoch_training = sys.argv[1]
    subj = sys.argv[2]
    repetition = sys.argv[3]
    epoch = sys.argv[4] 
    use_test_set = sys.argv[5] 
else:
    tot_epoch_training = 80
    subj = 2
    repetition = 5
    epoch = 25
    use_test_set = False

batch_size = 64
device = 'cuda' if torch.cuda.is_available() else 'cpu'


