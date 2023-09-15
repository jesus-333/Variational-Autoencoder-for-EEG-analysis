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

from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
subj = 2

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model([subj])

