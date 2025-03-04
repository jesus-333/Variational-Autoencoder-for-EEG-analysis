"""
@author : Alberto (Jesus) Zancanaro
@organization : University of Luxembourg
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import torch

from library.model import hvEEGNet

from library.dataset import preprocess as pp
from library.config import config_dataset as cd
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

subj_id = 3

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get dataset

dataset_config = cd.get_moabb_dataset_config(subjects_list = [subj_id])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, _, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get number of channels and number of time samples
C = train_dataset[0][0].shape[1]
T = train_dataset[0][0].shape[2]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)

# Load the weights
model.load_state_dict(torch.load('./examples/example_trained_weigths.pth', map_location = torch.device('cpu')))
