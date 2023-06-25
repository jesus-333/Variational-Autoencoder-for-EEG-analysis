"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train vEEGNet
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import torch
import wandb
import os
import sys

# Custom functions
import wandb_support
import metrics
import dataset
import vEEGNet 

# Config files
import config_model as cm
import config_dataset as cd
import config_training as ct
import loss_function as lf

import train_generic
    
"""
%load_ext autoreload
%autoreload 2

import train_vEEGNet 
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
