"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Functions used to check the creation of the various models and the forward step
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import torch

# from . import config 
from .config import config_model as cm
from .model import vEEGNet, hvEEGNet

"""
%load_ext autoreload
%autoreload 2
import library.check_model as cm
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def check_vEEGNet():
    """
    Function to check the absence of breaking bug during the creation and the forward pass of vEEGNet
    """

    C = 22
    T = 512
    hidden_space = 16
    type_encoder = 0
    type_decoder = 0

    model_config = cm.get_config_vEEGNet(C, T, hidden_space, type_encoder, type_decoder)
    model = vEEGNet.vEEGNet(model_config)
    
    x = torch.rand(5, 1, C, T)
    x_r, z_mean, z_log_var, predict_labels = model(x)

    print("Input shape : ", x.shape)
    print("Output shape: ", x_r.shape)
    print(z_mean.shape)
    print(z_log_var.shape)
    print(predict_labels)


def check_hVAE_shallow():
    """
    Function to check the absence of breaking bug during the creation and the forward pass of the shallow hierarchical vEEGNet
    """

    C, T = 22, 512

    type_decoder = 0
    parameters_map_type = 1

    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)
    model = hvEEGNet.hvEEGNet_shallow(model_config)

    x = torch.rand(5, 1, C, T)
    output = model(x)

