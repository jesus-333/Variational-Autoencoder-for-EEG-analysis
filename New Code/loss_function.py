"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

List of loss function used during training.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# TODO Is it worth to implement?

def kl_loss_normal(mu, log_var):
    """
    Compute the Kullback–Leibler divergence between the distribution parametrized by mu and log var and a normal distribution
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

def kl_loss(mu_1, log_var_1, mu_2, log_var_2):
    pass

