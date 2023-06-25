"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

List of loss function used during training.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
import torch.nn.functional as F

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Classifier loss

def classifier_loss(true_label, predicted_label):
    return F.nll_loss(predicted_label, true_label)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Reconstruction loss

def recon_loss_function(x, x_r):
    return F.mse_loss(x, x_r)

def recon_loss_frequency_function(x, x_r):
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Kullback loss

def kl_loss_normal_function(mu, log_var):
    """
    Compute the Kullback–Leibler divergence between the distribution parametrized by mu and log var and a normal distribution
    """
    return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

def kl_loss_function(mu_1, log_var_1, mu_2, log_var_2):
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# VAE loss

def vae_loss_function(x, x_r, mu, log_var, config):
    if config['recon_loss_type'] == 0:
        recon_loss = recon_loss_function(x, x_r)
    elif config['recon_loss_type'] == 1:
        recon_loss = recon_loss_frequency_function(x, x_r)
    else:
        raise ValueError("Wrong recon_loss_type")

    kl_loss = kl_loss_normal_function(mu, log_var)

    return config['alpha'] * recon_loss + config['beta'] * kl_loss, recon_loss, kl_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def vEEGNet_loss(x, x_r, mu, log_var, true_label, predicted_label, config):
    # Compute the loss of the VAE
    vae_loss, recon_loss, kl_loss = vae_loss_function(x, x_r, mu, log_var, config)
    
    # Compute the loss for the classifier
    clf_loss = classifier_loss(true_label, predicted_label)
    
    # Compute the final loss (note that the multiplication with the hyperparameter for Reconstruction and kl loss is done inside the vae_loss_function)
    final_loss = vae_loss + config['gamma'] * clf_loss

    return final_loss, recon_loss, kl_loss, clf_loss



