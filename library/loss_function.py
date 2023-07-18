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

def kl_loss_normal_function(mu, log_var, delta_mu = None, delta_log_var = None):
    """
    Compute the Kullback–Leibler divergence between the distribution parametrized by mu and log var and a normal distribution
    """

    # If mu is a matrix it will have shape [B x D x H x W] so len(shape) = 4 > 2
    # If mu is a vector it will have shape [B x N] so len(shape)
    if len(mu.shape) > 2: dim_sum = [1,2,3]
    else: dim_sum = 1

    if delta_mu is None and delta_log_var is None: # Classic KL
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = dim_sum), dim = 0)
    else: # "Shifted" KL (Section 3.2 NVAE Paper)
        return torch.mean(-0.5 * torch.sum(1 + delta_log_var - delta_mu ** 2 / log_var.exp() - delta_log_var.exp(), dim = dim_sum), dim = 0)


def kl_loss_function(mu_1, log_var_1, mu_2, log_var_2):
    """
    General form of the KL divergence between two gaussian distribution
    """
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


def hvEEGNet_loss(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, config : dict):
    recon_loss = recon_loss_function(x, x_r)
    
    kl_loss = 0
    kl_loss_list = []
    for i in range(len(mu_list)):
        if i == 0:
            tmp_kl = kl_loss_normal_function(mu_list[i], log_var_list[i])
        else:
            tmp_kl = kl_loss_normal_function(mu_list[i], log_var_list[i], delta_mu_list[i - 1], delta_log_var_list[i - 1])
            # Remember that there are 1 element less the of the shift term (i.e. delta_mu and delta_log_var) respect the mus and log_vars
            # This because the deepest layer has not a shift term

        kl_loss += tmp_kl
        kl_loss_list.append(tmp_kl)

    final_loss = config['alpha'] * recon_loss + config['beta'] * kl_loss

    return final_loss, recon_loss, kl_loss, kl_loss_list

