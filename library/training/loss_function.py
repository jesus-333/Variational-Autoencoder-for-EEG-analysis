"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

List of loss function used during training.

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

from .soft_dtw_cuda import SoftDTW

import torch
import torch.nn.functional as F

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Losses declaration

# Classifier loss

def classifier_loss(predicted_label, true_label):
    return F.nll_loss(predicted_label, true_label)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction loss

def recon_loss_function(x, x_r):
    return F.mse_loss(x, x_r)

def recon_loss_frequency_function(x, x_r):
    pass

def compute_dtw_loss_along_channels(x : torch.tensor, x_r : torch.tensor, dtw_loss_function, soft_DTW_type : int = 1, average_channels : True = False):
    """
    Compute the dtw between two tensor x and x_r using the softDTW .
    The two tensors (x and x_r) must both have shape B x 1 x C x T. The dtw will be computed element by element and channel by channel.
    An element is an EEG signal of shape C x T and the number of element is B (i.e. the first dimension).
    For each EEG signal the dtw is computed independently channel by channel, sum together and, eventually, averaged along channels if average_channels is True.

    Example :
    If your input x has shape 4 x 1 x 22 x 1000 it means that the tensor contains 4 EEG signals, and each signal has 22 channels and 1000 temporal samples.
    The tensor x_r must have the same shape. The dtw is computed element by element in the sense that the first element of x_1 is compared with the first element of x_2 etc.
    Then given a pair of EEG signal of shape 22 x 1000, inside the signal the dtw is computed channel by channel (because the dtw compute the difference between 1d signal).
    So for a pair of EEG signal of shape 22 x 1000 the dtw ouput are 22 values, 1 for each channel. This 22 values are sum together to obtain the final dtw value for the pair of EEG signal.
    If average_channels is True instead of the sum the final dtw value will be the average of the 22 values

    @param x: (torch.tensor) First input tensor of shape B x 1 x C x T
    @param x_r: (torch.tensor) Second input tensor of shape B x 1 x C x T
    @param dtw_loss_function: (function) The dtw implementation to use. Actually during the training I use the one provided by https://github.com/Maghoumi/pytorch-softdtw-cuda. This parameter exist to allow the use of other implementation
    @param soft_DTW_type : (int). Default to 1. Specify the type of soft-DTW to use. 1 = classical soft-DTW. 2 = soft-dtw divergence. For more info https://arxiv.org/pdf/2010.08354
    @param average_channels : (bool) If True compute the average of the reconstruction error along the channels

    @return recon_error: (torch.tensor) Tensor of shape B
    """

    if x.shape != x_r.shape :
        raise ValueError("x_1 and x_2 must have the same shape. Current shape x_1 : {}, x_2 : {}".format(x.shape, x_r.shape))

    recon_loss = 0
    for i in range(x.shape[2]): # Iterate through EEG Channels
        x_ch = x[:, :, i, :].swapaxes(1, 2)
        x_r_ch = x_r[:, :, i, :].swapaxes(1, 2)
        # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
        # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
        
        if soft_DTW_type == 1 :
            tmp_recon_loss = dtw_loss_function(x_ch, x_r_ch)
        elif soft_DTW_type == 2 :
            tmp_recon_loss = dtw_loss_function(x_ch, x_r_ch) - 0.5 * (dtw_loss_function(x_ch, x_ch) + dtw_loss_function(x_r_ch, x_r_ch))
        else :
            raise ValueError("soft_DTW_type must have value 1 (classical soft-DTW) or 2 (soft-DTW divergence). Current value is {}".format(soft_DTW_type))
        
        recon_loss += tmp_recon_loss.mean()

    if average_channels: recon_loss /= x.shape[2]

    return recon_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Kullback loss

def kl_loss_normal_function(mu, log_var, delta_mu = None, delta_log_var = None):
    """
    Compute the Kullback–Leibler divergence between the distribution parametrized by mu and log var and a normal distribution
    """

    # If mu is a matrix it will have shape [B x D x H x W] so len(shape) = 4 > 2
    # If mu is a vector it will have shape [B x N] so len(shape)
    if len(mu.shape) > 2: dim_sum = [1, 2, 3]
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

def hierarchical_KL(mu_list, log_var_list, delta_mu_list, delta_log_var_list):
    # Kullback
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
        
    return kl_loss, kl_loss_list

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


def hvEEGNet_loss(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, config : dict, predicted_label = None, true_label = None):
    # Reconstruction loss
    recon_loss = recon_loss_function(x, x_r)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Kullback
    kl_loss, kl_loss_list = hierarchical_KL(mu_list, log_var_list, delta_mu_list, delta_log_var_list)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Classifier

    if config['use_classifier'] and (predicted_label is not None and true_label is not None):
        clf_loss = classifier_loss(predicted_label, true_label)

        final_loss = config['alpha'] * recon_loss + config['beta'] * kl_loss + config['gamma'] * clf_loss

        return final_loss, recon_loss, kl_loss, kl_loss_list, clf_loss
    else:

        final_loss = config['alpha'] * recon_loss + config['beta'] * kl_loss

        return final_loss, recon_loss, kl_loss, kl_loss_list

class vEEGNet_loss():
    def __init__(self, config : dict):
        """
        Class that compute the loss function for the Variational autoencoder
        """

        # Reconstruction loss
        if config['recon_loss_type'] == 0: # L2 loss
            self.recon_loss_function = recon_loss_function
        elif config['recon_loss_type'] == 1 or config['recon_loss_type'] == 2: # Soft-DTW or soft-DTW divergence
            gamma_dtw = config['gamma_dtw'] if 'gamma_dtw' in config else 1
            use_cuda = True if config['device'] == 'cuda' else False
            self.recon_loss_function = SoftDTW(use_cuda = use_cuda, gamma = gamma_dtw)
        else :
            raise ValueError("soft_DTW_type must have value 1 (classical soft-DTW) or 2 (soft-DTW divergence). Current value is {}".format(config['recon_loss_type']))
        self.recon_loss_type = config['recon_loss_type']
        
        self.edge_samples_ignored = config['edge_samples_ignored'] if 'edge_samples_ignored' in config else 0
        if self.edge_samples_ignored < 0: self.edge_samples_ignored = 0
            
        # Kullback
        self.kl_loss_function = kl_loss_normal_function
        
        # Classifier loss
        self.clf_loss_function = classifier_loss
        
        # Hyperparameter for the various part of the loss
        self.alpha = config['alpha'] if 'alpha' in config else 1 # Recon
        self.beta = config['beta'] if 'beta' in config else 1    # KL
        self.gamma = config['gamma'] if 'gamma' in config else 1 # Clf
        
    def compute_loss(self, x, x_r, mu, log_var, predicted_label = None, true_label = None):
        recon_loss = self.compute_recon_loss(x, x_r)
        kl_loss  = self.kl_loss_function(mu, log_var)
        
        if predicted_label is not None and true_label is not None: # Compute classifier loss
            clf_loss = classifier_loss(predicted_label, true_label)
            final_loss = self.alpha * recon_loss + self.beta * kl_loss + self.gamma * clf_loss
            return final_loss, recon_loss, kl_loss, clf_loss
        else: # Not compute classifier loss
            final_loss = self.alpha * recon_loss + self.beta * kl_loss
            return final_loss, recon_loss, kl_loss
        
    def compute_recon_loss(self, x, x_r):
        # (OPTIONAL) Remove sample from border (could contain artifacts due to padding)
        x = x[:, :, :, self.edge_samples_ignored: - 1 - self.edge_samples_ignored]
        x_r = x_r[:, :, :, self.edge_samples_ignored: - 1 - self.edge_samples_ignored]
        
        if self.recon_loss_type == 0: # Mean Squere Error (L2)
            recon_loss = self.recon_loss_function(x, x_r)
        elif self.recon_loss_type == 1 or self.recon_loss_type == 2: # SDTW or SDTW-Divergene
            recon_loss = compute_dtw_loss_along_channels(x, x_r, self.recon_loss_function, self.recon_loss_type)
        else:
            raise ValueError("Type of loss function for reconstruction not recognized (recon_loss_type has wrong value)")

        return recon_loss
        
class hvEEGNet_loss(vEEGNet_loss):
    
    def __init__(self, config : dict):
        """
        Class that compute the loss function for the Hierarchical Variational autoencoder
        """
        super().__init__(config)
            
        # Kullback
        self.kl_loss_function = hierarchical_KL
        
    def compute_loss(self, x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predicted_label = None, true_label = None):
        recon_loss = self.compute_recon_loss(x, x_r)
        kl_loss, kl_loss_list = self.kl_loss_function(mu_list, log_var_list, delta_mu_list, delta_log_var_list)
        
        if predicted_label is not None and true_label is not None:
            clf_loss = classifier_loss(predicted_label, true_label)
            final_loss = self.alpha * recon_loss + self.beta * kl_loss + self.gamma * clf_loss
            return final_loss, recon_loss, kl_loss, kl_loss_list, clf_loss
        else:
            final_loss = self.alpha * recon_loss + self.beta * kl_loss
            return final_loss, recon_loss, kl_loss, kl_loss_list
