"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Minor support function used in the various script
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Support during creation

def get_activation(activation_name: dict):
    """
    Receive a string and return the relative activation function in pytorch.
    Implemented for relu, elu, selu, gelu
    """

    if activation_name.lower() == 'relu':
        return nn.ReLU()
    elif activation_name.lower() == 'elu':
        return nn.ELU()
    elif activation_name.lower() == 'selu':
        return nn.SELU()
    elif activation_name.lower() == 'gelu':
        return nn.GELU()
    else:
        error_message = 'The activation must have one of the following string: relu, elu, selu, gelu'
        raise ValueError(error_message)

def get_dropout(prob_dropout: float, use_droput_2d : bool):
    if use_droput_2d:
        return nn.Dropout2d(prob_dropout)
    else: 
        return nn.Dropout(prob_dropout)


def count_trainable_parameters(layer):
    n_parameters = sum(p.numel() for p in  layer.parameters() if p.requires_grad)
    return n_parameters

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Pytorch Modules

class weighted_sum_tensor(nn.Module):
    def __init__(self, depth_x1 : int, depth_x2 : int, depth_output):
        """
        Implement a 1 x 1 convolution that mix togheter two tensors. The convolution works along the depth dimension.
        Sinche the size of the kernel is (1 x 1) this operation is equal to a weighted sum between the features maps, with the weighted learned during the training
        """
        super().__init__()

        self.mix = nn.Conv2d(depth_x1 + depth_x2, depth_output, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.mix(x)
        return x

class map_to_distribution_parameters_with_convolution(nn.Module):
    def __init__(self, depth : int, use_activation : bool = False, activation : str = 'elu'):
        """
        Take in input a tensor of size B x D x H x W and return a tensor of size B x 2D x H x W
        It practically doubles the size of the depth. The first D map will be used as mean and the other will be used a logvar 
        """
        super().__init__()

        self.map = nn.Sequential(
            get_activation(activation) if use_activation else nn.Identity(),
            nn.Conv2d(depth, int(depth * 2), 1)
        )

    def forward(self, x):
        x = self.map(x)
        mean, log_var = x.chunk(2, 1)
        return mean, log_var


class map_to_distribution_parameters_with_vector(nn.Module):
    def __init__(self, hidden_space_dimension : int, input_shape, use_activation : bool = False, activation : str = 'elu'):
        """
        Take in input a tensor of size B x D x H x W and return two vector of size hidden_space_dimension
        The first vector will be used as mean and the second as log var
        """
        super().__init__()

        n_neurons_input = input_shape[1] * input_shape[2] * input_shape[3]

        self.map_mu = nn.Sequential(
            get_activation(activation) if use_activation else nn.Identity(),
            nn.Linear(n_neurons_input, hidden_space_dimension)
        )

        self.map_sigma = nn.Sequential(
            get_activation(activation) if use_activation else nn.Identity(),
            nn.Linear(n_neurons_input, hidden_space_dimension)
        )

    def forward(self, x):
        return self.map_mu(x), self.map_sigma(x)


class sample_layer(nn.Module):
    def __init__(self, input_shape, config : dict, hidden_space_dimension : int = -1):
        """
        PyTorch module used to implement the reparametrization trick and the sampling from the laten space in the hierarchical VAE (hVAE)
        input_shape = shape of the input tensor
        config = dictionary with parameters for the sampling layer. Check the README for more information.
        hidden_space_dimension = int with the dimension of the laten/hidden space. Used only if we have a vector latent space
        """
        super().__init__()

        self.parameters_map_type = config['parameters_map_type']
        self.input_shape = list(input_shape)

        if self.parameters_map_type == 0: # Convolution (i.e. matrix latent space)
            depth = config['input_depth'] if 'input_depth' in config else input_shape[1]
            self.parameters_map = map_to_distribution_parameters_with_convolution(depth = depth,
                                                                               use_activation = config['use_activation_in_sampling'], 
                                                                               activation = config['sampling_activation'])
        elif self.parameters_map_type == 1: # Feedforward (i.e. vector latent space)
            self.parameters_map = map_to_distribution_parameters_with_vector(hidden_space_dimension = hidden_space_dimension,
                                                                          input_shape = input_shape,
                                                                          use_activation = config['use_activation_in_sampling'], 
                                                                          activation = config['sampling_activation'])
            n_neurons_input = hidden_space_dimension
            n_neurons_output = input_shape[1] * input_shape[2] * input_shape[3]
            
            self.ff_layer = nn.Sequential(
                nn.Linear(n_neurons_input, n_neurons_output),
                get_activation(config['sampling_activation']) if config['use_activation_in_sampling'] else nn.Identity(),
            )

            self.input_shape[0] = -1
        else:
            raise ValueError("config['parameters_map_type'] must have value 0 (convolution-matrix) or 1 (Feedforward-vector)")

    def forward(self, x):
        if self.parameters_map_type == 1: # Feedforward
            x = x.flatten(1)
        
        # Get distribution parameters and "sample" through reparametrization trick
        mean, log_var = self.parameters_map(x)
        z = self.reparametrize(mean, log_var)

        if self.parameters_map_type == 0: # Convolution
            x = z
        elif self.parameters_map_type == 1: # Feedforward (i.e. vector latent space)
            x = self.ff_layer(z)
            x = x.reshape(self.input_shape)
        else:
            raise ValueError("config['parameters_map_type'] must have value 0 (convolution-matrix) or 1 (Feedforward-vector)")

        return x, mean, log_var

    def reparametrize(self, mu, log_var, return_as_tensor = False):
        """
        Execute the reparametrization trick to allow gradient backpropagation
        mu = mean of the normal distribution 
        log_var = logarithm of the variance of the normal distribution
        return_as_matrix = pass z through the ff layer and return it with the dimension of the orignal tensor. Works only if self.parameters_map_type == 1
        """

        z = reparametrize(mu, log_var)

        if return_as_tensor and self.parameters_map_type == 1: 
            return self.ff_layer(z).reshape(self.input_shape)
        else: return z


def reparametrize(mu, log_var):
    """
    Execute the reparametrization trick to allow gradient backpropagation
    mu = mean of the normal distribution 
    log_var = logarithm of the variance of the normal distribution
    """
    sigma = torch.exp(0.5 * log_var)

    eps = torch.randn_like(sigma)
    eps = eps.type_as(mu) # Setting z to be cuda when using GPU training 

    return  mu + sigma * eps


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function for aEEGNet

def create_head(config : dict) :
    """
    Create the head (Feedforward layer) for the attention module
    """

    if config['use_head_for_q'] :
        q_head = nn.Linear(config['input_length'], config['qk_head_output_length'])
    else :
        q_head = nn.Identity()

    if config['use_head_for_k'] :
        k_head = nn.Linear(config['input_length'], config['qk_head_output_length'])
    else :
        k_head = nn.Identity()

    if config['use_head_for_v'] :
        v_head = nn.Linear(config['input_length'], config['v_head_output_length'])
    else :
        v_head = nn.Identity()

    return q_head, k_head, v_head

def compute_attention_values(q : torch.Tensor, k : torch.Tensor, v : torch.Tensor, normalize_qk : bool) -> torch.Tensor :
    # Transpose k arrays
    k = torch.permute(k, (0, 2, 1))
    
    # Compute the score to use in the linear combination of values
    score = torch.bmm(q, k)
    if normalize_qk : score /= torch.sqrt(torch.tensor(q.shape[2]))
    score = nn.functional.softmax(score, dim = 2)

    z = torch.bmm(score, v)

    return z
