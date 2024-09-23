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
        Sinche the size of the kernel is (1 x 1) this operation is equal to a weighted sum between the features maps, with the weights learned during the training
        """
        super().__init__()

        self.mix = nn.Conv2d(depth_x1 + depth_x2, depth_output, 1)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim = 1)
        x = self.mix(x)
        return x

class LoRALinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha : float = -1, use_spectral_norm : bool = False):
        super().__init__()

        # These are the new LoRA params. In general rank << in_dim, out_dim
        if use_spectral_norm : 
            self.lora_a = nn.utils.parametrizations.spectral_norm(nn.Linear(in_dim, rank, bias = False))
            self.lora_b = nn.utils.parametrizations.spectral_norm(nn.Linear(rank, out_dim, bias = False))
        else : 
            self.lora_a = nn.Linear(in_dim, rank, bias = False)
            self.lora_b = nn.Linear(rank, out_dim, bias = False)

        # Rank and alpha are commonly-tuned hyperparameters
        self.rank = rank
        self.alpha = alpha if alpha > 0 else rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lora_a projects inputs down to the much smaller self.rank,
        # then lora_b projects back up to the output dimension
        lora_out = self.lora_b(self.lora_a(x))

        # Finally, scale by the alpha parameter (normalized by rank)
        # and add to the original model's outputs
        return (self.alpha / self.rank) * lora_out

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

class map_to_distribution_parameters_with_vector_LoRA(nn.Module):
    def __init__(self, hidden_space_dimension : int, input_shape, rank : int, use_activation : bool = False, activation : str = 'elu', use_spectral_norm : bool = False):
        """
        Work as map_to_distribution_parameters_with_vector but use the basic idea behind LoRA to reduce the number of wieghts in the linear layer
        """
        super().__init__()

        n_neurons_input = input_shape[1] * input_shape[2] * input_shape[3]

        self.map_mu = nn.Sequential(
            get_activation(activation) if use_activation else nn.Identity(),
            LoRALinear(n_neurons_input, hidden_space_dimension, rank, use_spectral_norm = use_spectral_norm)
        )

        self.map_sigma = nn.Sequential(
            get_activation(activation) if use_activation else nn.Identity(),
            LoRALinear(n_neurons_input, hidden_space_dimension, rank, use_spectral_norm = use_spectral_norm)
        )

    def forward(self, x):
        return self.map_mu(x), self.map_sigma(x)


class sample_layer(nn.Module):
    def __init__(self, input_shape, config : dict, hidden_space_dimension : int = -1):
        """
        PyTorch module used to implement the reparametrization trick and the sampling from the laten space in the hierarchical VAE (hVAE)
        input_shape = shape of the input tensor
        config = dictionary with some parameters
        hidden_space_dimension = int with the dimension of the laten/hidden space. Used only if we have a vector as hidden variable
        """
        super().__init__()

        self.parameters_map_type = config['parameters_map_type']
        self.input_shape = list(input_shape)

        if self.parameters_map_type == 0: # Convolution (i.e. matrix latent space)
            self.parameters_map = map_to_distribution_parameters_with_convolution(depth = input_shape[1],
                                                                               use_activation = config['use_activation_in_sampling'], 
                                                                               activation = config['sampling_activation'])
        elif self.parameters_map_type == 1 or self.parameters_map_type == 2: # Feedforward (i.e. vector latent space)(1 without LoRA, 2 with LoRA)
            # Used to map from the latent space sample to the tensor that will be used as input of the decoder
            # print(input_shape)
            n_neurons_output = input_shape[1] * input_shape[2] * input_shape[3]

            if self.parameters_map_type == 1 : # Feedforward layer
                # Map from encoder output to latent space
                self.parameters_map = map_to_distribution_parameters_with_vector(hidden_space_dimension = hidden_space_dimension,
                                                                              input_shape = input_shape,
                                                                              use_activation = config['use_activation_in_sampling'], 
                                                                              activation = config['sampling_activation'])

                # Map from latent space sample to decoder input
                map_from_hidden_space_sample = nn.Linear(hidden_space_dimension, n_neurons_output)

            elif self.parameters_map_type == 2 : # Feedforward layer with LoRA
                # Map from encoder output to latent space
                self.parameters_map = map_to_distribution_parameters_with_vector_LoRA(hidden_space_dimension = hidden_space_dimension,
                                                                                      input_shape = input_shape,
                                                                                      rank = config['rank'],
                                                                                      use_activation = config['use_activation_in_sampling'], 
                                                                                      activation = config['sampling_activation'],
                                                                                      use_spectral_norm = config['use_spectral_norm']
                                                                                      )
            
                # Map from latent space sample to decoder input
                map_from_hidden_space_sample = LoRALinear(hidden_space_dimension, n_neurons_output, config['rank'], use_spectral_norm = config['use_spectral_norm'])

            self.ff_layer = nn.Sequential(
                map_from_hidden_space_sample,
                get_activation(config['sampling_activation']) if config['use_activation_in_sampling'] else nn.Identity(),
            )

            self.input_shape[0] = -1
        else:
            raise ValueError("config['parameters_map_type'] must have value 0 (convolution-matrix) or 1 (Feedforward-vector) or 2 (Feedforward-vector with Lora). Curent value {}".format(self.parameters_map_type))

    def forward(self, x):
        if self.parameters_map_type == 1 or self.parameters_map_type == 2: # Feedforward/Feedforward with LoRA
            x = x.flatten(1)
        
        # Get distribution parameters and "sample" through reparametrization trick
        mean, log_var = self.parameters_map(x)
        z = self.reparametrize(mean, log_var)

        if self.parameters_map_type == 0: # Convolution
            x = z
        elif self.parameters_map_type == 1 or self.parameters_map_type == 2: # Feedforward (i.e. vector latent space)
            x = self.ff_layer(z)
            x = x.reshape(self.input_shape)
        else:
            raise ValueError("config['parameters_map_type'] must have value 0 (convolution-matrix) or 1 (Feedforward-vector) or 2 (Feedforward-vector with LoRA)")

        return x, mean, log_var

    def reparametrize(self, mu, log_var, return_as_tensor = False):
        """
        Execute the reparametrization trick to allow gradient backpropagation
        mu = mean of the normal distribution 
        log_var = logarithm of the variance of the normal distribution
        return_as_matrix = pass z through the ff layer and return it with the dimension of the orignal tensor. Works only if self.parameters_map_type == 1
        """

        z = reparametrize(mu, log_var)

        if return_as_tensor and (self.parameters_map_type == 1 or self.parameters_map_type == 2): 
            return self.ff_layer(z).reshape(self.input_shape)
        else : 
            return z


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
