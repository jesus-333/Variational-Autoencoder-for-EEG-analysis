"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of EEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Network declaration

class EEGNet(nn.Module):
    
    def __init__(self, config: dict):
        """
        Implementation of EEGNet in PyTorch.
        
        Note that compared to the original EEGNet the last layer (feedforward + classification) it is not present.
        The network offer the possibility to flatten the output (and obtain a vector of features of shape batch x n. features) or to mantain it as a tensor 

        The name of the variable respect the nomeclature of the original paper where it is possible.
        """

        super().__init__()
        
        use_bias = config['use_bias']
        D = config['D']
        activation = get_activation(config['activation'])

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section
        
        # Block 1 - Temporal filters
        self.temporal_filter = nn.Sequential(
            nn.Conv2d(1, config['filter_1'], kernel_size = config['c_kernel_1'], padding = 'same', bias = use_bias),
            nn.BatchNorm2d(config['filter_1']),
        )
        
        # Block 1 - Spatial filters
        self.spatial_filter = nn.Sequential(
            nn.Conv2d(config['filter_1'], config['filter_1'] * D, kernel_size = config['c_kernel_2'], groups = config['filter_1'], bias = use_bias),
            nn.BatchNorm2d(config['filter_1'] * D),
            activation,
            nn.AvgPool2d(config['p_kernel_1']),
            nn.Dropout(config['dropout'])
        )

        # Block 2
        self.separable_convolution = nn.Sequential(
            nn.Conv2d(config['filter_1'] * D, config['filter_1'] * D, kernel_size = config['c_kernel_3'], groups = config['filter_1'] * D, padding = 'same', bias = use_bias),
            nn.Conv2d(config['filter_1'] * D, config['filter_2'], kernel_size = (1, 1), groups = 1, bias = use_bias),
            nn.BatchNorm2d(config['filter_2']),
            activation,
            nn.AvgPool2d(config['p_kernel_2']),
            nn.Dropout(config['dropout'])
        )
        
        
        if 'flatten_output' not in config: config['flatten_output'] = False
        self.flatten_output = config['flatten_output']

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Print additional information after the creation of the network
        if 'print_var' not in config: config['print_var'] = False
        if config['print_var']:
            print("EEGNet Created. Number of parameters:")
            print("\tNumber of trainable parameters (Block 1 - Temporal) = {}".format(count_trainable_parameters(self.temporal_filter)))
            print("\tNumber of trainable parameters (Block 1 - Spatial)  = {}".format(count_trainable_parameters(self.spatial_filter)))
            print("\tNumber of trainable parameters (Block 2)            = {}".format(count_trainable_parameters(self.separable_convolution)))


    def forward(self, x):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 1 (temporal + spatial filters)
        x = self.temporal_filter(x)
        x = self.spatial_filter(x)
        
        #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Block 2 (separable convolution)
        x = self.separable_convolution(x)
        
        # (OPTIONAL) Flat the output
        if self.flatten_output: return x.flatten(1)
        else: return x

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Other funcion

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

def count_trainable_parameters(layer):
    n_paramters = sum(p.numel() for p in  layer.parameters() if p.requires_grad)
    return n_paramters
