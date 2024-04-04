"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config for the various models
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%%

def get_config_EEGNet(C : int, T : int) -> dict:
    config = dict(
        # Convolution: kernel size
        c_kernel_1 = (1, 128),  # Kernel for the temporal convolution
        c_kernel_2 = (C, 1),    # Kernel for the spatial convolution
        c_kernel_3 = (1, 32),   # Kernel of the horizontal convolution inside the separable convolution

        # Convolution: number of filter
        filter_1 = 8,   # Number of depth maps created by the temporal convolution
        filter_2 = 16,  # Number of depth maps created by the separable convolution

        # Pooling kernel. The pooling kernel must have the following form (1, x) with x as the pooling factor
        # e.g. p_kernel_1 = (1, 4) means that the first pooling kernel will reduce by a factor of 4 the length of the input (with lenght refers to the number of temporal samples)
        # For more information about pooling check the EEGNet paper
        p_kernel_1 = (1, 4),
        p_kernel_2 = (1, 8),

        # Other parameters
        C = C,  # Number of EEG Channels
        T = T,  # Number of time samples
        D = 2,  # Depth multipliers (Check EEGNet paper)
        activation = 'elu',     # Activation function to use. Supported activation functions are relu, elu, selu, gelu. The original EEGNet used elu
        use_bias = False,       # If True add bias during convolution computation.
        prob_dropout = 0.5,     # Probability of dropout
        use_dropout_2d = True,  # If True use drouput2d instead of the classic drouput between the convolution layer. For more info about drouput and drouput2d check PyTorch documenatation.
        flatten_output = True,  # If True flatten the tensor after the separable convolution. 
        print_var = True,       # If set to True print some information during the creation of the network
    )

    return config

def get_config_EEGNet_classifier(C : int, T : int, n_classes : int):
    config = get_config_EEGNet(C, T)
    config['n_classes'] = n_classes
    config['flatten_output'] = True

    return config

def get_config_EEGNet_stft_classifier(C : int, T : int, n_channels : int):
    config = get_config_EEGNet(C, T)
    config['depth_first_layer'] = n_channels
    config['c_kernel_1'] = config['c_kernel_2'] = config['c_kernel_3'] = (7, 7)
    config['n_classes'] = 4 
    config['flatten_output'] = True

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_MBEEGNet(C: int, T: int) -> dict:
    config = dict(
        # EEGNet 1
        temporal_kernel_1 = (1, 64),
        dropout_1 = 0.5,
        # EEGNet 2
        temporal_kernel_2 = (1, 16),
        dropout_2 = 0.5,
        # EEGNet 3
        temporal_kernel_3 = (1, 4),
        dropout_3 = 0.5,
        # Other
        C = C, # Number of EEG Channels
        T = T, # Number of EEG Temporal samples
        eegnet_config = get_config_EEGNet(C, T)
    )

    return config

def get_config_MBEEGNet_classifier(C: int, T: int, n_classes: int) -> dict:
    config = get_config_MBEEGNet(C, T)
    config['n_classes'] = n_classes
    config['eegnet_config']['flatten_output'] = True

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_vEEGNet(C : int, T : int, hidden_space : int, type_encoder : int, type_decoder : int) -> dict:
    # Get the config for the encoder (used also for the decoder)
    if type_encoder == 0: # EEGNet
        encoder_config = get_config_EEGNet(C, T)
        encoder_config['flatten_output'] = False # Set to False to use the custom sample_layer 
    elif type_encoder == 1: #MBEEGnet
        encoder_config = get_config_MBEEGNet(C, T)
        encoder_config['eegnet_config']['flatten_output'] = False
    else:
        raise ValueError("type_encoder must be 0 (EEGNET) or 1 (MBEEGNet)")
    
    # Config specific for vEEGNet
    config = dict(
        hidden_space = hidden_space, # Not used if parameters_map_type == 0
        type_encoder = type_encoder,
        encoder_config = encoder_config,
        type_decoder = type_decoder, # N.b. specify if use upsample or transposed convolution in the encoder
        type_vae = 0, # 0 = normal VAE, 1 = conditional VAE
        n_classes = 4,
        use_classifier = False,
        parameters_map_type = 0, # 0 (convolution), 1 (feed forward layer). This parameter specify if use a convolution to create the mean and variance variables of the latent hidden_space
        use_activation_in_sampling = True,
        sampling_activation = 'elu',
    )

    return config

def get_config_hierarchical_vEEGNet(C : int, T : int, type_decoder : int = 0, parameters_map_type : int = 0) -> dict:
    """
    Config for hierarchical vEEGNet (hvEEGNet)
    
    @param C: int with the number of EEG channels.
    @param T: int with number of time samples
    @param type_decoder: int. Decide if use upsample (0) or transpose convolution (1) to increase the size of the data inside the decoder. Keep the value to 0.
    @param parameters_map_type: int. Defined how to map the data inside the latent space. Keep the value to 0
    @return: config: dictionary with all the config necessary to create hvEEGNet
    """

    # Get the config for the encoder (used also for the decoder)
    encoder_config = get_config_EEGNet(C, T)    # Since the encoder is basically EEGNet, its config are the config of EEGNet model. For more information check the EEGNet config
    encoder_config['flatten_output'] = True     # Legacy stuff. Not used during hvEEGNet creation. Ignore.
    encoder_config['p_kernel_1'] = None         # Specify the kernel for the first pooling (None -> no pooling)
    encoder_config['p_kernel_2'] = (1, 10)      # Specify the kernel for the second pooling
    
    # Config specific for vEEGNet
    config = dict(
        hidden_space = 1, # Note that this parameter is not important since it is necessary for the creation of a complete STANDARD vEEGNet but after the creation we will use the single modules and not the entire network
        type_encoder = 0,
        encoder_config = encoder_config, # Used also for the decoder
        type_decoder = type_decoder, # N.b. specified the architecture of decoder
        type_vae = 0, # 0 = normal VAE, 1 = conditional VAE (not implemented)
        n_classes = 4,
        use_h_in_decoder = False,
        use_activation_in_sampling = True,
        sampling_activation = 'elu',
        convert_logvar_to_var = False,
        hidden_space_dimension_list = [32, 128, 512], # Important only if parameters_map_type = 1
        parameters_map_type = parameters_map_type,
        use_classifier = False,
    )

    config['encoder_config']['print_var'] = False

    return config

