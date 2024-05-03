"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config for the various models.
Note that the specific parameters present are intended as examples only.
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
        hidden_space = 1,   # Note that this parameter is not important since it is necessary for the creation of a complete STANDARD vEEGNet but after the creation we will use the single modules and not the entire network
        type_encoder = 0,   # Ignore. (0 -> EEGNet, 1 -> MBEEGNet (to check implementation)).
        encoder_config = encoder_config,    # Parameter of the encoder. For more info check get_config_EEGNet. (also note that this parameters will be also used for the decoder).
        type_decoder = type_decoder,        # N.b. specified the architecture of decoder. More info under this function definition.
        type_vae = 0,                       # 0 = normal VAE, 1 = conditional VAE (not implemented)
        n_classes = 4,                      # Used if there is a classifier. Ignore
        use_h_in_decoder = False,           # Ignore.
        use_activation_in_sampling = True,  # Ignore.
        sampling_activation = 'elu',        # Ignore.
        convert_logvar_to_var = False,      # Ignore.
        hidden_space_dimension_list = [32, 128, 512],   # Important only if parameters_map_type = 1. Ignore otherwise.
        parameters_map_type = parameters_map_type,      # More info under this function definition.
        use_classifier = False,             # Ignore. If True add a classifier to hvEEGNet.
    )

    config['encoder_config']['print_var'] = False

    return config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_config_ChWiAutoencoder() -> dict:

    config = dict(
        encoder_config = None,
        decoder_config = None,
        sample_layer_config = None
    )

    return config

def get_config_ChWi_encoder() -> dict:
    config = dict()

    return config

def get_config_ChWi_module_encoder() -> dict : 
    config = dict(
        in_channels = 1,
        out_channels = 1,
        c_kernel = 1,
        padding = 'same',
        group = None,
        use_batch_normalization = True,
        activation = 'elu',
        p_kernel = 1
    )

    return config

def get_config_classifier_model_v1(C : int, T : int, type_decoder : int, parameters_map_type : int, path_weights_hvEEGNet : str, 
                                   use_only_mu_for_classification = True, freeze_encoder = True) -> dict:
    """
    Used for classifier in classifier_model_v1 (encoder + classifier network)
    """
    
    hvEEGNet_config = get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)
    classifier_config = get_config_classifier_v1(use_only_mu_for_classification)

    model_config = dict(
        config_clf = classifier_config,
        config_hvEEGNet = hvEEGNet_config,
        path_weights = path_weights_hvEEGNet,
        freeze_encoder = freeze_encoder   # If True freeze the encoder weights
    )
    
    return model_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Config for aEEGNet

def get_config_aEEGNet(C : int, T : int, n_external_query : int = 0) -> dict :
    config = dict(
        eegnet_config = get_config_EEGNet(C, T),
        use_channels_self_attention = True, # Compute the attention along the EEG channels
        channels_self_attention_config = get_config_attention_module(),
        channels_n_external_query = n_external_query, # Number of external query array to use in the computation of attention in channel attention
        channels_external_attention_config_list = [], # Containts the config for each attention module used to compute the attention with the external query
        use_depthwise_attention = False, # Compute the attention along the depth map after the spatial filters. STILL TO IMPLEMENTS
    )

    for i in range(n_external_query) : config['external_attention_config_list'].append(get_config_attention_module())

    print("Note that the attention_module config (both for self-attention and external-attention) must be complted with the values of qk_head_output_length, v_head_output_length and external_query_input_length")
    print("The external_query_input_length is needed only for the external-attention module.")

    return config

def get_config_attention_module() -> dict :
    config = dict(
        use_head_for_q = True,
        use_head_for_k = True,
        use_head_for_v = True,
        qk_head_output_length = -1, # To set outside the function
        v_head_output_length = -1, # To set outside the function
        normalize_qk = True,
        external_query_input_length = -1, # Used only for external attention_module
    )

    return config
