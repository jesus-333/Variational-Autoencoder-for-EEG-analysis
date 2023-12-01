In this README are described the model structure and how to declare/use them.
Each model is implemented with PyTorch and when declared, the class constructor require a dictionary of parameters (specific for each class).

Example of generic declaraion:
```python
config = dict(
    ...
)

model = model_name.model_name(config)
```

Example of complete config dictionary are presented in the file `config_model.py`.
Note that when you create a specific model you need to insert in the dictionary all the parameters required inside the constructor. 

Index of the README.
* [EEGNet](#eegnet)
* [MBEEGNet](#mbeegnet)
* [MBEEGNet](#mbeegnet)
* [vEEGNet](#veegnet)
* [hierarchical-vEEGNet (hvEEGNet)](#hierarchical-veegnet)

# EEGNet
This repository contains our own implementation of EEGNet ([ArXiv][EEGNet_Arxiv], [Journal][EEGNet_Journal]).

Example of config file:
```python
config = dict(
    # Convolution: kernel size
    c_kernel_1 = (1, 64),
    c_kernel_2 = (C, 1),
    c_kernel_3 = (1, 16),
    # Convolution: number of filter
    filter_1 = 8,   # F1 in the original paper
    filter_2 = 16,  # F2 in the original paper
    # Pooling kernel
    p_kernel_1 = (1, 4),
    p_kernel_2 = (1, 8),
    # Other parameters
    C = 22,     # Number of EEG Channels
    D = 2,      # Depth multipliers
    activation = 'elu',
    use_bias = False,
    dropout = 0.5,
    flatten_output = True,
)
```

The model is defined in the file `EEGNet.py`, inside the class `EEGNet`. Note that this is a slight modification from the original network because the final output is the feature vector and not directly the label of the input signal. For the classifier see the class `EEGNET_classifier`.


# MBEEGNet
This repository contains our own implementation of MBEEGNet ([mdpi][MBEEGNet_mdpi], [pubmed][MBEEGNet_pubmed]).
MBEEGNet is composed by 3 EEGNet with different temporal kernel that work in parallel. The difference between the various EEGNet is in the kernel of the first layer (i.e. the temporal filter). The parameters for the rest of the network are the same for all 3 networks.

Example of config file:
```python
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
    eegnet_config = dict( ... ) # Used for the parameters after the first layer. See EEGNet section for the field names 
)
```

N.B. Note that the field `eegnet_config` is a dictionary that contains the exact configuration of EEGNet. 

The model is defined in the file `MBEEGNet.py`, inside the class `MBEEGNet`. Note that this is a slight modification from the original network because the final output is the feature vector and not directly the label of the input signal. For the classifier see the class `MBEEGNET_classifier`. 


# vEEGNet
This repository contains the implementation of vEEGNet, i.e. a variational autoencoder where the encoder and the decoder are EEGNet.
The function to get the config is available at the following [link](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/blob/aca703e9ec014338396d4239e98285918de74ac7/library/config/config_model.py#L83C14-L83C14)
Example of config file:
```python
    config = dict(
        hidden_space = hidden_space, # (int) Size of the hidden space. Not used if parameters_map_type == 0 
        type_encoder = type_encoder, # (int) Type of the decoder : 0 for EEGNet, 1 for MBEEGNet (not implemented)
        encoder_config = encoder_config, # Dictionary with the config of the encoder. Used also for the decoder
        type_decoder = type_decoder, # N.b. specify if use upsample or transposed convolution in the encoder
        type_vae = 0, # 0 = normal VAE, 1 = conditional VAE (not implemented)
        n_classes = 4, # N. of labels for the classifier. Used only if use_classifier == True
        use_classifier = False,
        parameters_map_type = 0, # 0 (convolution), 1 (feed forward layer). This parameter specify if use a 1x1 convolution or a feedforward to create the mean and variance variables of the latent hidden_space
        use_activation_in_sampling = False, # Add activation function in the sampling layer
        sampling_activation = 'elu',
    )
```

# hierarchical-vEEGNet
This repository contains the implementation of hierarchical-vEEGNet (hvEEGNet) ([preprint][hvEEGNet_preprint]). The model is a vEEGNet with a hierarchical structure (inspired by [NVAE][NVAE])

Example of config file:
```python
    config = dict(
        hidden_space = 1, # Note that this parameter is not important since it is necessary for the creation of a complete STANDARD vEEGNet but after the creation we will use the single modules and not the entire network. More info below
        type_encoder = 0, # (int) Type of the decoder : 0 for EEGNet, 1 for MBEEGNet (not implemented)
        encoder_config = encoder_config, # Dictionary with the config of the encoder. Used also for the decoder
        type_decoder = type_decoder, # N.b. specified the architecture of decoder 
        type_vae = 0, # 0 = normal VAE, 1 = conditional VAE (not implemented)
        n_classes = 4,
        use_h_in_decoder = False, # See Fig. 2 NVAE Paper. Basically h is an extra input in the deepest hidden space
        use_activation_in_sampling = True,
        sampling_activation = 'elu',
        convert_logvar_to_var = False, # If true the model return the variance of the distribution instead of of the log var
        hidden_space_dimension_list = [32, 128, 512], # Important only if parameters_map_type = 1
        parameters_map_type = parameters_map_type,# 0 (convolution), 1 (feed forward layer). This parameter specify if use a 1x1 convolution or a feedforward to create the mean and variance variables of the latent hidden_space
        use_classifier = False, # Add a classifier that receive in input the samples from the deepest latent space
    )
```

## Note about hvEEGNet creation
The hierarchical VAE is implemented as self-contained class in the file [hierarchical_VAE.py](../model/hierarchical_VAE.py). This class must receive in input two lists of modules : one for the encoder and one for the decoder. A module is considered as a level of hierarchy, i.e. the output of each module correspond to a latent space and it is the input of the next module. To create hvEEGNet we use the file [hvEEGNet.py](../model/hvEEGNet.py) that basically create a standard vEEGNet, divide it in the 3 modules (temporal convolution, spatial convolution and separable convolution) and use them as input for the the hierarchical_VAE class.

Technically the hierarchical_VAE class allow you to implement a high variety of models, not just hvEEGNet.




<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->

<!-- Reference Link -->
[EEGNet_Journal]: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
[EEGNet_Arxiv]: https://arxiv.org/abs/1611.08024
[MBEEGNet_mdpi]: https://www.mdpi.com/2079-6374/12/1/22
[MBEEGNet_pubmed]: https://pubmed.ncbi.nlm.nih.gov/35049650/
[NVAE]:https://arxiv.org/abs/2007.03898

[hvEEGNet_preprint]: https://www.researchgate.net/publication/375868326_hvEEGNet_exploiting_hierarchical_VAEs_on_EEG_data_for_neuroscience_applications
[vEEGNet_v2_preprint]:https://www.researchgate.net/publication/375867809_vEEGNet_learning_latent_representations_to_reconstruct_EEG_raw_data_via_variational_autoencoders/related
