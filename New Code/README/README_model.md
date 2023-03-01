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

# EEGNet
This repository offer our own implementation of EEGNet ([ArXiv][EEGNet_Arxiv], [Journal][EEGNet_Journal]).


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
This repository offer our own implementation of MBEEGNet ([mdpi][MBEEGNet_mdpi], [pubmed][MBEEGNet_pubmed]).
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




<!-- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  -->

<!-- Reference Link -->
[EEGNet_Journal]: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
[EEGNet_Arxiv]: https://arxiv.org/abs/1611.08024
[MBEEGNet_mdpi]: https://www.mdpi.com/2079-6374/12/1/22
[MBEEGNet_pubmed]: https://pubmed.ncbi.nlm.nih.gov/35049650/
