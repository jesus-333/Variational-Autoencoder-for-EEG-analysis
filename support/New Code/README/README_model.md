In this README are described the model structure and how to declare/use them.
Each model is implemented with PyTorch and when declare the class constructor require a dictionary of parameter (specific for each class).

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

The model is defined in the file `EEGNet.py`. Note that this is a slight modification from the original network because the final output is the feature vector and not directly the label of the input signal.

Example of config file:
```python
config = dict(
    # Convolution: kernel size
    c_kernel_1 = (1, 64),
    c_kernel_2 = (C, 1),
    c_kernel_3 = (1, 16),
    # Convolution: number of filter
    filter_1 = 8, # F1 in the original paper
    filter_2 = 16, # F2 in the original paper
    # Pooling kernel
    p_kernel_1 = (1, 4),
    p_kernel_2 = (1, 8),
    # Other parameters
    C = 22, # Number of EEG Channels
    D = 2, # Depth multipliers
    activation = 'elu',
    use_bias = False,
    dropout = 0.5,
    flatten_output = True,
)
```

# MBEEGNet
This repository offer our own implementation of MBEEGNet ([mdpi][./README_list_of_file.md#[MBEEGNet_mdpi]], [pubmed][EEGNet_Journal]).


<!-- Reference Link -->
[EEGNet_Journal]: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
[EEGNet_Arxiv]: https://arxiv.org/abs/1611.08024
