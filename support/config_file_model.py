
"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function with the config for the model
"""


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%%

def get_config_vEEGNet_encoder(C = 22, T = 512):
    config = dict(
        # EEGNet 1
        temporal_kernel_1 = (1, 64),
        stride_1 = 4,
        # EEGNet 2
        temporal_kernel_2 = (1, 16),
        stride_2 = 4,
        # EEGNet 3
        temporal_kernel_3 = (1, 4),
        stride_3 = 1,
        # Other
        C = C,
        T = T,
        hidden_space_dimension = 64
    )

    return config
