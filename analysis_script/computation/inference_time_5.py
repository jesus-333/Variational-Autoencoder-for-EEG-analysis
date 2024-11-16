"""
Compute the inference time for 1 trial and different values of T (time samples) and C (number of channels)
The loss is computed with the CUDA implementations of the soft-DTW (inside the library) and the Rust implementations (if the pip package is installed)

Save the results in npy and txt files.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import time
import numpy as np
import os
try :
    import soft_dtw_rust
    import_soft_dtw_rust = True 
except ImportError:
    import_soft_dtw_rust = False
    print("Rust implementation of the soft-DTW is not available")
    print("Please install the package from https://pypi.org/project/soft-dtw-rust/")

from library.model import hvEEGNet
from library.training import loss_function
from library.config import config_model as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Values of C and T
C_list = [8, 22, 64]
C_list = [2, 4, 8, 16, 32]
T_list = (np.arange(40) + 1) * 50
T_list = (np.arange(20) + 1) * 50
T_list = [1000]

# Specify the loss type to use
# 0 : use the Rust implementation of the soft-DTW
# 1 : use the CUDA implementation of the soft-DTW
# 2 : use the Soft-DTW divergence (implemented in the library)
# 3 : use the Block version of the Soft-DTW (implemented in the library)
# 4 : use the Block version of the Soft-DTW divergence (implemented in the library)
loss_type_to_use = 3

# Other parameters
use_cuda = False
n_average = 10
pc_name = "CUDA_WSL"
save_results = False

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if loss_type_to_use == 0 and not import_soft_dtw_rust :
    raise ValueError("The Rust implementation of the soft-DTW is not available. Please install the package from https://pypi.org/project/soft-dtw-rust/")

device = 'cuda' if torch.cuda.is_available() and use_cuda else 'cpu'

def repeat_inference(x, model, n_average : int, recon_loss_type : int) :
    time_list_inference = []
    time_list_inference_and_loss = []

    if recon_loss_type > 0 :
        train_config = ct.get_config_hierarchical_vEEGNet_training()
        train_config['device'] = 'cpu'
        train_config['gamma_dtw'] = 0.1
        train_config['recon_loss_type'] = recon_loss_type
        train_config['device'] = device
        if recon_loss_type == 3 or recon_loss_type == 4 : train_config['block_size'] = 100
        loss_function_function = loss_function.hvEEGNet_loss(train_config)

    with torch.no_grad() :
        for i in range(n_average) :
            # Start time
            time_start = time.time()

            # Forward pass
            x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list = model(x)

            # Save inference time
            time_inference = time.time() - time_start

            # Compute loss
            if recon_loss_type == 0 :
                loss_value = soft_dtw_rust.compute_sdtw_2d(x.squeeze().numpy().astype('float64'), x_r.squeeze().numpy().astype('float64'), 1)
            else :
                loss_value = loss_function_function .compute_loss(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)

            # Save computations times
            time_list_inference.append(time_inference)
            time_list_inference_and_loss.append(time.time() - time_start)

    return time_list_inference, time_list_inference_and_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute the average inference time

time_average_matrix = np.zeros((len(C_list), len(T_list)))
time_std_matrix = np.zeros((len(C_list), len(T_list)))

for i in range(len(C_list)) : # Loop over the number of channels
    C = C_list[i]
    print("i = {} (C = {})".format(i, C))
    for j in range(len(T_list)) : # Loop over the number of time samples
        T  = T_list[j]
        print("\tj = {} (T = {})".format(j, T))
        
        # Create model
        model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
        model_config['input_size'] = (1, 1, C, T)
        model_config['use_classifier'] = False
        model_hv = hvEEGNet.hvEEGNet_shallow(model_config)
        model_hv.to(device)
        model_hv.eval()
        
        # Create synthetic data
        x = torch.rand(1, 1, C, T).to(device)
        
        # Compute inference time
        time_list_inference, time_list_inference_and_loss = repeat_inference(x, model_hv, n_average, loss_type_to_use)

        time_average_matrix[i, j] = np.mean(time_list_inference_and_loss)
        time_std_matrix[i, j] = np.std(time_list_inference_and_loss)

        if save_results :
            loss_type_str = 'SDTW_rust' if loss_type_to_use == 0 else 'SDTW_standard' if loss_type_to_use == 1 else 'SDTW_divergence' if loss_type_to_use == 2 else 'SDTW_block' if loss_type_to_use == 3 else 'SDTW_block_divergence'

            # Create the path if it does not exist
            path_save = 'Saved_results/computation time/inference_time/{}/'.format(pc_name)
            os.makedirs(path_save, exist_ok = True)
            path_save = 'Saved_results/computation time/inference_time/{}/C_{}_T_{}_{}'.format(pc_name, C, T, loss_type_str)

            # Save matrix in npy format
            np.save(path_save + '_time_list_inference.npy', time_list_inference)
            np.save(path_save + '_time_list_inference_and_loss.npy', time_list_inference_and_loss)

            # Save matrix in text format
            # np.save(path_save + '_time_list_inference.txt', time_list_inference)
            # np.save(path_save + '_time_list_inference_and_loss.txt', time_list_inference_and_loss)
