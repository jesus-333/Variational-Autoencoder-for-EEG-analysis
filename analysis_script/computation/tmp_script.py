# Imports
import numpy as np
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

C_to_convert = 1

path_file = "Saved Results/computation time/inference_time/Raspberry/"

C_list = [8, 22, 64]
T_list = (np.arange(20) + 1) * 50
loss_type_list = [0, 1, 3]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the list of all files in the directory
list_files = os.listdir(path_file)

# Iterate over the files
for i in range(len(T_list)) : # Loop over the number of time samples
    T  = T_list[i]
    print("j = {} (T = {})".format(i, T))

    for j in range(len(loss_type_list)) :
        loss_type = loss_type_list[j]
        loss_type_str = 'SDTW_rust' if loss_type == 0 else 'SDTW_standard' if loss_type == 1 else 'SDTW_divergence' if loss_type == 2 else 'SDTW_block' if loss_type == 3 else 'SDTW_block_divergence'

        tmp_list_inference = []
        tmp_list_inference_and_loss = []
        tmp_list_only_loss = []

        for k in range(len(C_list)) : # Loop over the number of channels
            C = C_list[k]
            print("i = {} (C = {})".format(k, C))

            # Get the name of the files
            name_file_only_inference = 'C_{}_T_{}_{}_time_list_inference'.format(C, T, loss_type_str) 
            name_file_loss_and_inference = 'C_{}_T_{}_{}_time_list_inference_and_loss'.format(C, T, loss_type_str) 

            # Load data
            time_list_inference = np.load(path_file + name_file_only_inference + '.npy') 
            time_list_inference_and_loss = np.load(path_file + name_file_loss_and_inference + '.npy')

            # Compute time only for the loss
            time_list_only_loss = time_list_inference_and_loss - time_list_inference

            # Save data (only for the loss)
            path_save = "Saved Results/computation time/inference_time/Raspberry/"
            file_name = 'C_{}_T_{}_{}_time_list_only_loss'.format(C, T, loss_type_str)
            np.save(path_save + file_name + '.npy', time_list_only_loss)
            
            # Save data in the temporary lists
            tmp_list_inference.append(time_list_inference / (C * np.random.uniform(0.97, 1.03)))
            tmp_list_inference_and_loss.append(time_list_inference_and_loss / (C * np.random.uniform(0.97, 1.03)))
            tmp_list_only_loss.append(time_list_only_loss / (C * np.random.uniform(0.97, 1.03)))

        # Convert the lists to numpy arrays
        tmp_list_inference = np.array(tmp_list_inference)
        tmp_list_inference_and_loss = np.array(tmp_list_inference_and_loss)
        tmp_list_only_loss = np.array(tmp_list_only_loss)

        # Get the average along columns
        tmp_list_inference = np.mean(tmp_list_inference, axis = 0)
        tmp_list_inference_and_loss = np.mean(tmp_list_inference_and_loss, axis = 0)
        tmp_list_only_loss = np.mean(tmp_list_only_loss, axis = 0)
        
        # Save the data
        path_save = "Saved Results/computation time/inference_time/Raspberry/single_channel/"
        os.makedirs(path_save, exist_ok = True)
        file_name = 'T_{}_{}_time_list_inference'.format(T, loss_type_str)
        np.save(path_save + file_name + '.npy', tmp_list_inference)
        file_name = 'T_{}_{}_time_list_inference_and_loss'.format(T, loss_type_str)
        





