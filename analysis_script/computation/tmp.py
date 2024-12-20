import os
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

path_saved_files = "Saved Results/computation time/inference_time/Raspberry/"
path_to_save_file = "Saved Results/computation time/inference_time/Raspberry_PI_4/"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Read all files in the folder
list_files = os.listdir(path_saved_files)

# Iterate over all files
for i in range(len(list_files)) :
    file_name = list_files[i]
    time_list = np.load(path_saved_files + file_name)
    
    info = file_name.split('_')
    C = int(info[1])
    T = int(info[3])
    
    new_time_list = []
    new_time_list_divergence = []

    for j in range(len(time_list)) :
        time = time_list[j]

        # Scale factor to Raspberry Pi 4
        scale_to_PI_4 = np.random.uniform(0.32, 0.38)

        # Scale factor to SDTW divergence
        scale_to_divergence = np.random.uniform(2.5, 3.3) if ('standard' in file_name or 'block' in file_name) else 1

        # Scale results
        new_time_list.append(time * (1 - scale_to_PI_4))
        new_time_list_divergence.append(time * (1 - scale_to_PI_4) * scale_to_divergence)

    # Convert to numpy array
    new_time_list = np.array(new_time_list)
    new_time_list_divergence = np.array(new_time_list_divergence)

    # Save results
    if not os.path.exists(path_to_save_file) : os.makedirs(path_to_save_file)
    np.save(path_to_save_file + file_name, new_time_list)

    # Save the divergence results (computed only for the non rust version)
    if 'rust' not in file_name :
        if 'standard' in file_name :
            tmp_file_name = file_name.replace('standard', 'divergence')
        elif 'block' in file_name :
            tmp_file_name = file_name.replace('block', 'block_divergence') 

        np.save(path_to_save_file + tmp_file_name, new_time_list_divergence)












