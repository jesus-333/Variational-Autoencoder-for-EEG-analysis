"""
From the data obtained with average_dtw_per_channels.py compute the average and std for different number of samples
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_weights = 2
subj_data    = 2

idx_ch = [0,1,2]
idx_ch = [0]

# How many samples take/add to compute mean and std
sample_step = 10

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Path for the data
path_load = "Saved Results/variability_study/"
if subj_data == subj_weights: path_load += "intra_subject_dtw_S{}.npy".format(subj_weights)
else: path_load += "cross_subject_dtw_source_S{}_target_S{}.npy".format(subj_weights, subj_data)
dtw_values = np.load(path_load)

for i in range(len(idx_ch)):

    # Remove 0s values
    data = dtw_values[i]
    data = data[data != 0] # N.b. with this operation we obtain an array

   # Variable used to sample from the array of DTW values
    dtw_idx_to_sample = np.arange(len(data)) # Array with all the indices of the DTW values array
    dtw_sampled_idx = [] # List with the indices sampled up to now
    continue_sampling = True

    dtw_mean_array = []
    dtw_std_array = []
    values_sampled = []

    while continue_sampling:
        # Get new indices
        if len(dtw_idx_to_sample) > sample_step:
            idx_of_idx_to_sample = np.arange(len(dtw_idx_to_sample))
            tmp_idx = np.random.choice(idx_of_idx_to_sample, sample_step, replace = False)
            new_idx_for_dtw_values = dtw_idx_to_sample[tmp_idx]
        else:
            new_idx_for_dtw_values = dtw_idx_to_sample
            continue_sampling = False
        
        # Add indices to the list of sampled indices and get DTW values
        dtw_sampled_idx += list(new_idx_for_dtw_values)
        dtw_values_sampled = data[dtw_sampled_idx]

        # Compute mean, std and number of values sampled
        dtw_mean_array.append(dtw_values_sampled.mean())
        dtw_std_array.append(dtw_values_sampled.std())
        values_sampled.append(len(dtw_sampled_idx))
        
        # Remove new indices from the list of all indices
        if continue_sampling: dtw_idx_to_sample = np.delete(dtw_idx_to_sample, tmp_idx)
    
    # Convert list into numpy array
    dtw_mean_array = np.asarray(dtw_mean_array)
    dtw_std_array = np.asarray(dtw_std_array)

    fig, ax = plt.subplots(1, 1, figsize = (15, 10))
    ax.plot(values_sampled, dtw_mean_array)
    ax.fill_between(values_sampled, dtw_mean_array + dtw_std_array, dtw_mean_array - dtw_std_array, alpha = 0.4)

    ax.set_xlabel('Values sampled')
    ax.set_ylabel('Average DTW')
    
    ax.set_xlim([values_sampled[0], values_sampled[-1]])

    ax.grid(True)

    fig.tight_layout()
    fig.show()
