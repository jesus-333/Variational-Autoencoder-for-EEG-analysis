"""
Use the output of the script compute_wasserstein_distance_latent_space.py to compute the average wasserstein distance between repetitions

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os

import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Subjects used for the plot
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Used the results obtained with the specifi training repetition at the specific epoch
epoch = 80
repetition_list = np.arange(20)

# If True the distance is computed between array of sampled data (i.e. obtained sampling from the distribution)
# Otherwise the array of means will be used as representative of the data
sample_from_latent_space = True

plot_config = dict(
	figsize = (12, 8),
	fontsize = 15,
	save_fig = True,
)
plt.rcParams.update({'font.size': plot_config['fontsize']})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

distance_matrix_1_list = []
distance_matrix_2_list = []

for i in range(len(repetition_list)) : # Iterate through training repetition
    repetition = repetition_list[i]

    # Load wasserstein distance
    path_wasserstrain_distance_session_1 = 'Saved Results/wasserstein/distance_train_session_1_epoch_{}_rep_{}'.format(epoch, repetition)
    path_wasserstrain_distance_session_2  = 'Saved Results/wasserstein/distance_train_session_2_epoch_{}_rep_{}'.format(epoch, repetition)

    # Add suffix to indicate if computation is between latent space samples or mu tensor
    if sample_from_latent_space :
        path_wasserstrain_distance_session_1 += '_latent_space_samples.npy'
        path_wasserstrain_distance_session_2 += '_latent_space_samples.npy'
    else :
        path_wasserstrain_distance_session_1 += '_mu_tensor.npy'
        path_wasserstrain_distance_session_2 += '_mu_tensor.npy'

    # Load the file with the wasserstein distances (session 1)
    if os.path.isfile(path_wasserstrain_distance_session_1) :
        distance_matrix_1 = np.load(path_wasserstrain_distance_session_1)
        distance_matrix_1_list.append(distance_matrix_1)

    # Load the file with the wasserstein distances (session 2)
    if os.path.isfile(path_wasserstrain_distance_session_2) :
        distance_matrix_2 = np.load(path_wasserstrain_distance_session_2)
        distance_matrix_2_list.append(distance_matrix_2)

# Convert list in numpy array
distance_matrix_1_list = np.asarray(distance_matrix_1_list)
distance_matrix_2_list = np.asarray(distance_matrix_2_list)

# Compute the average distance
distance_matrix_1_average = np.mean(distance_matrix_1_list, 0)
distance_matrix_2_average = np.mean(distance_matrix_2_list, 0)

# Create path to save the results
path_save_session_1 = 'Saved Results/wasserstein/distance_train_session_1_epoch_{}_average'.format(epoch, repetition)
path_save_session_2 = 'Saved Results/wasserstein/distance_train_session_2_epoch_{}_average'.format(epoch, repetition)
if sample_from_latent_space :
    path_save_session_1 += '_latent_space_samples.npy'
    path_save_session_2  += '_latent_space_samples.npy'
else :
    path_save_session_1 += '_mu_tensor.npy'
    path_save_session_2 += '_mu_tensor.npy'

# Save the results
np.save(path_save_session_1, distance_matrix_1_average)
np.save(path_save_session_2, distance_matrix_2_average)
