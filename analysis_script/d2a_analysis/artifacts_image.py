"""

"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

threshold = -1

plot_config = dict(
    figsize = (12, 8),
    save_fig = True,
)

tot_epoch_training = 80
epoch_to_plot = 80

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_artifacts_for_subject(recon_error_matrix, threshold : float = -1):
    if threshold < 0: threshold = np.mean(recon_error_matrix) + 2 * np.abs(np.std(recon_error_matrix))
    # print(threshold, np.mean(recon_error_matrix))

    artifacts_map = np.sum(recon_error_matrix >= threshold, 1)
    artifacts_map[artifacts_map != 0] = 1

    return artifacts_map

def plot_image(image_to_plot):
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax.imshow(image_to_plot_train, aspect = 'auto')

    xticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48 
    yticks = np.arange(9)
    ax.set_xticks(xticks - 0.5, labels = xticks)
    ax.set_yticks(yticks - 0.51, labels = [])
    ax.set_yticks(yticks, labels = yticks + 1, minor = True)
    ax.grid(True, color = 'black')

    fig.tight_layout()
    fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the histogram

image_to_plot_train = np.zeros((9, 288))
image_to_plot_test = np.zeros((9, 288))

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for i in range(len(subj_list)):
    subj = subj_list[i]

    # Load train reconstruction error (session 1)
    path_load_train = 'Saved Results/repetition_hvEEGNet_{}/train/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_train = np.load(path_load_train)

    # Load test reconstruction error (session 2)
    path_load_test = 'Saved Results/repetition_hvEEGNet_{}/test/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_test = np.load(path_load_test)

    artifacts_map_train = compute_artifacts_for_subject(recon_error_train, threshold)
    artifacts_map_test = compute_artifacts_for_subject(recon_error_test, threshold)
    
    image_to_plot_train[i, :] = artifacts_map_train
    image_to_plot_test[i, :] = artifacts_map_test

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Plot image

plot_image(image_to_plot_train)
plot_image(image_to_plot_test)


