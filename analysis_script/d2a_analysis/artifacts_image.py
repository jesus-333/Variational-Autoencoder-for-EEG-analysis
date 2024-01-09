"""

"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

threshold = 100

plot_config = dict(
    figsize = (12, 8),
    colormap = 'Reds',
    # colormap = 'Greys',
    save_fig = True,
)

tot_epoch_training = 80
epoch_to_plot = 80

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_artifacts_for_subject(recon_error_matrix, threshold : float = -1):
    if threshold < 0: threshold = np.mean(recon_error_matrix) + 3 * np.abs(np.std(recon_error_matrix))
    # print(threshold, np.mean(recon_error_matrix))

    artifacts_map = np.sum(recon_error_matrix >= threshold, 1)
    artifacts_map[artifacts_map != 0] = 1

    return artifacts_map

def compute_artifacts_for_subject_2(recon_error_matrix, threshold : float = -1):
    if threshold < 0: threshold = np.mean(recon_error_matrix) + 3 * np.abs(np.std(recon_error_matrix))
    # print(threshold, np.mean(recon_error_matrix))

    artifacts_map = np.sum(recon_error_matrix, 1)
    artifacts_map = artifacts_map > threshold

    return artifacts_map.astype(int)

def plot_image(image_to_plot, plot_config : dict, title = None):
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax.imshow(image_to_plot, aspect = 'auto', 
              cmap = plot_config['colormap'], interpolation='nearest'
              )

    xticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48 
    yticks = np.arange(9)
    ax.set_xticks(xticks - 0.5, labels = xticks)
    ax.set_yticks(yticks - 0.51, labels = [])
    ax.set_yticks(yticks, labels = yticks + 1, minor = True)
    ax.grid(True, color = 'black')
    
    ax.set_xlabel('Trials')
    ax.set_ylabel('Subjects')
    
    if title is not None : ax.set_title(title)

    fig.tight_layout()
    fig.show()
    
    return fig, ax

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the image

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

    # artifacts_map_train = compute_artifacts_for_subject(recon_error_train, threshold)
    # artifacts_map_test = compute_artifacts_for_subject(recon_error_test, threshold)
    
    artifacts_map_train = compute_artifacts_for_subject_2(recon_error_train, threshold)
    artifacts_map_test = compute_artifacts_for_subject_2(recon_error_test, threshold)
    
    image_to_plot_train[i, :] = artifacts_map_train
    image_to_plot_test[i, :] = artifacts_map_test
    
image_to_plot_train_original = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_train.csv')
image_to_plot_test_original = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_test.csv')

image_to_plot_train_original = image_to_plot_train_original.T.to_numpy()[1:, :]
image_to_plot_test_original = image_to_plot_test_original.T.to_numpy()[1:, :]


#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot image

plot_image(image_to_plot_train, plot_config, 'Session 1 (hvEEGNet)')
plot_image(image_to_plot_train_original, plot_config, 'Session 1 (original)')

plot_image(image_to_plot_test, plot_config, 'Session 2 (hvEEGNet)')
plot_image(image_to_plot_test_original, plot_config, 'Session 2 (original)')


