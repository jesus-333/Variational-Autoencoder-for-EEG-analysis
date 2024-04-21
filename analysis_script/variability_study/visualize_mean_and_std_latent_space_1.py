"""
This script produce a figure similar to Fig. 5 of Barandas et al. 2024 (https://www.sciencedirect.com/science/article/pii/S1566253523002944)

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import matplotlib.pyplot as plt
import numpy as np

from library.analysis import dtw_analysis, support
from library.dataset import preprocess as pp

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

# Subject to use for the weights of trained network.
# E.g. if subj_train = 3 the script load in hvEEGNet the weights obtained after the traiing with data from subject 3
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_train_list = [3]

# Training repetition and epoch of the weights
repetition = 5
epoch = 80

# The wesserstein distance will be computed respect this subjects
subj_for_distance_computation_list  = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_for_distance_computation_list  = [1]

# If True the distance is computed between array of sampled data (i.e. obtained sampling from the distribution)
# Otherwise the array of means will be used as representative of the data
sample_from_latent_space = False

plot_config = dict(
    figsize = (10, 8),
    fontsize = 15,
    save_fig = False,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(subj_train_list)) :
    # Get model
    subj_train = subj_train_list[i]
    dataset_config_train = cd.get_moabb_dataset_config([subj_train])
    dataset_config_train['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, _, model_hv = support.get_dataset_and_model(dataset_config_train, 'hvEEGNet_shallow')
    print("Subj train {}".format(subj_train))
    
    # Set model to evaluation mode
    model_hv.eval()

    # Load weights

    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(80, subj_train, repetition, epoch)
    
    # Get tensor with all training data
    x_train, _ = train_dataset[:]
    
    # Encode train data
    model_output_train_data = model_hv.encode(x_train, return_distribution = True)

    # Note on the encode function. By default the parameter return_distribution of the function is set to True.
    # This allow the function encode to return the sample from the latent space but also the mean and the variance of the latent space.
    # If it is set to False the output is tensor just before passing through the layer that map data in the mean and variance of latent space.
    # Note that model output is a list where the first element contains the tensor with the z samples from the latent space, the seond element the tensor of mean and the third element the tensor of variance
    
    mu = model_output_train_data[1].flatten(1)
    sigma = model_output_train_data[2].flatten(1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot the average of mean and std of the latent space distribution
    # Note that each eeg signal is mapped in different values for mean and std
    # So at the end I have 288 means and std

    shape_mu_before_flatten = model_output_train_data[1].shape
    
    # Update fontsize
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])
    
    xticks = np.arange(mu.shape[1])
    xticks_values = np.tile(np.linspace(2, 6, shape_mu_before_flatten[3]), shape_mu_before_flatten[1])
    
    # Plot the average
    axs[0].plot(xticks, mu.mean(0), label = 'Mean z')
    axs[1].plot(xticks, sigma.mean(0), label = 'Mean z')

    # Plot the variation of the average
    axs[0].fill_between(xticks, mu.mean(0) + mu.std(0), mu.mean(0) - mu.std(0), alpha = 0.5)
    axs[1].fill_between(xticks, sigma.mean(0) + sigma.std(0), sigma.mean(0) - sigma.std(0), alpha = 0.5)

    axs[0].set_title("Mean of z")
    axs[1].set_title("Std of z")

    for ax in axs :
        ax.grid(True)
        # ax.set_xticks(ticks = xticks, labels = xticks_values)
        # ax.set_xlim([300, 400])

    fig.tight_layout()
    fig.show()



