"""
Similar to version 1. But instead of plotting the average of std it creates a 2d image plot, with samples on x-axis and channel on y-axis
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import toml
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from library.dataset import download
from library.analysis import support
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

plot_config = dict(
    use_TkAgg_backend = False,
    figsize = (22, 16),
    fontsize = 20,
    cmap = 'Reds',
    aspect = 'auto',
    height_ratios = [3, 1],
    max_std_multiplier_cmap = 2,
    max_std_multiplier_plot = 1.5,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def plot_function(std_ch, ch_list, label, plot_config) :
    average_std = std_ch.mean(axis = 1)

    fig, axs = plt.subplots(2, 1, figsize = plot_config['figsize'], height_ratios = plot_config['height_ratios'])
    
    # Plot the data
    img = axs[0].imshow(std_ch.T, 
                  cmap = plot_config['cmap'], aspect = plot_config['aspect'],
                  vmin = 0, vmax = average_std.max() * plot_config['max_std_multiplier_cmap']
                  )

    axs[1].plot(average_std, color = 'black')
    axs[1].fill_between(np.arange(std_ch.shape[0]), std_ch.mean(axis = 1) - std_ch.std(axis = 1), std_ch.mean(axis = 1) + std_ch.std(axis = 1), color = 'black', alpha = 0.2)
    
    # Aesthetic stuff axs[0] (std image)
    axs[0].set_yticks(np.arange(len(ch_list)) - 0.5)
    axs[0].set_yticks(np.arange(len(ch_list)), minor = True)
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])
    axs[0].set_yticklabels(ch_list, minor = True, fontsize = plot_config['fontsize'] - 5)
    axs[0].set_ylabel('Channels', fontsize = plot_config['fontsize'])

    # Aesthetic stuff axs[1] (std plot)
    axs[1].set_ylim([0, average_std.max() * plot_config['max_std_multiplier_plot']])
    axs[1].set_xlabel('Trial number', fontsize = plot_config['fontsize'])
    axs[1].set_ylabel('Std value', fontsize = plot_config['fontsize'])
    axs[1].tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
    

    # Aesthetic stuff both axs 
    for ax in axs:
        ax.set_xlim(0, std_ch_train.shape[0])
        ax.grid(True)
    
    # Other stuff
    fig.suptitle('Subject {} - {}'.format(subj, label), fontsize = plot_config['fontsize'] + 2)
    fig.tight_layout()
    fig.show()

    # Colorbar
    bounds = list(np.linspace(0, average_std.max() * plot_config['max_std_multiplier_cmap'], 15)) + [std_ch.max()]
    stretched_bounds = np.interp(np.linspace(0, 1, 257), np.linspace(0, 1, len(bounds)), bounds)
    norm = mpl.colors.BoundaryNorm(bounds, ncolors = 512)
    fig.subplots_adjust(right = 0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    # fig.colorbar(img, cax = cbar_ax, cmap = plot_config['cmap'], norm=norm, extend = 'neither', ticks=bounds)
    fig.colorbar(img, cax = cbar_ax, cmap = plot_config['cmap'])

    if plot_config['save_fig']:
        path_save = "Saved Results/d2a_analysis/stats_ch/std/"
        os.makedirs(path_save, exist_ok = True)
        fig.savefig(path_save + 'avg_per_trial_S{}_{}_V2.png'.format(subj, label), format = 'png')
        # fig.savefig(path_save + 'avg_per_trial_S{}.pdf'.format(subj), format = 'pdf')
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if plot_config['use_TkAgg_backend']:
    plt.switch_backend('TkAgg')

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]

    # Get dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    ch_list = train_dataset.ch_list
    
    # Get data (in numpy array)
    train_data = train_dataset.data.numpy().squeeze()
    test_data = test_dataset.data.numpy().squeeze()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the std for each channel

    std_ch_train = train_data.std(2)
    std_ch_test = test_data.std(2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    plot_function(std_ch_train, ch_list, 'Train', plot_config)
    plot_function(std_ch_test, ch_list, 'Test', plot_config)

