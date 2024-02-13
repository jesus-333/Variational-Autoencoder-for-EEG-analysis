"""
Create an histogram with all the data of all the trials and all the channels for a specific subject

Optionally you can select only a specific channel
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import os

import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [3]
use_test_set = False

channel_to_plot = None

plot_config = dict(
    figsize = (16, 10),
    bins = 1000,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subj {}".format(subj))

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # select train/test data
    if use_test_set:
        dataset = test_dataset
        dataset_string = 'test'
    else:
        dataset = train_dataset
        dataset_string = 'train'
    
    # Flatten the data and (OPTIONAL) select a channel
    if channel_to_plot is not None:
        idx_ch = dataset.ch_list == channel_to_plot
        data = dataset.data.squeeze()[:, idx_ch, :].flatten()
    else:
        data = dataset.data.flatten()

    # Plot the data
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    ax.hist(data, bins = plot_config['bins'], label = 'S{}'.format(subj))

    # ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"Amplitude [$\mu$V]")
    ax.set_title("S{} - {}".format(subj, dataset_string))
    ax.set_xlim([-100, 100])
    ax.set_ylim([0, 70000])
    
    if plot_config['use_log_scale_x']: ax.set_xscale('log')
    if plot_config['use_log_scale_y']: ax.set_yscale('log')

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        # Create pat
        path_save = 'Saved Results/d2a_analysis/hist_samples/bins {}/'.format(plot_config['bins'])
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'hist_samples_S{}_{}_bins_{}'.format(subj, dataset_string, plot_config['bins'])
        if plot_config['use_log_scale_x']: path_save += '_LOGSCALE_X'
        if plot_config['use_log_scale_y']: path_save += '_LOGSCALE_Y'
        fig.savefig(path_save + ".png", format = 'png')
        # fig.savefig(path_save + ".eps", format = 'eps')
