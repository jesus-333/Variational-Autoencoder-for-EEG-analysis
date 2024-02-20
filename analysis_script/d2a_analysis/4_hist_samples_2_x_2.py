"""
Compute the histogram for each class and for each subject and create 4 plot.
In each plot will represents a class and there will be 9 histograms, 1 for each subject

@author : Alberto (Jesus) Zancanaro
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os

import numpy as np
import matplotlib.pyplot as plt
import torch

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

dataset_to_use = 'train'
# dataset_to_use = 'test'
# dataset_to_use = 'train+test'

factor_to_average = None
factor_to_average = 'channel'
# factor_to_average = 'time'

normalize_hist = True

plot_config = dict(
    figsize = (30, 24),
    n_bins = 25,
    linewidth = 2,
    fontsize = 24,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [5,6,7,8,9]
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(2, 2, figsize = plot_config['figsize'])

idx_plot = []
for i in range(2):
    for j in range(2):
        idx_plot.append([i, j])

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subj {}".format(subj))

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    # Get train/test/both data
    if dataset_to_use == 'train':
        data = train_dataset.data.squeeze()
        labels = train_dataset.labels
    elif dataset_to_use == 'test':
        data = test_dataset.data.squeeze()
        labels = test_dataset.labels
    elif dataset_to_use == 'train+test':
        data_1 = train_dataset.data.squeeze()
        data_2 = test_dataset.data.squeeze()
        data = torch.cat((data_1, data_2), axis = 0)

        labels_1 = train_dataset.labels
        labels_2 = test_dataset.labels
        labels = torch.cat((labels_1, labels_2), axis = 0)

        del data_1, data_2, labels_1, labels_2
    else:
        raise ValueError("dataset_to_use must have value train or test or train+test")
    
    for j in range(4):
        # Get the data for specific class
        data_class = data[labels == j]

        # Get the axis for the plot
        ax = axs[idx_plot[j][0], idx_plot[j][1]]

        if factor_to_average is None:
            data_class = data_class.flatten().sort()[0]
            average_method_string = "no_average"
        elif factor_to_average == 'channel': # Perform the average accross
            data_class = data_class.mean(1).flatten().sort()[0]
            average_method_string = "average_channel"
        elif factor_to_average == 'time':
            data_class = data_class.mean(2).flatten().sort()[0]
            average_method_string = "average_time"
        else:
            raise ValueError("factor_to_average must have value None or channel or time")
        
        # Compute hist
        # p, x = np.histogram(data_class, plot_config['n_bins'], density = normalize_hist)
        # x = x[:-1] + (x[1] - x[0]) / 2

        # Plot the data
        # ax.plot(x, p, linewidth = plot_config['linewidth'], label = "S{}".format(subj))

        x_hist = ax.hist(data_class, plot_config['n_bins'], density = normalize_hist,
                    histtype = 'step', label = "S{}".format(subj), linewidth = plot_config['linewidth'],
                    )
        
        # Extra stuff for plot
        ax.legend()
        ax.grid(True)
        ax.set_xlabel(r"Amplitude [$\mu$V]")
        ax.set_title(label_dict[j])
        
        if factor_to_average is None:
            pass
        elif factor_to_average == 'channel': # Perform the average accross
            ax.set_ylim([0, 0.06])
            ax.set_xlim([-100, 100])
        elif factor_to_average == 'time':
            ax.set_ylim([0, 1.2])
            ax.set_xlim([-7.5, 7.5])

if factor_to_average is None:
    fig.suptitle("{} data - All Samples".format(dataset_to_use))
elif factor_to_average == 'channel':
    fig.suptitle("{} data - Average along channels (288 x 22 x 1000 --> 288 x 1000)".format(dataset_to_use))
elif factor_to_average == 'time':
    fig.suptitle("{} data - Average along time (288 x 22 x 1000 --> 288 x 22)".format(dataset_to_use))
fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = 'Saved Results/d2a_analysis/hist_samples/bins {}/'.format(plot_config['n_bins'])
    os.makedirs(path_save, exist_ok = True)

    path_save += 'hist_2_x_2_class_{}_{}{}'.format(dataset_to_use, average_method_string, "_NORMALIZE" if normalize_hist else "")
    fig.savefig(path_save + ".png", format = 'png')
