"""
Similar to V1 but the data are divided between the various classes instead of train/test
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import os

import torch
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

dataset_to_use = 'train'
# dataset_to_use = 'test'
# dataset_to_use = 'both'

factor_to_average = None
factor_to_average = 'channel'
# factor_to_average = 'time'

normalize_hist = True

plot_config = dict(
    figsize = (30, 24),
    bins = 50,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = False,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

# Create figure for the histogram
plt.rcParams.update({'font.size': plot_config['fontsize']})
fig, axs = plt.subplots(3, 3, figsize = plot_config['figsize'])

# Indices of the plot in the figres
idx_plot = []
for i in range(3):
    for j in range(3):
        idx_plot.append([i, j])

# Compute and plot histogram
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
    elif dataset_to_use == 'both':
        data_1 = train_dataset.data.squeeze()
        data_2 = test_dataset.data.squeeze()
        data = torch.cat((data_1, data_2), axis = 0)

        labels_1 = train_dataset.labels
        labels_2 = test_dataset.labels
        labels = torch.cat((labels_1, labels_2), axis = 0)

        del data_1, data_2, labels_1, labels_2
    else:
        raise ValueError("dataset_to_use must have value train or test or both")

    ax = axs[idx_plot[i][0], idx_plot[i][1]]

    for i in range(4): # Cycle through labels
        data_class = data[labels == i]

        if factor_to_average is None:
            data_class = data_class.flatten().sort()[0]
        elif factor_to_average == 'channel': # Perform the average accross
            data_class = data_class.mean(1).flatten().sort()[0]
        elif factor_to_average == 'time':
            data_class = data_class.mean(2).flatten().sort()[0]
        else:
            raise ValueError("factor_to_average must have value None or channel or time")

        ax.hist(data_class, bins = plot_config['bins'], density = normalize_hist,
                label = label_dict[i], histtype = 'step', linewidth = 1
                )

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"Amplitude [$\mu$V]")
    ax.set_title("S{}".format(subj))

    if normalize_hist :
        pass
    else :
        if factor_to_average is None:
            pass
            ax.set_xlim([-50, 50])
            ax.set_ylim([0, 300000])
        elif factor_to_average == 'channel':
            ax.set_xlim([-40, 40])
            ax.set_ylim([0, 13000])
        elif factor_to_average == 'time':
            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([0, 300])

if factor_to_average is None:
    fig.suptitle("{} data - All Samples".format(dataset_to_use))
elif factor_to_average == 'channel':
    fig.suptitle("{} data - Average along channels (288 x 22 x 1000 --> 288 x 1000)".format(dataset_to_use))
elif factor_to_average == 'time':
    fig.suptitle("{} data - Average along time (288 x 22 x 1000 --> 288 x 22)".format(dataset_to_use))
fig.tight_layout()
fig.show()

# TODO complete
path_save = "TMP"
fig.savefig(path_save + ".png", format = 'png')
