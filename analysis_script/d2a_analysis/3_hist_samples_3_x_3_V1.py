"""
Create an histogram with all the samples of the 9 subjects of Dataset 2a. The plot are arranged in a 3 x 3 grid.
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import os

import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

factor_to_average = None
factor_to_average = 'channel'
# factor_to_average = 'time'

plot_config = dict(
    figsize = (30, 24),
    bins = 100,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = False,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Create figure for the histogram
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

    if factor_to_average is None:
        train_data = train_dataset.data.flatten().sort()[0]
        test_data = test_dataset.data.flatten().sort()[0]
    elif factor_to_average == 'channel': # Perform the average accross 
        train_data = train_dataset.data.squeeze().mean(1).flatten().sort()[0]
        test_data = test_dataset.data.squeeze().mean(1).flatten().sort()[0]
    elif factor_to_average == 'time':
        train_data = train_dataset.data.squeeze().mean(2).flatten().sort()[0]
        test_data = test_dataset.data.squeeze().mean(2).flatten().sort()[0]
    else:
        raise ValueError("factor_to_average must be None or channel or time")

    ax = axs[idx_plot[i][0], idx_plot[i][1]]

    ax.hist(train_data, bins = plot_config['bins'], label = 'Train data', histtype = 'step', linewidth = 1, color = 'grey')
    ax.hist(test_data, bins = plot_config['bins'], label = 'Test data', histtype = 'step', linewidth = 1.5, color = 'black')

    ax.legend()
    ax.grid(True)
    ax.set_xlabel(r"Amplitude [$\mu$V]")
    ax.set_title("S{}".format(subj))

    if factor_to_average is None:
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, 600000])
    elif factor_to_average == 'channel':
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, 30000])
    elif factor_to_average == 'time':
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([0, 600])

if factor_to_average is None:
    fig.suptitle("All Samples")
elif factor_to_average == 'channel':
    fig.suptitle("Average along channels (288 x 22 x 1000 --> 288 x 1000)")
elif factor_to_average == 'time':
    fig.suptitle("Average along time (288 x 22 x 1000 --> 288 x 22)")
fig.tight_layout()
fig.show()

# TODO complete
path_save = "TMP"
fig.savefig(path_save + ".png", format = 'png')
