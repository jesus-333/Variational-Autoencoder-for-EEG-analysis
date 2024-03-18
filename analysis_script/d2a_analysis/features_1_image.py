"""
Compute the energy for each channel/trials and create an image with the values

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [3]

nperseg = 500
band_to_use = None

plot_config = dict(
    figsize = (20, 10),
    fontsize = 15,
    colormap = 'Reds',
    save_fig = False,
)
plt.rcParams.update({'font.size': plot_config['fontsize']})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_power(x_t, fs, nperseg, band_to_use = None):
    f, x_f = welch(x_t, fs, nperseg = nperseg)

    if band_to_use is None :
        power_x_f = np.sum(x_f)
    else :
        power_x_f = np.sum(x_f[np.logical_and(f >= band_to_use[0], f <= band_to_use[1])])

    return f, x_f, power_x_f

def show_image(img, plot_config, channel_list, title = None):
    # Visualize image
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    ax.imshow(img, cmap = plot_config['colormap'], aspect = 'auto',
              vmin = plot_config['vmin'], vmax = plot_config['vmax'],
              )

    xticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48
    yticks = np.arange(22)
    ax.set_xticks(xticks - 0.5, labels = xticks)
    ax.set_yticks(yticks - 0.5, labels = [])
    ax.set_yticks(yticks, labels = channel_list, minor = True)
    ax.grid(True, color = 'black')
    if title is not None : ax.set_title(title)

    fig.tight_layout()
    fig.show()

def print_info(subj, img_train, img_test):
    # Print info
    str_info = ""
    str_info += "Data S{} - Train\n".format(subj)
    str_info += "\tmean = {}\n".format(img_train.mean())
    str_info += "\tstd  = {}\n".format(img_train.std())
    str_info += "\tmax  = {}\n".format(img_train.max())
    str_info += "\tmin  = {}\n".format(img_train.min())

    str_info += "Data S{} - Test\n".format(subj)
    str_info += "\tmean = {}\n".format(img_test.mean())
    str_info += "\tstd  = {}\n".format(img_test.std())
    str_info += "\tmax  = {}\n".format(img_test.max())
    str_info += "\tmin  = {}\n".format(img_test.min())

    print(str_info)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

fs = 250

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subject {}".format(subj))

    # Get subject dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    ch_list = train_dataset.ch_list

    img_train, img_test = np.zeros((len(ch_list), len(train_dataset))), np.zeros((len(ch_list), len(test_dataset)))

    for idx_trial in range(len(train_dataset)):
        for idx_ch in range(len(ch_list)):
            # Get trail for train/test data
            x_t_train = train_dataset.data.squeeze()[idx_trial, idx_ch]
            x_t_test = test_dataset.data.squeeze()[idx_trial, idx_ch]
            
            # Compute power (OPTIONALLY you could use a specific band and not the entire spectrum)
            f, x_f_train, power_x_f_train = compute_power(x_t_train, fs, nperseg, band_to_use)
            f, x_f_test, power_x_f_test = compute_power(x_t_test, fs, nperseg, band_to_use)

            # Save the results
            img_train[idx_ch, idx_trial] = power_x_f_train
            img_test[idx_ch, idx_trial] = power_x_f_test
    
    # Show image
    plot_config['vmin'], plot_config['vmax'] = img_train.min(), img_train.max()
    show_image(img_train, plot_config, ch_list, "Train Data")

    plot_config['vmin'], plot_config['vmax'] = img_train.min(), img_train.max() # In this way the colorscale max and min have the same scale of the the train data.
    show_image(img_test, plot_config, ch_list, "Test Data")

    # Print info
    print_info(subj, img_train, img_test)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
