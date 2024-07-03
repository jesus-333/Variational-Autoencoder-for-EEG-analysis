"""
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal as signal

from library.analysis import support
from library.config import config_dataset as cd

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

tot_epoch_training = 80
repetition = 18
epoch = 80

t_min = 2
t_max = 6

compute_spectra_with_entire_signal = True
nperseg = 500

plot_fig_7 = True

if plot_fig_7 :
    # For figure 7
    subjects_list = [4, 5, 8]
    n_trial_list = [179, 220, 81]
    ch_list = ['C3', 'Cz', 'CP4']
    use_test_set_list = [False, False, False]
else :
    # For figure 8
    subjects_list = [2, 4, 9]
    n_trial_list = [0, 145, 250]
    ch_list = ['FC3', 'C5', 'C1']
    use_test_set_list = [False, False, False]

plot_config = dict(
    figsize = (20, 24),
    fontsize = 24,
    linewidth_original = 1,
    linewidth_reconstructed = 1,
    color_original = 'black',
    color_reconstructed = 'red',
    add_title = False,
    add_subtitle = True,
    save_fig = True,
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

xticks_time = None
xticks_time = [2, 3, 4, 5, 6]

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

plt.rcParams.update({'font.size': plot_config['fontsize']})
label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

fig, axs = plt.subplots(3, 2, figsize = plot_config['figsize'])

for i in range(len(subjects_list)) :
    subj = subjects_list[i]

    use_test_set = use_test_set_list[i]
    n_trial = n_trial_list[i]
    channel = ch_list[i]

    ax_time = axs[i, 0]
    ax_freq = axs[i, 1]

    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, 'hvEEGNet_shallow')

    # Decide if use the train or the test dataset
    if use_test_set: dataset = test_dataset
    else: dataset = train_dataset

    # Get trial and create vector for time and channel
    x, label = dataset[n_trial]
    tmp_t = np.linspace(2, 6, x.shape[-1])
    idx_t = np.logical_and(tmp_t >= t_min, tmp_t <= t_max)
    t = tmp_t[idx_t]
    idx_ch = dataset.ch_list == channel
    label_name = label_dict[int(label)]

    # Load weight and reconstruction
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    x_r = model_hv.reconstruct(x.unsqueeze(0)).squeeze()

    # Select channel and time samples
    x_original_to_plot = x[0, idx_ch, idx_t]
    x_r_to_plot = x_r[idx_ch, idx_t]

    if compute_spectra_with_entire_signal:
        x_original_for_psd = x[0, idx_ch, :].squeeze()
        x_r_for_psd = x_r[idx_ch, :].squeeze()
    else:
        x_original_for_psd = x_original_to_plot
        x_r_for_psd = x_r_to_plot

    # Compute PSD
    f, x_psd = signal.welch(x_original_for_psd, fs = 250, nperseg = nperseg)
    f, x_r_psd = signal.welch(x_r_for_psd, fs = 250, nperseg = nperseg)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in time domain

    ax_time.plot(t, x_original_to_plot, label = 'original signal',
                 color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax_time.plot(t, x_r_to_plot, label = 'reconstructed signal',
                 color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])
    ax_time.set_xlabel("Time [s]", fontsize = plot_config['fontsize'])
    ax_time.set_ylabel(r"Amplitude [$\mu$V]", fontsize = plot_config['fontsize'])
    # ax_time.legend()
    if xticks_time is not None: ax_time.set_xticks(xticks_time)
    ax_time.set_xlim([t_min, t_max])
    ax_time.grid(True)
    ax_time.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
    ax_time.tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'])

    if plot_config['add_title']: ax_time.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Plot in frequency domain

    ax_freq.plot(f, x_psd, label = 'original signal',
                 color = plot_config['color_original'], linewidth = plot_config['linewidth_original'])
    ax_freq.plot(f, x_r_psd, label = 'reconstructed signal',
                 color = plot_config['color_reconstructed'], linewidth = plot_config['linewidth_reconstructed'])

    ax_freq.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
    ax_freq.set_ylabel(r"PSD [$\mu V^2/Hz$]", fontsize = plot_config['fontsize'])
    ax_freq.set_xlim([0, 80])
    if i == 0: ax_freq.legend(fontsize = plot_config['fontsize'] - 2)
    ax_freq.grid(True)
    ax_freq.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
    ax_freq.tick_params(axis = 'both', which = 'minor', labelsize = plot_config['fontsize'])

    if plot_config['add_title']: ax_freq.set_title('S{} - Ch. {} - Trial {}'.format(subj, channel, n_trial))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if plot_config['add_subtitle'] :
        ax_time.text(0.5, -0.2, "{}) S{}, segment no. {}, ch. {}, time domain.".format(chr(97 + 2 * i), subj, n_trial + 1, channel),
                     transform = ax_time.transAxes, ha = "center", fontsize = plot_config['fontsize'] - 1)
        ax_freq.text(0.5, -0.2, "{}) S{}, segment no. {}, ch. {}, freq domain.".format(chr(97 + 2 * i + 1), subj, n_trial + 1, channel),
                     transform = ax_freq.transAxes, ha = "center", fontsize = plot_config['fontsize'] - 1)

fig.tight_layout()
fig.show()

if plot_config['save_fig']:
    path_save = "Saved Results/figure_paper_frontiers/"
    os.makedirs(path_save, exist_ok = True)
    if plot_fig_7 : path_save += "FIG_7_bad_reconstruction_saturation"
    else : path_save += "FIG_8_bad_reconstruction_other_artifacts"
    if plot_config['add_subtitle'] : path_save += "_with_subtitle"
    else : path_save += "_NO_subtitle"
    fig.savefig(path_save + ".png", format = 'png')
    fig.savefig(path_save + ".jpeg", format = 'jpeg')
    fig.savefig(path_save + ".eps", format = 'eps')
