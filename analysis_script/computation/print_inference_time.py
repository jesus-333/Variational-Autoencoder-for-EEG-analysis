"""
Plot the results obtained with script inference_time_5.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import torch
import numpy as np
import matplotlib.pyplot as plt

from library.model import hvEEGNet
from library.training import loss_function
from library.config import config_model as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

C_list = [8, 22, 64]
T_list = (np.arange(19) + 2) * 50
loss_type_list = [0, 1, 2, 3, 4]
loss_type_list = [1]

# Other parameters
use_cuda = False
pc_name = "Raspberry"
pc_name = "Raspberry_PI_4"
# pc_name = "CPU_pc_unipd"
# pc_name = "CPU_Colab"

plot_config = dict(
    figsize = (16, 10),
    fontsize = 20,
    linewidth = 2,
    C_list = C_list,
    T_list = T_list,
    use_seconds_for_x_axis = True,
    fs = 250,
    y_axis_lim = 6,
    loss_type_list = loss_type_list,
    use_log_scale = False,
    save_fig = True,
    # extension_list = ['png', 'pdf', 'eps']
    extension_list = ['png']
)

loss_to_string_dict = {
    0 : 'SDTW_rust',
    1 : 'SDTW_standard',
    2 : 'SDTW_divergence',
    3 : 'SDTW_block',
    4 : 'SDTW_block_divergence'
}

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot function

def create_plot(avg_matrix : np.array, std_matrix : np.array, plot_config : dict, title : str) :
    # Create the figure
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])

    if plot_config['use_seconds_for_x_axis'] :
        plot_config['T_list'] = np.array(plot_config['T_list']) / plot_config['fs']

    # Loop over the loss types
    for i in range(len(plot_config['loss_type_list'])) :
        loss_type = plot_config['loss_type_list'][i]
        loss_str = plot_config['loss_to_string_dict'][loss_type]

        # Loop over the number of channels
        for j in range(len(plot_config['C_list'])) :
            C = plot_config['C_list'][j]

            # Plot the results
            # ax.errorbar(plot_config['T_list'], avg_matrix[i, j, :], yerr = std_matrix[i, j, :], 
            #             label = "C = {}, loss = {}".format(C, loss_str),
            #             )

            ax.plot(plot_config['T_list'], avg_matrix[i, j, :], 
                    label = "C = {}, loss = {}".format(C, loss_str), linewidth = plot_config['linewidth']
                    )
            # ax.fill_between(plot_config['T_list'], avg_matrix[i, j, :] - std_matrix[i, j, :], avg_matrix[i, j, :] + std_matrix[i, j, :], alpha = 0.3)
    
    # Plot straight line for reference
    if plot_config['use_seconds_for_x_axis'] :
        ax.plot(plot_config['T_list'], plot_config['T_list'], 'k--', label = "Reference ")

    # Other plot settings
    ax.set_xlim([plot_config['T_list'][0], plot_config['T_list'][-1]])
    if plot_config['y_axis_lim'] : 
        ax.set_ylim([0, plot_config['y_axis_lim']])
    if plot_config['use_seconds_for_x_axis'] :
        ax.set_xlabel("Trial Length [s]", fontsize = plot_config['fontsize'])
    else :
        ax.set_xlabel("Number of time samples (T)", fontsize = plot_config['fontsize'])
    ax.set_ylabel("Inference time [s]", fontsize = plot_config['fontsize'])
    ax.tick_params(axis = 'both', labelsize = plot_config['fontsize'])
    ax.legend(fontsize = plot_config['fontsize'])
    ax.set_title(title, fontsize = plot_config['fontsize'])
    ax.tick_params(axis = 'both', which = 'major', labelsize = plot_config['fontsize'])
    if plot_config['use_log_scale'] : ax.set_yscale('log')
    ax.grid(True)

    if pc_name == 'Raspberry' :
        min_xlim = 100 if not plot_config['use_seconds_for_x_axis'] else 100 / 250
        ax.set_xlim([min_xlim, plot_config['T_list'][-1]])
        ax.set_ylim([-0.1, 10])

    fig.tight_layout()
    fig.show()

    return fig, ax

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load the results

plot_config['loss_to_string_dict'] = loss_to_string_dict

# Variables to store the results
inference_avg = np.zeros((len(loss_type_list), len(C_list), len(T_list)))
inference_std = np.zeros((len(loss_type_list), len(C_list), len(T_list)))
inference_and_loss_avg = np.zeros((len(loss_type_list), len(C_list), len(T_list)))
inference_and_loss_std = np.zeros((len(loss_type_list), len(C_list), len(T_list)))

for i in range(len(loss_type_list)) :
    loss_type_to_use = loss_type_list[i]
    loss_type_str = loss_to_string_dict[loss_type_to_use]
    for j in range(len(C_list)) :
        C = C_list[j]
        for k in range(len(T_list)) :
            T = T_list[k]
            
            # Path to results files
            path = "Saved Results/computation time/inference_time/{}/".format(pc_name)
            path_load_only_inference = path + "C_{}_T_{}_{}_time_list_inference.npy".format(C, T, loss_type_str)
            path_load_inference_and_training = path + "C_{}_T_{}_{}_time_list_inference_and_loss.npy".format(C, T, loss_type_str)

            # Load the results
            inference_avg[i, j, k] = np.mean(np.load(path_load_only_inference))
            inference_std[i, j, k] = np.std(np.load(path_load_only_inference))
            inference_and_loss_avg[i, j, k] = np.mean(np.load(path_load_inference_and_training))
            inference_and_loss_std[i, j, k] = np.std(np.load(path_load_inference_and_training))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the results
    
# fig_only_inference, ax_only_inference = create_plot(inference_avg, inference_std, plot_config, "Inference time")
fig_inference_and_loss, ax_inference_and_loss = create_plot(inference_and_loss_avg, inference_and_loss_std, plot_config, "Inference and loss time")

if plot_config['save_fig'] :
    # fig_list = [fig_only_inference, fig_inference_and_loss]
    fig_list = [fig_inference_and_loss]
    path_save = path + 'average_inference_time'

    for fig in fig_list :
        for extension in plot_config['extension_list'] :
            fig.savefig(path_save + '.' + extension, format = extension)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# For some PC compute the differce between standard and rust version

pc_name_list = ["CPU_pc_unipd", "CPU_Colab"]

if pc_name in pc_name_list :
    if 0 in loss_type_list and 1 in loss_type_list :
        inference_and_loss_avg_rust = inference_and_loss_avg[0, :, :]
        inference_and_loss_avg_standard = inference_and_loss_avg[1, :, :]

        print("Rust version is executed in {}% of the time of the standard version".format(np.mean(inference_and_loss_avg_rust / inference_and_loss_avg_standard) * 100))




