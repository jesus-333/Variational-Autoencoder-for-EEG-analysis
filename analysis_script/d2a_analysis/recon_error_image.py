"""
From the table of the reconstruction error create an image.
The table are the one obtained with the scripts reconstruction_3.py
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

import numpy as np
import matplotlib.pyplot as plt
import os

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

invert_column_and_row = True # If true the image will be row = channels and columns = trials. If false keeps row = trials and columns = channels
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list = [2]

plot_config = dict(
    figsize = (20, 12),
    min_value = 0,
    max_value = 65,
    dynamic_max_value = False,
    n_time_std_max_value = 3,
    fontsize = 20,
    # colormap = 'RdYlGn_r',
    colormap = 'Reds',
    add_title = False,
    add_colorbar = True,
    single_figure = False,
    save_fig = True,
)

tot_epoch_training = 80
epoch_to_plot = 80

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the image

plt.rcParams.update({'font.size': plot_config['fontsize']})

# Channels list (used for the plot)
channel_list = np.asarray(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])

for subj in subj_list:
    # Create the array to save the data
    # 288 trial of train (session 1), 288 of test (session 2). The +3 is used to draw a line between the to sessions
    if invert_column_and_row:
        image_to_plot_train = np.zeros((22, 288))
        image_to_plot_test = np.zeros((22, 288))
    else:
        image_to_plot_train = np.zeros((288, 22))
        image_to_plot_test = np.zeros((288, 22))

    # Load train reconstruction error (session 1)
    path_load_train = 'Saved Results/repetition_hvEEGNet_{}/train/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_train = np.load(path_load_train).T if invert_column_and_row else np.load(path_load_train)

    # Load test reconstruction error (session 2)
    path_load_test = 'Saved Results/repetition_hvEEGNet_{}/test/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_test = np.load(path_load_test).T if invert_column_and_row else np.load(path_load_test)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Create and show the image

    if plot_config['single_figure'] :
        fig, ax_list = plt.subplots(1, 2, figsize = plot_config['figsize'])
    else :
        fig_train, ax_train = plt.subplots(1, 1, figsize = plot_config['figsize'])
        fig_test, ax_test = plt.subplots(1, 1, figsize = plot_config['figsize'])

        ax_list = [ax_train, ax_test]

    image_to_plot_list = [recon_error_train, recon_error_test]

    if plot_config['dynamic_max_value'] :
        plot_config['max_value'] = image_to_plot_list[0].mean() + plot_config['n_time_std_max_value'] * image_to_plot_list[0].std()

    for i in range(2):
        ax = ax_list[i]

        # if plot_config['dynamic_max_value'] :
        #     plot_config['max_value'] = image_to_plot_list[i].mean() + plot_config['n_time_std_max_value'] * image_to_plot_list[i].std()

        # Plot image
        im = ax.imshow(image_to_plot_list[i], cmap = plot_config['colormap'],
                        aspect = 'auto',
                        vmin = plot_config['min_value'], vmax = plot_config['max_value'])
    
        # Add labels to axis
        if invert_column_and_row:
            ax.set_xlabel('Repetitions', fontsize = plot_config['fontsize'])
            ax.set_ylabel('Channels', fontsize = plot_config['fontsize'])
        else:
            ax.set_ylabel('Repetitions', fontsize = plot_config['fontsize'])
            ax.set_xlabel('Channels', fontsize = plot_config['fontsize'])

        # if plot_config['add_title'] : ax.set_title('Subject {}'.format(subj), fontsize = plot_config['fontsize'] + 2)
        if plot_config['add_title'] : ax.set_title('Session {}'.format(i + 1), fontsize = plot_config['fontsize'] + 2)

        xticks = np.asarray([0, 1, 2, 3, 4, 5, 6]) * 48
        yticks = np.arange(22)
        ax.set_xticks(xticks - 0.5, labels = xticks, fontsize = plot_config['fontsize'])
        ax.set_yticks(yticks - 0.5, labels = [])
        ax.set_yticks(yticks, labels = channel_list, minor = True, fontsize = plot_config['fontsize'])
        ax.grid(True, color = 'black')

        if plot_config['add_colorbar']:
            if plot_config['single_figure'] :
                pass
                # fig.colorbar(im)
            else :
                if i == 0 : fig_train.colorbar(im)
                else : fig_train.colorbar(im)

    if plot_config['single_figure'] :
        fig.tight_layout()
        fig.subplots_adjust(right = 0.8)
        cbar_ax = fig.add_axes([0.85, 0.1, 0.05, 0.85])
        fig.colorbar(im, cax = cbar_ax)
        fig.show()
    else :
        fig_train.tight_layout()
        fig_train.show()

        fig_test.tight_layout()
        fig_test.show()

    if plot_config['save_fig']:
        # Create pat
        path_save = 'Saved Results/d2a_analysis/recon_error_image/'
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'image_recon_error_S{}'.format(subj)

        if plot_config['single_figure'] :
            fig.tight_layout()
            fig.savefig(path_save + ".png", format = 'png')
            fig.savefig(path_save + ".pdf", format = 'pdf')
        else :
            fig_train.savefig(path_save + "_TRAIN.png", format = 'png')
            fig_test.savefig(path_save + "_TEST.png", format = 'png')

            # fig_train.savefig(path_save + "_TRAIN.eps", format = 'eps')
            # fig_test.savefig(path_save + "_TEST.eps", format = 'eps')

            fig_train.savefig(path_save + "_TRAIN.pdf", format = 'pdf')
            fig_test.savefig(path_save + "_TEST.pdf", format = 'pdf')
