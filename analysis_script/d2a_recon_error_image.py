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
# subj_list = [2]

plot_config = dict(
    width_separation_line = 4,
    min_value = 0,
    max_value = 50,
    colormap = 'RdYlGn_r',
    save_fig = True,
)

tot_epoch_training = 80
epoch_to_plot = 80

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get the data and create the image


for subj in subj_list:
    # Create the array to save the data
    # 288 trial of train (session 1), 288 of test (session 2). The +3 is used to draw a line between the to sessions
    if invert_column_and_row:
        image_to_plot = np.zeros((22, (288 * 2) + plot_config['width_separation_line']))
    else:
        image_to_plot = np.zeros(((288 * 2) + plot_config['width_separation_line'], 22))

    # Load train reconstruction error (session 1)
    path_load_train = 'Saved Results/repetition_hvEEGNet_{}/train/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_train = np.load(path_load_train).T if invert_column_and_row else np.load(path_load_train)

    # Load test reconstruction error (session 2)
    path_load_test = 'Saved Results/repetition_hvEEGNet_{}/test/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch_to_plot)
    recon_error_test = np.load(path_load_test).T if invert_column_and_row else np.load(path_load_test)

    # Save the data
    if invert_column_and_row:
        image_to_plot[:, 0:288]  = recon_error_train
        image_to_plot[:, 288 + plot_config['width_separation_line']:]   = recon_error_test
    else:
        image_to_plot[0:288, :]  = recon_error_train
        image_to_plot[288 + plot_config['width_separation_line']:, :]   = recon_error_test

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Create and show the image

    fig, ax = plt.subplots()
    
    # Plot image
    ax.imshow(image_to_plot, cmap = plot_config['colormap'],
              aspect = 'auto', 
              vmin = plot_config['min_value'], vmax = plot_config['max_value'])
    
    if invert_column_and_row:
        # Line to divede the two sessions
        ax.axvline(x = 288, color = 'black', linewidth = 1.5)

        # Add labels to axis
        ax.set_xlabel('Trials')
        ax.set_ylabel('Channels')
    else:
        ax.axhline(y = 288, color = 'black', linewidth = 1.5)
        ax.set_ylabel('Trials')
        ax.set_xlabel('Channels')

    ax.set_title('Subject {}'.format(subj))
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        # Create pat
        path_save = 'Saved Results/d2a_analysis/'
        os.makedirs(path_save, exist_ok = True)
        
        # Save fig
        path_save += 'image_recon_error_S{}'.format(subj)
        fig.savefig(path_save + ".png", format = 'png')
