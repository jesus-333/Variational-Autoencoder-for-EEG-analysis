"""
@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import os

import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Subjects used for the plot
subj_train_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Used the results obtained with the specific training repetition at the specific epoch
epoch = 80
repetition = -1 # If negative use the average between all repetition

use_raw_data = True

# If True the distance is computed between array of sampled data (i.e. obtained sampling from the distribution)
# Otherwise the array of means will be used as representative of the data
sample_from_latent_space = True

plot_config = dict(
	figsize = (12, 8),
	fontsize = 15,
	save_fig = True,
)
plt.rcParams.update({'font.size': plot_config['fontsize']})

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for subj_train in subj_train_list :
    # Load wasserstein distance
    subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    if use_raw_data :
        path_wasserstrain_distance_session_1 = 'Saved Results/wasserstein/raw_data/distance_session_1_to_session_1_raw_data.npy'
        path_wasserstrain_distance_session_2 = 'Saved Results/wasserstein/raw_data/distance_session_1_to_session_2_raw_data.npy'
    else :
        # Load wasserstein distance
        if repetition > 0 :
            path_wasserstrain_distance_session_1 = 'Saved Results/wasserstein/distance_train_session_1_epoch_{}_rep_{}'.format(epoch, repetition)
            path_wasserstrain_distance_session_2  = 'Saved Results/wasserstein/distance_train_session_2_epoch_{}_rep_{}'.format(epoch, repetition)
        else :
            path_wasserstrain_distance_session_1 = 'Saved Results/wasserstein/distance_train_session_1_epoch_{}_average'.format(epoch)
            path_wasserstrain_distance_session_2  = 'Saved Results/wasserstein/distance_train_session_2_epoch_{}_average'.format(epoch)

        # Add suffix to indicate if computation is between latent space samples or mu tensor
        if sample_from_latent_space :
            path_wasserstrain_distance_session_1 += '_latent_space_samples.npy'
            path_wasserstrain_distance_session_2 += '_latent_space_samples.npy'
        else :
            path_wasserstrain_distance_session_1 += '_mu_tensor.npy'
            path_wasserstrain_distance_session_2 += '_mu_tensor.npy'

    distance_matrix_1 = np.load(path_wasserstrain_distance_session_1)
    distance_matrix_2 = np.load(path_wasserstrain_distance_session_2)

    distance_to_plot_1 = distance_matrix_1[subj_train - 1]
    distance_to_plot_2 = distance_matrix_2[subj_train - 1]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Load reconstruction error
    reconstruction_error_1 = np.zeros(9)
    reconstruction_error_2 = np.zeros(9)

    for i in range(len(subj_list)) :
        # Path of the saved reconstruction error
        if subj_list[i] == subj_train :
            if repetition > 0 :
                path_recon_error_session_1 = 'Saved Results/repetition_hvEEGNet_80/train/subj {}/recon_error_{}_rep_{}.npy'.format(subj_train, epoch, repetition)
                path_recon_error_session_2 = 'Saved Results/repetition_hvEEGNet_80/test/subj {}/recon_error_{}_rep_{}.npy'.format(subj_train, epoch, repetition)
            else :
                path_recon_error_session_1 = 'Saved Results/repetition_hvEEGNet_80/train/subj {}/recon_error_{}_average.npy'.format(subj_train, epoch)
                path_recon_error_session_2 = 'Saved Results/repetition_hvEEGNet_80/test/subj {}/recon_error_{}_average.npy'.format(subj_train, epoch)
        else :
            if repetition > 0 :
                path_recon_error_session_1 = 'Saved Results/repetition_hvEEGNet_80/train/subj {}/cross_subject/recon_error_cross_subj_{}_to_{}_epoch_{}_rep_{}.npy'.format(subj_train, subj_train, subj_list[i], epoch, repetition)
                path_recon_error_session_2 = 'Saved Results/repetition_hvEEGNet_80/test/subj {}/cross_subject/recon_error_cross_subj_{}_to_{}_epoch_{}_rep_{}.npy'.format(subj_train, subj_train, subj_list[i], epoch, repetition)
            else :
                path_recon_error_session_1 = 'Saved Results/repetition_hvEEGNet_80/train/subj {}/cross_subject/recon_error_subj_{}_to_{}_epoch_{}_average.npy'.format(subj_train, subj_train, subj_list[i], epoch)
                path_recon_error_session_2 = 'Saved Results/repetition_hvEEGNet_80/test/subj {}/cross_subject/recon_error_subj_{}_to_{}_epoch_{}_average.npy'.format(subj_train, subj_train, subj_list[i], epoch)
        
        # Load reconstruction error
        tmp_recon_error_1 = np.load(path_recon_error_session_1)
        tmp_recon_error_2 = np.load(path_recon_error_session_2)
        
        # Compute the average reconstruction error
        reconstruction_error_1[i] = np.mean(tmp_recon_error_1)
        reconstruction_error_2[i] = np.mean(tmp_recon_error_2)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Plot the average
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    for i in range(len(reconstruction_error_1)) :
        ax.plot(distance_to_plot_1[i], reconstruction_error_1[i],
                marker = '$S{}$'.format(subj_list[i]), color = 'green', markersize = 15
                )
        ax.plot(distance_to_plot_2[i], reconstruction_error_2[i], marker = '$S{}$'.format(subj_list[i]), color = 'red', markersize = 15)

    ax.set_ylabel("Reconstruction error")
    ax.set_xlabel("Wasserstein distance")
    ax.legend(["Session 1", "Session 2"])

    if use_raw_data :
        ax.set_title("Wasserstein distance from S{} (train data)".format(subj_train))
    else :
        ax.set_title("Results with hvEEGNet trained with S{} for {} epoch".format(subj_train, epoch))

    ax.set_xlim(np.sort(distance_to_plot_2)[1] * 0.9, np.sort(distance_to_plot_2)[-1] * 1.1)
    # ax.set_xlim(10, np.sort(distance_to_plot_2)[1] * 1.1)

    # ax.set_xlim([1540, 2800])
    # ax.set_ylim([0, 80])

    ax.grid(True)

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = 'Saved Results/wasserstein/'

        if use_raw_data :
            path_save += 'raw_data/plot/'
        else :
            if sample_from_latent_space :
                path_save += 'latent_space_samples/plot/'
            else :
                path_save += 'mu_tensor/plot/'
        os.makedirs(path_save, exist_ok = True)
    
        path_save += 'wass_vs_recon_error_S{}'.format(subj_train)

        if not use_raw_data :
            path_save += '_epoch_{}'.format(epoch)

        fig.savefig(path_save + ".png", format = 'png')
        # fig_freq.savefig(path_save + ".pdf", format = 'pdf')
