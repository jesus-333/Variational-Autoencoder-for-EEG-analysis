"""
Combine compute_outliers_2.py and compute_outliers_3.py and execute the stuff subject by subjcet.
I.e. It creates 9 plot and in each plot there is the average error of the outliers and the number of outliers
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.preprocessing import RobustScaler

try:
    from kneed import KneeLocator
except ImportError as e:
    print("Error -> ", e)
    raise ImportError("To run this script you need the kneed package")

from library.analysis import support
from library.config import config_dataset as cd

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

tot_epoch_training = 80
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [2, 5]
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
repetition_list = np.arange(19) + 1

use_test_set = False
save_outliers = True

method_std_computation = 2
normalize_recon_error = False
neighborhood_order_list = [5, 15]
knn_algorithm = 'auto'
s_knee = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 16,
    save_fig = True,
    color_1 = 'darkcyan',
    color_2 = 'skyblue',
    color_3 = 'red'
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if normalize_recon_error:
    norm_string = "NORMALIZED"
else:
    norm_string = "NOT_NORMALIZED"

output = support.compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, 
                                                              method_std_computation = method_std_computation, skip_run = True)
recon_loss_results_mean, recon_loss_results_std, recon_loss_to_plot_mean, recon_loss_to_plot_std = output

for neighborhood_order in neighborhood_order_list:
    plt.rcParams.update({'font.size': plot_config['fontsize']})

    for subj in subj_list:
        print(subj)
        
        fig, ax_error_outliers = plt.subplots(1, 1, figsize = plot_config['figsize'])
        ax_n_outliers = ax_error_outliers.twinx()
        
        idx_outliers_list = []
        average_error_outliers_list = []
        n_outliers_list = []
        
        average_error_subject_list = []
        std_error_subject_list = []

        for epoch in epoch_list:

            # Load the reconstruction error
            # path_recon_error = './Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch)
            # recon_error = np.load(path_recon_error)
            recon_error = recon_loss_results_mean[subj][epoch]

            # Apply the scaling to data
            if normalize_recon_error:
                scaler = RobustScaler()
                recon_error = scaler.fit_transform(recon_error)

            #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            # Outlier identifications (NORMALIZED)

            # Compute the KNN
            neighborhood_set   = knn(n_neighbors = neighborhood_order, algorithm = knn_algorithm).fit(recon_error)
            distances, indices = neighborhood_set.kneighbors(recon_error)

            # compute distances from nth nearest neighbors (given by neighborhood_order) and sort them
            dk_sorted     = np.sort(distances[:,-1])
            dk_sorted_ind = np.argsort(distances[:,-1])

            knee = KneeLocator(np.arange(len(distances)), dk_sorted, S = s_knee, curve = 'convex', direction = 'increasing', interp_method = 'interp1d', online = True)
            knee_x = knee.knee
            knee_y = knee.knee_y    # OR: distances[knee.knee]

            # Get outliers
            n_outliers = recon_error.shape[0] - knee_x
            idx_outliers = dk_sorted_ind[-(n_outliers + 1) : -1]
            idx_outliers_list.append(idx_outliers)
            n_outliers_list.append(n_outliers)

            # Get average error outliers
            average_error_outliers = recon_error[idx_outliers].mean()
            average_error_outliers_list.append(average_error_outliers)
            
            # Get the average error per subject
            average_error_subject = recon_error.mean()
            average_error_subject_list.append(average_error_subject)
            std_error_subject = recon_error.std(1).mean()
            std_error_subject_list.append(std_error_subject)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #%% Plot n. outliers vs epoch
        line_1 = ax_error_outliers.plot(epoch_list, average_error_outliers_list, label = "reconstruction error outliers", 
                                        color = plot_config['color_1'], linestyle = 'dashed', marker = 'o')
        
        line_2 = ax_error_outliers.errorbar(epoch_list, average_error_subject_list, yerr = std_error_subject_list, 
                    label = "reconstruction error subject",
                    color = plot_config['color_2'], linestyle = 'dashed', marker = 'X')
    
        ax_error_outliers.set_xlabel("Epoch")
        ax_error_outliers.set_ylabel("Reconstruction error", color = plot_config['color_1'])
        # ax_error_outliers.set_title("Average Error Outliers - neighborhood order {}".format(neighborhood_order))
 
        line_3 = ax_n_outliers.plot(epoch_list, n_outliers_list, label = "n. of outliers", 
                                    color = plot_config['color_3'], linestyle = 'solid', marker = '^')
        ax_n_outliers.set_ylabel("N. of outliers", color = plot_config['color_3'])
        ax_n_outliers.set_xlabel("Epoch")
        
        # Grid
        ax_n_outliers.grid(True, axis = 'both', linestyle = 'solid')
        ax_error_outliers.grid(True, which = 'both', axis = 'both', linestyle = 'dashed')
        
        # Add legend
        lns = [line_1[0], line_2, line_3[0]]
        labs = [l.get_label() for l in lns]
        ax_error_outliers.legend(lns, labs, loc=0)
    
        fig.tight_layout()
        fig.show()

        if plot_config['save_fig']:
            path_save = "Saved Results/repetition_hvEEGNet_{}/recon_error_outliers/neighborhood_order_{}/".format(tot_epoch_training, neighborhood_order)
            os.makedirs(path_save, exist_ok = True)
            path_save += "outliers_{}_neighborhood_order_{}_subj_{}".format(norm_string, neighborhood_order, subj)
            fig.savefig(path_save + ".png", format = 'png')
            fig.savefig(path_save + ".pdf", format = 'pdf')
