"""
Compute the outliers trials based on the reconstruction error matrix and create a knee plot
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
subj_list = [2, 9]
epoch_list = [10, 20, 40, 60, 80] 

normalize_recon_error = True
neighborhood_order_list = [5, 15] 
knn_algorithm = 'auto'
s_knee = 1

plot_config = dict(
    figsize = (12, 8),
    fontsize = 12,
    save_fig = True,
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

if normalize_recon_error:
    norm_string = "NORMALIZED"
else:
    norm_string = "NOT_NORMALIZED"

       
for neighborhood_order in neighborhood_order_list:
    for epoch in epoch_list:
        plt.rcParams.update({'font.size': plot_config['fontsize']})
        fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
        print("\nN. outliers - neighborhood order = {} - epoch = {} ({})".format(neighborhood_order, epoch, norm_string))
        for subj in subj_list:
            path_recon_error = './Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch)
            # Load data

            # dataset_config = cd.get_moabb_dataset_config([subj])
            # dataset_config['percentage_split_train_validation'] = -1
            # train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

            # Load the reconstruction error
            recon_error = np.load(path_recon_error)

            # Apply the scaling to data
            if normalize_recon_error:
                scaler = RobustScaler()
                recon_error = scaler.fit_transform(recon_error)

            #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            # Outlier identifications (NOT NORMALIZED)

            # Compute the KNN
            neighborhood_set   = knn(n_neighbors = neighborhood_order, algorithm = knn_algorithm).fit(recon_error)
            distances, indices = neighborhood_set.kneighbors(recon_error)

            # compute distances from nth nearest neighbors (given by neighborhood_order) and sort them
            dk_sorted     = np.sort(distances[:,-1])
            dk_sorted_ind = np.argsort(distances[:,-1])

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
            #%% Knee plot

            i = np.arange(len(distances))
            knee = KneeLocator(i, dk_sorted, S = s_knee, curve = 'convex', direction = 'increasing', interp_method = 'interp1d', online = True)
            knee_x = knee.knee
            knee_y = knee.knee_y    # OR: distances[knee.knee]
            
            n_outliers = recon_error.shape[0] - knee_x
            print("\tsubj {} (S = {}): {}".format(subj, s_knee, n_outliers))

            ax.plot(dk_sorted, 'o-', label = 'Subject {} (N. outliers {})'.format(subj, n_outliers))
            ax.set_xlabel('EEG Trials', fontsize = plot_config['fontsize'])
            ax.set_ylabel('Distances (sorted)', fontsize = plot_config['fontsize'])
            ax.axvline(x = knee_x, color = 'k', linestyle = '--')
            ax.axhline(y = knee_y, color = 'k', linestyle = '--')
            ax.plot((knee_x), (knee_y), 'o', color = 'r')
            ax.set_title("Knee plot - epoch {} - n_neighbors {} - {}".format(epoch, neighborhood_order, norm_string))
            ax.grid(True)
            ax.legend()

            fig.tight_layout()
            fig.show()

        if plot_config['save_fig']:
            path_save = "Saved Results/repetition_hvEEGNet_{}/recon_error_outliers/".format(tot_epoch_training)
            os.makedirs(path_save, exist_ok = True)
            path_save += "cluster_recon_error_{}_epoch_{}_neighborhood_order_{}".format(norm_string, epoch, neighborhood_order)
            fig.savefig(path_save)

