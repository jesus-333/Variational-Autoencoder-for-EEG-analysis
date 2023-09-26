"""
Similar to compute_outliers_2.py but show the average error of the outliers accross the epochs
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
subj_list = [1, 2, 3, 4, 6, 9]
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

use_test_set = False
save_outliers = True

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
    plt.rcParams.update({'font.size': plot_config['fontsize']})
    fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
    for subj in subj_list:
        print(subj)
        idx_outliers_list = []
        average_error_list = []

        for epoch in epoch_list:
            # Load the reconstruction error
            path_recon_error = './Saved Results/repetition_hvEEGNet_{}/subj {}/recon_error_{}_average.npy'.format(tot_epoch_training, subj, epoch)
            recon_error = np.load(path_recon_error)

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

            # Get average error outliers
            average_error = recon_error[idx_outliers].mean()
            average_error_list.append(average_error)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #%% Plot n. outliers vs epoch
        ax.plot(epoch_list, average_error_list, label = "Subject {}".format(subj))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Average Error")
    ax.set_title("Average Error Outliers - neighborhood order {}".format(neighborhood_order))
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.show()

    if plot_config['save_fig']:
        path_save = "Saved Results/repetition_hvEEGNet_{}/recon_error_outliers/".format(tot_epoch_training)
        os.makedirs(path_save, exist_ok = True)
        path_save += "average_error_outliers_{}_neighborhood_order_{}".format(norm_string, neighborhood_order)
        fig.savefig(path_save)
