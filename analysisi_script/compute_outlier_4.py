"""
Similar to compute_outliers_2.py but create the plot of the trials considered as outliers
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

from library.analysis import support, visualize
from library.config import config_dataset as cd 

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

tot_epoch_training = 80
subj_list = [2, 9]
# epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
epoch_list = [10, 25, 40, 60]

use_test_set = False

normalize_recon_error = True
neighborhood_order_list = [15] 
knn_algorithm = 'auto'
s_knee = 1

plot_config = dict(
    figsize = (20, 8),
    fontsize = 12,
    save_fig = True,
    fs = 250,
    nperseg = 500,
)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

plt.rcParams.update({'font.size': plot_config['fontsize']})

t = np.linspace(2, 6, 1000)

if normalize_recon_error:
    norm_string = "NORMALIZED"
else:
    norm_string = "NOT_NORMALIZED"

outliers_error_list = dict()
outliers_idx_list = dict()
for neighborhood_order in neighborhood_order_list:
    outliers_error_list[neighborhood_order] = dict()
    outliers_idx_list[neighborhood_order] = dict()
    
    for subj in subj_list:    
        outliers_error_list[neighborhood_order][subj] = dict()
        outliers_idx_list[neighborhood_order][subj] = []
        
        dataset_config = cd.get_moabb_dataset_config([subj])
        dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
        train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

        # Decide if use the train or the test dataset
        if use_test_set: 
            dataset = test_dataset
            dataset_string = 'Test dataset'
        else: 
            dataset = train_dataset
            dataset_string = 'Train dataset'
        
        ch_list = dataset.ch_list
            
        for epoch in epoch_list:
            print("Neighborhood {} - subj {} - epoch {}".format(neighborhood_order, subj, epoch))
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
            
            # Save error of the outliers and the indices
            outliers_error_list[neighborhood_order][subj][epoch] = recon_error[idx_outliers] 
            outliers_idx_list[neighborhood_order][subj] += list(idx_outliers)
            
        idx_outliers = list(set(outliers_idx_list[neighborhood_order][subj]))
        for i in range(len(idx_outliers)):
            eeg_trial = dataset[idx_outliers[i]][0].squeeze()
            for j in range(eeg_trial.shape[0]):
                eeg_ch = eeg_trial[j]
                
                plot_config['title'] = "Trial {} ({}) -  Channel {}".format(idx_outliers[i] + 1, dataset_string, ch_list[j])
                fig, ax = visualize.plot_single_eeg_channel_both_time_and_frequency(eeg_ch, t, plot_config)
    
                if plot_config['save_fig']:
                    path_save = "Saved Results/repetition_hvEEGNet_{}/recon_error_outliers/EEG_trials_subj_{}/neighborhood_{}/trial_{}/".format(tot_epoch_training, subj, neighborhood_order, idx_outliers[i] + 1)
                    os.makedirs(path_save, exist_ok = True)
                    
                    path_save += "subj_{}_trial_{}_ch_{}".format(subj, idx_outliers[i] + 1, ch_list[j])
                    fig.savefig(path_save)
                plt.close(fig)