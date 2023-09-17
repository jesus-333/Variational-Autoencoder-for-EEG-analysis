#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors as knn

from library.analysis import support
from library.config import config_dataset as cd 

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Parameters

subj = 2
neighborhood_order = 5

knn_algorithm = 'ball_tree'

path_recon_error = ''

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

# Load the reconstruction error
recon_error = np.load(path_recon_error)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Outlier identifications

# Compute the KNN
neighborhood_set   = knn(n_neighbors = neighborhood_order, algorithm = knn_algorithm).fit(recon_error)
distances, indices = neighborhood_set.kneighbors(recon_error)

# compute distances from nth nearest neighbors (given by neighborhood_order) and sort them
dk_sorted     = np.sort(distances[:,-1])
dk_sorted_ind = np.argsort(distances[:,-1])

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Knee plot

try:
    from kneed import KneeLocator
    i = np.arange(len(distances))
    knee = KneeLocator(i, dk_sorted, S = 1, curve = 'convex', direction = 'increasing', interp_method = 'interp1d', online = True)
    knee_x = knee.knee
    knee_y = knee.knee_y    # OR: distances[knee.knee]

    print([knee_x, np.round(knee_y,2)])
    # Plot distances
    fig = plt.figure(figsize=(18,2))

    ax = fig.add_subplot(111)
    plt.plot(dk_sorted, 'o-')
    ax.set_xlabel('Data points', fontsize=10)
    ax.set_ylabel('Distances (sorted)', fontsize=10)
    plt.axvline(x = knee_x, color='k', linestyle='--')
    plt.axhline(y = knee_y, color='k', linestyle='--')
    plt.plot((knee_x), (knee_y), 'o', color='r')
    plt.grid()
    plt.show()

except ImportError as e:
    print("Error -> ", e)
    print("If you want the knee plot install the kneed package (pip install kneed)")
