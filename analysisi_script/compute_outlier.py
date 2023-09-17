#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np 
from sklearn.neighbors import NearestNeighbors as knn

from library.analysis import support
from library.config import config_dataset as cd 

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj = 2
neighborhood_order = 5

knn_algorithm = 'ball_tree'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

neighborhood_set   = knn(n_neighbors = neighborhood_order, algorithm = knn_algorithm).fit(recon_error)
distances, indices = neighborhood_set.kneighbors(recon_error)
