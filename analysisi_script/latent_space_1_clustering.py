"""
Script to perform clustering on the latent space.
Used on the hierarchical vEEGNet
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import torch
from sklearn.cluster import KMeans

from library.analysis import support
from library.analysis import latent_space

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Config 

if len(sys.argv) > 1:
    tot_epoch_training = sys.argv[1]
    subj = sys.argv[2]
    repetition = sys.argv[3]
    epoch = sys.argv[4]
else:
    tot_epoch_training = 80
    subj = 2
    repetition = np.random.randint(19)
    epoch = 40

use_test_set = False

config_latent_space_computation = dict(
    sample_from_distribution = False,
    compute_recon_error = False,
    reduce_dimension = False,
    # - - - - - - - - - - 
    # tsne config (preconfigured parameter for tsne)
    perplexity = 30,
    n_iter = 1000,
    # - - - - - - - - - - 
)

config_cluster_kmeans = dict(
    n_clusters = 4,
    init = 'random', # Options are random or k-means++
    n_init = 'auto',
    max_iter = 300,
)
#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

device = "cuda" if torch.cuda.is_available() else "cpu",  # device (i.e. cpu/gpu) used to train the network. 

# Get datasets and model (untrained)
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model([subj])

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset

# Load model weight
path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))

# Compute latent space representation (deepest latent space)
z = latent_space.compute_latent_space_dataset(model_hv, dataset, config_latent_space_computation)

# Define k_meanse
kmeans = KMeans(n_clusters = 4, random_state = 0, n_init = "auto")
