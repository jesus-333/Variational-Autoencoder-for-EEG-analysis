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
    epoch = sys.argv[4]
else:
    tot_epoch_training = 20
    subj = 2
    epoch = 20

if tot_epoch_training == 20:
    max_repetition_training = 8
elif max_repetition_training == 80:
    max_repetition_training = 19

use_test_set = False
repeat_clustering = 1000

config_latent_space_computation = dict(
    sample_from_distribution = False,
    compute_recon_error = False,
    reduce_dimension = False,
    batch_size = 64,
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

def count_percentage_matrix(matrix, low_value, high_value):
    tot_element = np.sum(np.logical_and(count_same_cluster >= low_value, count_same_cluster < high_value))
    
    return tot_element / len(matrix.flatten())


#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

device = "cuda" if torch.cuda.is_available() else "cpu",  # device (i.e. cpu/gpu) used to train the network. 

# Get datasets and model (untrained)
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model([subj])
print("Model and dataset created")

# Decide if use the train or the test dataset
if use_test_set: dataset = test_dataset
else: dataset = train_dataset


# Compute latent space representation (deepest latent space)
z = latent_space.compute_latent_space_dataset(model_hv, dataset, config_latent_space_computation)
print("Latent space representation computed")

# Matrix to save the number of times samples were in the same cluster
count_same_cluster = np.zeros((len(dataset), len(dataset)))

idx_step = 0
step_print = [0.2, 0.4, 0.6, 0.8, 1, 1.01]

for i in range(repeat_clustering):
    if ((i + 1) / repeat_clustering >= step_print[idx_step]): 
        print("Complete the {}% of the clusterings".format(step_print[idx_step] * 100 ))
        idx_step += 1
    
    # Load model weight
    repetition = np.random.randint(max_repetition_training) + 1
    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj, repetition, epoch)
    model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
    
    # Define k_meanse
    kmeans = KMeans(n_clusters = config_cluster_kmeans['n_clusters'], 
                    n_init = config_cluster_kmeans['n_init'], max_iter = config_cluster_kmeans['max_iter'])

    # Compute the id of each sample
    z_cluster_id = kmeans.fit_predict(z)

    for j in range(z_cluster_id.shape[0]):
        current_sample_id = z_cluster_id[j]
        for k in range(z_cluster_id.shape[0]):
            if j != k: # For different sample
                if current_sample_id == z_cluster_id[k]: # Check if they are in the same cluster
                    count_same_cluster[j, k] += 1


count_same_cluster /= repeat_clustering

print("0-20%   : ", round(count_percentage_matrix(count_same_cluster, 0, 0.2) * 100, 2))
print("20-40%  : ", round(count_percentage_matrix(count_same_cluster, 0.2, 0.4) * 100, 2))
print("40-60%  : ", round(count_percentage_matrix(count_same_cluster, 0.4, 0.6) * 100, 2))
print("60-80%  : ", round(count_percentage_matrix(count_same_cluster, 0.6, 0.8) * 100, 2))
print("80-100% : ", round(count_percentage_matrix(count_same_cluster, 0.8, 1.01) * 100, 2))