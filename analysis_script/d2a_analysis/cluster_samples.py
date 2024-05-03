"""
Cluster the d2a data, subject by subject through the kmeans
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [3, 6]
use_test_set = False

split_channels = False # If True each channel of each trials is considered as an input for the kmeans. Otherwise an input is all the trial

# Possible way to reduce the number of samples for each trial. Note that a trial is a matrix of 22 x 1000
reduce_size_method = 1 # Use min and max
# reduce_size_method = 2 # Use the average between 2s and 3.25s and the average between 3.25s and 6s

n_cluster = 9
random_state = None

plot_config = dict(
    figsize = (24, 8),
    fontsize = '18',
    markersize = 4,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Get the data

# Get the artifacts map
artifacts_map_train = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_test.csv').T.to_numpy()[1:, :] # N.b. the first line contains simply the index of the samples. The other lines containts the map of the artifacts
artifacts_map_test = pd.read_csv('Saved Results/d2a_analysis/d2a_artifacts_list_train.csv').T.to_numpy()[1:, :]

data = None # Save the data of the trials
class_labels = None # Indicate the class of the trials 
subj_id = None  # Indicate the subject of the trials

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subj {}".format(subj))

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # select train/test data
    if use_test_set:
        dataset = test_dataset
        artifacts_map = artifacts_map_test
        dataset_string = 'test'
    else:
        dataset = train_dataset
        artifacts_map = artifacts_map_train
        dataset_string = 'train'
    
    # Get data and label for the subj
    tmp_subj_data = dataset.data.squeeze().numpy()
    tmp_label = dataset.labels.numpy()
    
    # Crate a single array with the data of all subject
    if data is None and class_labels is None:
        data = tmp_subj_data.copy()
        class_labels = tmp_label.copy()
        subj_id = np.ones(tmp_subj_data.shape[0]) * subj
    else:
        data = np.concatenate((data, tmp_subj_data), 0)
        class_labels = np.concatenate((class_labels, tmp_label), 0)
        subj_id = np.concatenate((subj_id, np.ones(tmp_subj_data.shape[0]) * subj), 0)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Reduce the size of each trial and apply kmeans

reduced_data = np.zeros([data.shape[0], 2])

if reduce_size_method == 1: # Use the min and the max to describe the trials
    reduced_data[:, 0] = data.max(1).max(1)
    reduced_data[:, 1] = data.min(1).min(1)
    x_label = 'Max value'
    y_label = 'Min value'

elif reduce_size_method == 2: # Use the average between 2s and 3.25s and between 3.25s and 6s
    average_cue = data[:, :, 0:int(1.25 * 250)].mean(1).mean(1)
    average_task = data[:, :, int(1.25 * 250):].mean(1).mean(1)
    reduced_data[:, 0] = average_cue
    reduced_data[:, 1] = average_task

    x_label = 'Cue Average'
    y_label = 'Task Average'

else:
    raise ValueError("reduce_size_method must have value 1 or 2")

kmeans = KMeans(n_clusters = 9, random_state = random_state)
predict_cluster = kmeans.fit_predict(reduced_data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Plot data

plt.rcParams.update({'font.size': plot_config['fontsize']})

fig, axs = plt.subplots(1, 3, figsize = plot_config['figsize'])

for el in set(predict_cluster):
    print(el)
    idx_cluster = predict_cluster == el
    axs[0].scatter(reduced_data[idx_cluster, 0], reduced_data[idx_cluster, 1],
                   s = plot_config['markersize'], label = el)

for el in set(subj_id):
    print(el)
    idx_subj = subj_id == el
    axs[1].scatter(reduced_data[idx_subj, 0], reduced_data[idx_subj, 1],
                   s = plot_config['markersize'], label = el)

for el in set(class_labels):
    print(el)
    idx_class = class_labels == el
    axs[2].scatter(reduced_data[idx_class, 0], reduced_data[idx_class, 1],
                   s = plot_config['markersize'], label = el)

axs[0].set_title("Prediceted cluster")
axs[1].set_title("Subject")
axs[2].set_title("Class")

for ax in axs:
    ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if reduce_size_method == 2:
        ax.set_xlim([-10, 10])
        ax.set_ylim([-3, 3])

fig.tight_layout()
fig.show()
