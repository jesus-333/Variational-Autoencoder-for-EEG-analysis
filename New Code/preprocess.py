"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function related to data preprocessing and visualization
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

import moabb.datasets as mb
import moabb.paradigms as mp

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Download the data

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Visualization

def config_plot():
    config = dict(
        # Figure config
        figsize = (15, 10),
        # Data config
        ch_to_plot = ['C3, CZ, C4'],
        label_to_plot = [0,1,2]
    )

    return config

def visualize_single_subject_average_channel(data, ch_list, label_list, config : dict):
    """
    Visualize

    data = numpy array with the EEG data. The shape must be N x C x T with N = number of trials, C = number of channels, T = time samples
    ch_list = list of length ch with the name of the channels
    label_list = numpy array of length N with the list of the label for each trial
    """
    extracted_data = np.extract(data, ch_list, label_list, config)

    fig, ax = plt.subplots(len(config['label_to_plot']), len(config['ch_to_plot']), figisize = config['figsize'])

    for i in range(len(config['label_to_plot'])):
        for j in range(len(config['ch_to_plot'])):
            ax[i, j].plot(extracted_data[i, j])

            ax[i, j].title.set_text("{} - {}".format(label_list[i], ch_list[j]))

    fig.tight_layout()
    fig.show()

def extract_data(data, ch_list, label_list, config):
    # List of eeg data with specific channels and average along the class
    extracted_data = []

    for label in config['label_to_plot']:
        # Get the indices of all the trial for a specific label
        idx_label = label_list == label

        # Get all the trial of the specific label and do the mean across the trial
        # The results is a matrix of shape C x T that contain the average accross the trial for a specific class
        average_per_label = data[idx_label].mean(0)
        
        # Get the channels that I want to plot
        tmp_list = []
        for ch in config['ch_to_plot']:
            idx_ch = ch_list == ch
            tmp_list.append(average_per_label[idx_ch])
        
        # Save the list of data for the channels that I want to plot for this specific label
        extracted_data.append(tmp_list)

    return extracted_data 

    
