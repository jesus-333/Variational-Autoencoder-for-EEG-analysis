"""
In this example you will see how to download the data from dataset 2a using the library.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

from library.dataset import preprocess as pp
from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get config for the dataset

# Dataset 2a contains motor imagery for 9 subjects (indentified with number from 1 to 9). You can decide to download the data for one or for multiple subjects
subj_list = [3]       # If you want the data of a single subject create a list with a single element
# subj_list = [2, 4, 9]   # If you want the data for multiple subjects create a list with the number of all the subjects you want

# Get the dataset config. Check inside the functions for all the details about the various parameters
dataset_config = cd.get_moabb_dataset_config(subj_list)

# If you want to modify the data you can change the default settings after the creation of the dataset (or inside the get_moabb_dataset_config function)
# dataset_config['resample_data'] = True
# dataset_config['resample_freq'] = 128

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset creation

# This function automatically download the data through moabb, preprocess the data and create the dataset
# Note that the dataset is an istance of the class EEG_Dataset (you can find the code for the class inside the dataset_time subpackage)
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# If you specify multuple subjects in the subj_list the data of the various subjects are mixed together
# So for example if subj_list = [2, 4, 9] the train_dataset will contain all the train data of subjects 2, 4 and 9 while the test_dataset will contain all the test data of the three subjects

# The valdiation_dataset contain a percentage of the training data that will not used for training but only for validation.
# The split between train/validation is controlled by the variable percentage_split_train_validation inside the dataset_config dictionary.
# E.g. percentage_split_train_validation = 0.9 means that 90% of the training data will be used for training and 10% for validation.

# Take a single EEG trial
single_trial_eeg, single_trial_label = train_dataset[2]

# Take multiple EEG trial
multiple_trial_eeg, multiple_trial_labels = train_dataset[4:20]
