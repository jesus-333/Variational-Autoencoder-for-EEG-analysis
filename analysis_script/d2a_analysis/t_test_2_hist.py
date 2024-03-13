"""
Perform the t-test between the data of the various subjects.
With data of a subject I mean all the samples of all the trials. Train and test data are kept separated
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
from scipy.stats import ttest_ind, ttest_rel

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [8]

channel_to_use = None

distribution_type = 1
bins = 5000

transpose_data = True # If true the data are saved in row array instead of column array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_hist(dataset, bins, distribution_type, channel_to_use = None):
    # Flatten the data and (OPTIONAL) select a channel
    if channel_to_use is not None:
        idx_ch = dataset.ch_list == channel_to_use
        data = dataset.data.squeeze()[:, idx_ch, :].flatten()
    else:
        data = dataset.data.flatten()
        
    if distribution_type == 1 : # Continuos PDF
        p_x, bins_position = np.histogram(data.sort()[0], bins = bins, density = True)
    elif distribution_type == 2 : # Discrete PDF
        p_x, bins_position = np.histogram(data.sort()[0], bins = bins, density = False)
        p_x = p_x / len(data)
    else :
        raise ValueError("distribution_type must have value 1 (continuos PDF) or 2 (discrete PDF)")

    step_bins = bins_position[1] - bins_position[0]
    bins_position = bins_position[1:] - step_bins

    # Remove bins with 0 samples inside
    idx_not_zeros = p_x != 0
    p_x = p_x[idx_not_zeros]
    bins_position = bins_position[idx_not_zeros]

    return p_x, bins_position
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

p_value_list = np.zeros(len(subj_list))

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subject {}".format(subj))

    # Get subject dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    train_data, bins_position = compute_hist(train_dataset, bins, distribution_type, channel_to_use)
    test_data, bins_position = compute_hist(test_dataset, bins, distribution_type, channel_to_use)
    
    # Do ttest
    t_test_output = ttest_ind(train_data, test_data, equal_var = False)
    # t_test_output = ttest_rel(train_data, test_data)

    # Save results
    # t_statistics, p_value, df = t_test_output.statistic, t_test_output.pvalue, t_test_output.df
    t_statistics, p_value  = t_test_output.statistic, t_test_output.pvalue

    p_value_list[i] = p_value
