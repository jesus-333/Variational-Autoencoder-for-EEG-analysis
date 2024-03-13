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
use_test_data = True

channel_to_use = None

transpose_data = True # If true the data are saved in row array instead of column array

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

p_value_list = np.zeros((len(subj_list), 22, 22))

for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subject {}".format(subj))

    # Get subject dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    ch_list = train_dataset.ch_list
    
    # Get data from dataset and flatten them
    if use_test_data :
        data = test_dataset.data.squeeze()
    else :
        data = train_dataset.data.squeeze()

    for j in range(len(ch_list)):
        ch_source = ch_list[j]
        source_data = data[:, ch_source, :].flatten()
        for k in range(len(ch_list)):
            # if j == k : continue

            ch_target = ch_list[k]
            target_data = data[:, ch_target, :].flatten()
    
            # Do ttest and save results
            t_test_output = ttest_ind(ch_source, ch_target, equal_var = True)
            # t_test_output = ttest_rel(train_data, test_data)
            t_statistics, p_value, df = t_test_output.statistic, t_test_output.pvalue, t_test_output.df

            p_value_list[i, j, k] = p_value
