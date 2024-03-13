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
use_test_data = False

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
        idx_ch_source = ch_list[j] == ch_list
        source_data = data[:, idx_ch_source, :].flatten()
        print("\t{} - {} SOURCE channel".format(j, ch_list[j]))
        
        for k in range(len(ch_list)):
            if j > k : continue
            
            idx_ch_target = ch_list[k] == ch_list
            target_data = data[:, idx_ch_target, :].flatten()
            print("\t\t{} - {} TARGET channel".format(k, ch_list[k]))
    
            # Do ttest and save results
            t_test_output = ttest_ind(source_data, target_data, equal_var = False)
            # t_test_output = ttest_rel(train_data, test_data)
            t_statistics, p_value, df = t_test_output.statistic, t_test_output.pvalue, t_test_output.df

            p_value_list[i, j, k] = p_value
