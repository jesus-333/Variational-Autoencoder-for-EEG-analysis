"""
Fit a specified list of distributions and save the results of the fit
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import toml
from scipy import stats
import fitter
import numpy as np
import matplotlib.pyplot as plt

from library.dataset import download

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

distributions_list = ["genhyperbolic", "burr12", "burr", "norminvgauss", "mielke", "johnsonsu", "fisk", "lognorm", "skewnorm", "beta", "gamma", "norm"]
path_dataset_config = 'training_scripts/config/Ofner2017/dataset.toml'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def merge_two_dicts(x, y):
    z = x.copy()   # start with keys and values of x
    z.update(y)    # modifies z with keys and values of y
    return z

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the list of all distributions
subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
subj_list = [1, 2]

# Variables to store the distributions error for each subject
list_of_distributions_fit_error_train = []
list_of_distributions_fit_error_test = []

# Variables to store the distributions parameters for each subject
list_of_distributions_fit_parameters_train = []
list_of_distributions_fit_parameters_test  = []

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]

    # Get dataset
    dataset_config = toml.load(path_dataset_config)
    dataset_config['subjects_list'] = [subj]
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_data, labels_train, ch_list = download.get_Ofner2017(dataset_config, 'train')
    test_data, test_labels, ch_list = download.get_Ofner2017(dataset_config, 'test')
    print("Subject : ", subj)

    # Create dict to save distribution parameters
    list_of_distributions_fit_parameters_train.append(dict())
    list_of_distributions_fit_parameters_test.append(dict())
    list_of_distributions_fit_error_train.append(dict())
    list_of_distributions_fit_error_test.append(dict())

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the std for each channel

    std_ch_train = train_data.std(2).flatten()
    std_ch_test = test_data.std(2).flatten()

    for j in range(len(distributions_list)) :
        distribution = distributions_list[j]
        print("\tDistribution :", distribution)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Fit std train data
        fitter_object_train = fitter.Fitter(std_ch_train, distributions = [distribution])
        fitter_object_train.fit()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Fit std test data
        fitter_object_test = fitter.Fitter(std_ch_test, distributions = [distribution])
        fitter_object_test.fit()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Save error
        list_of_distributions_fit_error_train[-1] = fitter_object_train.df_errors['sumsquare_error'][0]
        list_of_distributions_fit_error_test[-1]  = fitter_object_test.df_errors['sumsquare_error'][0]

        # Save parameters
        list_of_distributions_fit_parameters_train[-1] = merge_two_dicts(list_of_distributions_fit_parameters_train[-1], fitter_object_train.get_best())
        list_of_distributions_fit_parameters_test[-1]  = merge_two_dicts(list_of_distributions_fit_parameters_test[-1], fitter_object_test.get_best())








