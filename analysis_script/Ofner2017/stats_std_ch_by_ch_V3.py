"""
Fit all possible distributions inside scipy on all channel std values of all the subjects.
Then it rank them based on fit error.
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Import

import toml
from scipy import stats
import fitter
import numpy as np
import matplotlib.pyplot as plt

from library.dataset import download

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

path_dataset_config = 'training_scripts/config/Ofner2017/dataset.toml'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_distribution_fit_postion(fitter_object : fitter.Fitter) :
    """
    Given a fitter return a dict where each pair key/value is a distribution and and another dict with its position in the ranking and the sumsquare error of the fit.
    For ranking I mean the list of position sorted based on the sumsquare error of the fit.
    """
    distributions_list = list(fitter_object.fitted_param.keys())
    distributions_position_and_error = {}

    fit_summary = fitter_object.summary(len(distributions_list))
    distributions_sorted = list(fit_summary.index)

    for i in range(len(distributions_sorted)) :
        distributions_position_and_error[distributions_sorted[i]] = {}
        distributions_position_and_error[distributions_sorted[i]]['position'] = i + 1
        distributions_position_and_error[distributions_sorted[i]]['error'] = fit_summary.loc[distributions_sorted[i]]['sumsquare_error']

    return distributions_position_and_error

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Get the list of all distributions
distributions = []
all_distributions = [str(getattr(stats, d)) for d in dir(stats) if isinstance(getattr(stats, d), stats.rv_continuous)]
for dist in all_distributions :
    distributions.append(dist.split(' ')[0].split('.')[-1].split('_')[0])

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

# Variables to store the distributions position for each subject
list_of_distributions_position_train = {}
list_of_distributions_position_test = {}
list_of_distributions_position_both = {}

# Variables to store the distributions error for each subject
list_of_distributions_fit_error_train = {}
list_of_distributions_fit_error_test = {}
list_of_distributions_fit_error_both = {}

for i in range(len(subj_list)) :
    # Get subject
    subj = subj_list[i]
    print("Subject : ", subj)

    # Get dataset
    dataset_config = toml.load(path_dataset_config)
    dataset_config['subjects_list'] = [subj]
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_data, labels_train, ch_list = download.get_Ofner2017(dataset_config, 'train')
    test_data, test_labels, ch_list = download.get_Ofner2017(dataset_config, 'test')
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Compute the std for each channel

    std_ch_train = train_data.std(2).flatten()
    std_ch_test = test_data.std(2).flatten()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Fit std train data
    fitter_object_train = fitter.Fitter(std_ch_train, distributions = distributions)
    fitter_object_train.fit()
    distributions_position_and_error_train = get_distribution_fit_postion(fitter_object_train)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Fit std test data
    fitter_object_test = fitter.Fitter(std_ch_test, distributions = distributions)
    fitter_object_test.fit()
    distributions_position_and_error_test  = get_distribution_fit_postion(fitter_object_test)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Save positons

    fitted_distribution_train = list(distributions_position_and_error_train.keys())
    fitted_distribution_test = list(distributions_position_and_error_test.keys())
    fitted_dsitribution_both = set(fitted_distribution_train + fitted_distribution_test)
    
    for distribution in distributions :
        # Check if the distribution is already in the dict
        if distribution not in list_of_distributions_position_train : 
            list_of_distributions_position_train[distribution] = []
            list_of_distributions_fit_error_train[distribution] = []
        if distribution not in list_of_distributions_position_test : 
            list_of_distributions_position_test[distribution] = []
            list_of_distributions_fit_error_test[distribution] = []
        if distribution not in list_of_distributions_position_both : 
            list_of_distributions_position_both[distribution] = []
            list_of_distributions_fit_error_both[distribution] = []

        # Save positions ad errors (TRAIN)
        if distribution in fitted_distribution_train :
            list_of_distributions_position_train[distribution].append(distributions_position_and_error_train[distribution]['position'])
            list_of_distributions_fit_error_train[distribution].append(distributions_position_and_error_train[distribution]['error'])
    
        # Save positions ad errors (TEST)
        if distribution in fitted_distribution_test :
            list_of_distributions_position_test[distribution].append(distributions_position_and_error_test[distribution]['position'])
            list_of_distributions_fit_error_test[distribution].append(distributions_position_and_error_test[distribution]['error'])

        # Save positions ad errors (BOTH)
        if distribution in fitted_dsitribution_both :
            if distribution in fitted_distribution_train :
                list_of_distributions_position_both[distribution].append(distributions_position_and_error_train[distribution]['position'])
                list_of_distributions_fit_error_both[distribution].append(distributions_position_and_error_train[distribution]['error'])
            if distribution in fitted_distribution_test :
                list_of_distributions_position_both[distribution].append(distributions_position_and_error_test[distribution]['position'])
                list_of_distributions_fit_error_both[distribution].append(distributions_position_and_error_test[distribution]['error'])

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Close plot created by summary
    plt.close()

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute the average position for each distribution

# Variables to store the average position for each distribution
average_position_train, average_position_test, average_position_both = [], [], []
average_error_train, average_error_test, average_error_both = [], [], []

# Compute the average position and error
for distribution in distributions :
    # Position
    average_position_train.append(np.round(np.mean(list_of_distributions_position_train[distribution]), 2))
    average_position_test.append(np.round(np.mean(list_of_distributions_position_test[distribution]), 2))
    average_position_both.append(np.round(np.mean(list_of_distributions_position_both[distribution]), 2))
    
    # Error
    average_error_train.append(np.round(np.mean(list_of_distributions_fit_error_train[distribution]), 3))
    average_error_test.append(np.round(np.mean(list_of_distributions_fit_error_test[distribution]), 3))
    average_error_both.append(np.round(np.mean(list_of_distributions_fit_error_both[distribution]), 3))

# Sort the distributions based on the average position
idx_sorted_train = np.argsort(average_position_train)
idx_sorted_test = np.argsort(average_position_test)
idx_sorted_both = np.argsort(average_position_both)

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Save the results (based on both version)

str_to_save = "Distribution, Average position (train), Average position (test), Average position (both), Average error (train), Average error (test), Average error (both)\n"
for i in range(len(distributions)) :
    idx = idx_sorted_both[i]
    str_to_save += "{}, {}, {}, {}, {}, {}, {}\n".format(distributions[idx], 
                                                         average_position_train[idx], average_position_test[idx], average_position_both[idx], 
                                                         average_error_train[idx], average_error_test[idx], average_error_both[idx]
                                                         )
    
print(str_to_save)

with open('Saved Results/Ofner2017/stats_ch/distributions_fit_results.txt', 'w') as f :
    f.write(str_to_save)

with open('Saved Results/Ofner2017/stats_ch/distributions_fit_results.csv', 'w') as f :
    f.write(str_to_save)
