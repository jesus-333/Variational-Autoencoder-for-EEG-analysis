"""
Compute the parameters of the gaussian distribution that fit the data.
In this version the gaussian fit the histrogram of the data
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import kurtosis, skew
import matplotlib.pyplot as plt

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# subj_list = [8]

distribution_type = 1 # 1 means normalize automatically through matplotlib. Create a continuos PDF (i.e. the integral of the area under the histogram is equal to 1)
# distribution_type = 2 # 2 means the creation of a discrete PDF ( i.e. the hights of the bins is divided by the total number of the samples )

channel_to_use = None

bins = 400

compute_from_samples = False # If True, instead of creating the histogram all the metrics are directly computed from all the samples

plot_config = dict(
    figsize = (10, 8),
    use_same_plot = True,
    bins = 50,
    linewidth = 1.5,
    use_log_scale_x = False, # If True use log scale for x axis
    use_log_scale_y = False, # If True use log scale for y axis
    fontsize = 24,
    save_fig = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def gaussian_function(x, mean, std):
    A = 1 / (std * np.sqrt(2 * np.pi))
    B = - ((x - mean) ** 2) / (2 * (std ** 2))

    y = A * np.exp(B)
    return y

def compute_parameter_from_distribution(dataset, bins, distribution_type, channel_to_use = None):
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

    # Fit the guassian curve
    parameters, covariance = curve_fit(gaussian_function, bins_position, p_x)
    
    # print(bins_position)

    # Get parameters
    fit_mean = parameters[0]
    fit_std = parameters[1]

    # Compute kurtosis
    fit_skew = skew(p_x)

    # Compute skewness
    fit_kurtosis = kurtosis(p_x)

    return fit_mean, fit_std, fit_skew, fit_kurtosis, p_x, bins_position

def compute_parameter_from_samples(dataset, channel_to_use = None):
    if channel_to_use is not None:
        idx_ch = dataset.ch_list == channel_to_use
        data = dataset.data.squeeze()[:, idx_ch, :].flatten()
    else:
        data = dataset.data.flatten()
    
    return data.mean(), data.std(), skew(data), kurtosis(data)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Variable to save parameters (train set)
train_mean_array     = np.zeros(len(subj_list))
train_std_array      = np.zeros(len(subj_list))
train_kurtosis_array = np.zeros(len(subj_list))
train_skew_array     = np.zeros(len(subj_list))

# Variable to save parameters (test set)
test_mean_array     = np.zeros(len(subj_list))
test_std_array      = np.zeros(len(subj_list))
test_kurtosis_array = np.zeros(len(subj_list))
test_skew_array     = np.zeros(len(subj_list))

string_to_print = ""

# Iterate through subjects
for i in range(len(subj_list)):
    subj = subj_list[i]
    print("Subject {}".format(subj))

    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # Compute the various parameters
    if compute_from_samples :
        # Directly from all the samples
        train_mean, train_std, train_skew, train_kurtosis = compute_parameter_from_samples(train_dataset, channel_to_use)
        test_mean, test_std, test_skew, test_kurtosis = compute_parameter_from_samples(test_dataset, channel_to_use)
    else :
        # Gaussian fit for train and test set (i.e. compute mean and std)
        train_mean, train_std, train_skew, train_kurtosis, p_x_train, x_train = compute_parameter_from_distribution(train_dataset, bins, distribution_type, channel_to_use)
        test_mean, test_std, test_skew, test_kurtosis, p_x_test, x_test = compute_parameter_from_distribution(test_dataset, bins, distribution_type, channel_to_use)

    # Save parameters
    train_mean_array[i]     = train_mean
    train_std_array[i]      = train_std
    train_skew_array[i]     = train_skew
    train_kurtosis_array[i] = train_kurtosis
    test_mean_array[i]      = test_mean
    test_std_array[i]       = test_std
    test_skew_array[i]      = test_skew
    test_kurtosis_array[i]  = test_kurtosis

    # Create string to print all the results
    string_to_print += "Subj {}\n".format(subj)
    string_to_print += "Train dataset\n"
    string_to_print += "\tmean     = {:.3f}\n".format(train_mean)
    string_to_print += "\tstd      = {:.3f}\n".format(train_std)
    string_to_print += "\tskew     = {:.3f}\n".format(train_skew)
    string_to_print += "\tkurtosis = {:.3f}\n".format(train_kurtosis)
    string_to_print += "Test dataset\n"
    string_to_print += "\tmean     = {:.3f}\n".format(test_mean)
    string_to_print += "\tstd      = {:.3f}\n".format(test_std)
    string_to_print += "\tskew     = {:.3f}\n".format(test_skew)
    string_to_print += "\tkurtosis = {:.3f}\n".format(test_kurtosis)
    string_to_print += "\n"

print(string_to_print)
