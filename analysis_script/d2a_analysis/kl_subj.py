"""
Compute the KL between the distribution of the different subjects. 
The distribution are obtained trough the hist function.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import mne
import numpy as np
from scipy.special import kl_div

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Settings

# Dataset for first distribution
dataset_to_use_source = 'train'
# dataset_to_use_source = 'test'

# Dataset for second distribution
# dataset_to_use_target = 'train'
dataset_to_use_target = 'test'

factor_to_average = None
factor_to_average = 'channel'
# factor_to_average = 'time'

normalize_hist = True

n_bins = 50

plot_config = dict(
    figsize = (30, 24),
    linewidth = 2,
    fontsize = 24,
    save_fig = True,
)

epsilon = 0.00001

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

mne.set_log_level(verbose = 0)

def get_data(subj, dataset_to_use):
    # Get subject data
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')

    if dataset_to_use == 'train':
        data = train_dataset.data.squeeze()
        labels = train_dataset.labels
    elif dataset_to_use == 'test':
        data = test_dataset.data.squeeze()
        labels = test_dataset.labels
    else:
        raise ValueError("dataset_to_use must have value train or test")

    if factor_to_average is None:
        data = data.flatten().sort()[0]
        average_method_string = "no_average"
    elif factor_to_average == 'channel': # Perform the average accross
        data = data.mean(1).flatten().sort()[0]
        average_method_string = "average_channel"
    elif factor_to_average == 'time':
        data = data.mean(2).flatten().sort()[0]
        average_method_string = "average_time"
    else:
        raise ValueError("factor_to_average must have value None or channel or time")

    return data, labels, average_method_string

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list_source = [1, 2, 3, 4, 5, 6, 7, 8, 9]
subj_list_target = [1, 2, 3, 4, 5, 6, 7, 8, 9]

label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}

# plt.rcParams.update({'font.size': plot_config['fontsize']})
# fig, axs = plt.subplots(2, 2, figsize = plot_config['figsize'])

kl_matrix = np.zeros((len(subj_list_source), len(subj_list_target)))

for i in range(len(subj_list_source)):
    subj_source = subj_list_source[i]
    print("Subj SOURCE {}".format(subj_source))
    
    data_source, label_source, average_method_string = get_data(subj_source, dataset_to_use_source)
    pdf_source, _ = np.histogram(data_source, bins = n_bins, density = True)

    for j in range(len(subj_list_target)):
        subj_target = subj_list_target[j]
        print("Subj TARGET {}".format(subj_target))

        data_target, label_target, average_method_string = get_data(subj_target, dataset_to_use_target)
        pdf_target, _ = np.histogram(data_target, bins = n_bins, density = True)

        kl_values = kl_div(pdf_source + epsilon, pdf_target  + epsilon)

        kl_matrix[i, j] = np.sum(kl_values)
        # kl_matrix[i, j] = np.sum(np.where(pdf_source != 0, pdf_source * np.log(pdf_source / pdf_target), 0))
