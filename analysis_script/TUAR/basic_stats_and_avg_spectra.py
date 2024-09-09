import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import matplotlib.pyplot as plt

from library.analysis import support


def plot_average_spectra(avg_spectra_train, std_spectra_train, avg_spectra_test, std_spectra_test, f, plot_config : dict):
    fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

    axs[0].plot(f, avg_spectra_train, label='Train', color='black')
    axs[0].fill_between(f, avg_spectra_train - std_spectra_train, avg_spectra_train + std_spectra_train, color='black', alpha=0.2)
    axs[0].set_title('Train data', fontsize = plot_config['fontsize'])

    axs[1].plot(f, avg_spectra_test, label='Test', color='black')
    axs[1].fill_between(f, avg_spectra_test - std_spectra_test, avg_spectra_test + std_spectra_test, color='black', alpha=0.2)
    axs[1].set_title('Test data', fontsize = plot_config['fontsize'])

    for ax in axs:
        ax.set_xlabel('Frequency [Hz]', fontsize = plot_config['fontsize'])
        ax.set_ylabel(r"PSD [$\mu V^2/Hz$]", fontsize = plot_config['fontsize'])
        if 'ylim' in plot_config: ax.set_ylim(plot_config['ylim'])
        if 'xlim' in plot_config: ax.set_xlim(plot_config['xlim'])
        ax.legend()
    
    if 'title' in plot_config: fig.suptitle(plot_config['title'], fontsize = plot_config['fontsize'])
    fig.tight_layout()
    fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

idx_ch = 13

nperseg = None
fs = 250

plot_config = dict(
    figsize = (12, 6),
    fontsize = 17,
    # ylim = [-10, 33]
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data

data_raw_train = np.load('data/TUAR/NO_NOTCH_train8.npz')['train_data']
data_raw_test = np.load('data/TUAR/NO_NOTCH_train8.npz')['test_data']

data_notch_train = np.load('data/TUAR/NOTCH_train8.npz')['train_data']
data_notch_test = np.load('data/TUAR/NOTCH_train8.npz')['test_data']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Print basic information

# Check shape
print("Shape of data")
print('data_raw_train   = {}'.format(data_raw_train.shape))
print('data_raw_test    = {}'.format(data_raw_test.shape))

print('data_notch_train = {}'.format(data_notch_train.shape))
print('data_notch_test  = {}'.format(data_notch_test.shape))

# Check meand and std
print('\nMean and std')
print('data_raw_train   = {:.5f} ± {:.5f}'.format(data_raw_train.mean(), data_raw_train.std()))
print('data_raw_test    = {:.5f} ± {:.5f}'.format(data_raw_test.mean(), data_raw_test.std()))

print('data_notch_train = {:.5f} ± {:.5f}'.format(data_notch_train.mean(), data_notch_train.std()))
print('data_notch_test  = {:.5f} ± {:.5f}'.format(data_notch_test.mean(), data_notch_test.std()))

# Min and max value
print('\nMin and max')
print('data_raw_train   = {:.5f} ~ {:.5f}'.format(data_raw_train.min(), data_raw_train.max()))
print('data_raw_test    = {:.5f} ~ {:.5f}'.format(data_raw_test.min(), data_raw_test.max()))

print('data_notch_train = {:.5f} ~ {:.5f}'.format(data_notch_train.min(), data_notch_train.max()))
print('data_notch_test  = {:.5f} ~ {:.5f}'.format(data_notch_test.min(), data_notch_test.max()))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute and plot average spectra (raw data)

# Compute average spectra
avg_spectra_raw_train, std_spectra_raw_train, f = support.compute_average_spectra(data_raw_train, nperseg, fs, np.arange(data_raw_train.shape[2]) == idx_ch)
avg_spectra_raw_test, std_spectra_raw_test, f = support.compute_average_spectra(data_raw_test, nperseg, fs, np.arange(data_raw_test.shape[2]) == idx_ch)

# Plot average spectra
plot_config['title'] = 'Average spectra (raw data)'
plot_average_spectra(avg_spectra_raw_train, std_spectra_raw_train, avg_spectra_raw_test, std_spectra_raw_test, f, plot_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute and plot average spectra (notch data)

# Compute average spectra
avg_spectra_notch_train, std_spectra_notch_train, f = support.compute_average_spectra(data_notch_train, nperseg, fs, np.arange(data_notch_train.shape[2]) == idx_ch)
avg_spectra_notch_test, std_spectra_notch_test, f = support.compute_average_spectra(data_notch_test, nperseg, fs, np.arange(data_notch_test.shape[2]) == idx_ch)

# Plot average spectra
plot_config['title'] = 'Average spectra (notch data)'
plot_average_spectra(avg_spectra_notch_train, std_spectra_notch_train, avg_spectra_notch_test, std_spectra_notch_test, f, plot_config)

