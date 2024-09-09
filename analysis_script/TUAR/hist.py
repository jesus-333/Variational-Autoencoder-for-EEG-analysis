import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

nperseg = None
fs = 250

plot_config = dict(
    figsize = (12, 6),
    fontsize = 17,
    bins = 1000,
    density = True,
    color = 'black',
    # ylim = [-10, 33]
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load data

data_raw_train = np.load('data/TUAR/NO_NOTCH_train8.npz')['train_data']
data_raw_test = np.load('data/TUAR/NO_NOTCH_train8.npz')['test_data']

data_notch_train = np.load('data/TUAR/NOTCH_train8.npz')['train_data']
data_notch_test = np.load('data/TUAR/NOTCH_train8.npz')['test_data']

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute histograms

fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

axs[0].hist(data_raw_train.flatten(), 
            bins=plot_config['bins'], density=plot_config['density'], 
            label='Train', color = plot_config['color'])
axs[1].hist(data_raw_test.flatten(), 
            bins=plot_config['bins'], density=plot_config['density'], 
            label='Train', color = plot_config['color'])

for ax in axs:
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    ax.set_yscale('log')
    ax.legend()

fig.tight_layout()
fig.show()
