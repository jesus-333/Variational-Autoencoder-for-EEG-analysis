import numpy as np
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_avg_and_std_per_trials(data) :
    avg_list = np.zeros(data.shape[0])
    std_list = np.zeros(data.shape[0])
    for i in range(data.shape[0]) :
        avg_list[i] = np.mean(data[i])
        std_list[i] = np.std(data[i])

    return avg_list, std_list


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

filename = 'NOTCH_train8'
filename = 'NO_NOTCH_suffle6'

ch = None

plot_config = dict(
    figsize = (12, 6),
    fontsize = 17,
    use_errorbar = True,
    color = 'black',
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Load data
data_raw_train = np.load('data/TUAR/{}.npz'.format(filename))['train_data']
data_raw_test = np.load('data/TUAR/{}.npz'.format(filename))['test_data']

# (OPTIONAL) Select channel
if ch is not None :
    data_raw_train = data_raw_train[:, :, ch]
    data_raw_test = data_raw_test[:, :, ch]

avg_list_train, std_list_train = compute_avg_and_std_per_trials(data_raw_train)
avg_list_test, std_list_test = compute_avg_and_std_per_trials(data_raw_test)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot data

fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])

if plot_config['use_errorbar'] :
    axs[0].errorbar(np.arange(len(avg_list_train)), avg_list_train, yerr=std_list_train, fmt='o', color = plot_config['color'])
    axs[1].errorbar(np.arange(len(avg_list_test)), avg_list_test, yerr=std_list_test, fmt='o', color = plot_config['color'])
else :
    axs[0].plot(avg_list_train, color = plot_config['color'])
    axs[1].plot(avg_list_test, color = plot_config['color'])
    axs[0].fill_between(np.arange(len(avg_list_train)), avg_list_train - std_list_train, avg_list_train + std_list_train, 
                        alpha=0.5, color = plot_config['color'])
    axs[1].fill_between(np.arange(len(avg_list_test)), avg_list_test - std_list_test, avg_list_test + std_list_test, 
                        alpha=0.5, color = plot_config['color'])

axs[0].set_title('Train')
axs[1].set_title('Test')

for ax in axs:
    ax.set_xlabel('Trial')
    ax.set_ylabel('Average value')

fig.suptitle(filename)
fig.tight_layout()
fig.show()
