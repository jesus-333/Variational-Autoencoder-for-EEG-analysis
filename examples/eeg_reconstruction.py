"""
In this example you will see how to reconstruct an EEG signal from dataset 2a

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
import matplotlib.pyplot as plt

from library.dataset import preprocess as pp
from library.model import hvEEGNet

from library.config import config_dataset as cd
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

subj = 3

ch_to_plot = 'C3'

# This line of code will automatically set the device as GPU whether the system has access to one
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get dataset and model

# Get dataset
dataset_config = cd.get_moabb_dataset_config([2])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get number of channels and number of time samples
C = train_dataset[0][0].shape[1]
T = train_dataset[0][0].shape[2]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)

# Load the weights
model.load_state_dict(torch.load('./examples/example_trained_weigths.pth', map_location = torch.device('cpu')))
# P.s. this line here can throw you an error, depending on how you run the code.
# You could see the error that python doesn't find the weights file. 
# This because the path to this specific weights file is defined relative to the root folder of the repository. So it is valid only if you run the script from that folder

# Note on device and weights
# By default when the weights are saved during the training they keep track of the device used by the model (i.e. CPU or GPU)
# So if you don't specify the map_location argument the torch load function expects that model and weights are in the same location.
# When a model is created in PyTorch its location is the CPU. Instead the weights are saved from the GPU (because on 99% of the the time you will train the model with GPU)
# So the torch.load() function will throw an error if it find the model on CPU and the weights that want a model on GPU.
# To avoid this, when you load the model remember to specify map_location as cpu.
# In this way everything will be loaded in the CPU.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction of a single EEG signal

# Get a random sample from the test dataset
idx_trial = int(torch.randint(0, len(test_dataset), (1, 1)))
x_eeg, label = test_dataset[idx_trial]

# Add batch dimension
x_eeg = x_eeg.unsqueeze(0)

# Note on input size.
# When you get a single signal from the dataset (as with the instruction x_eeg, label = test_dataset[idx]) the data you obtain has shape 1 x C x T
# But hvEEGNet want an input of shape B x 1 x C x T. So for a single sample I have to add the batch dimension

# EEG Reconstruction. To reconstruct an input signal you could use the function reconstruct, implemented inside hvEEGNet
x_r_eeg = model.reconstruct(x_eeg)

# Create time array (for the plot)
t = torch.linspace(dataset_config['trial_start'], dataset_config['trial_end'], T)

# Get the idx of channel to plot
idx_ch = test_dataset.ch_list == ch_to_plot

# Plot the original and reconstructed signal
plt.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(1, 1, figsize = (12, 8))

ax.plot(t, x_eeg.squeeze()[idx_ch].squeeze(), label = 'Original EEG', color = 'grey', linewidth = 2)
ax.plot(t, x_r_eeg.squeeze()[idx_ch].squeeze(), label = 'Reconstructed EEG', color = 'black', linewidth = 1)

ax.legend()
ax.set_xlim([2, 4]) # Note that the original signal is 4s long. Here I plot only 2 second to have a better visualization
ax.set_xlabel('Time [s]')
ax.set_ylabel(r"Amplitude [$\mu$V]")
ax.set_title("Subj {} (test) - Trial {} - Ch. {}".format(subj, idx_trial, ch_to_plot))
ax.grid(True)

fig.tight_layout()
fig.show()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction multple EEG signal
# Instead of reconstruct a signal EEG signal maybe you want to reconstruct tens or hundreds of signal simultaneously.
# In this case you just need to put all signal in a single tensor and use the reconstruct method.

x_eeg_multi, label_multi = test_dataset[50:200] # Get 150 eeg signals
# Note that if you use the notation above to get more than 1 eeg signal the data you get from the dataset have shape N x 1 x C x T, with N = the number of EEG signals, 150 in this case.

# Move the model and data to device
x_eeg_multi = x_eeg_multi.to(device)
model.to(device)

# Other note on device
# If you want to reconstruct a single signal the time difference between CPU and GPU is negligible.
# But if you want the reconstruction of multiple signals together the GPU is much more efficient.
# You will find some benchmarks in the last section of hvEEGNet paper

# Reconstruct multiple EEG signal
x_r_eeg_multi = model.reconstruct(x_eeg_multi)
