"""
In this example you will see how to compute the reconstruction error with hvEEGNet and DTW
I will use the data from dataset 2a.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

from library.dataset import preprocess as pp
from library.model import hvEEGNet
from library.analysis import dtw_analysis
from library.training.soft_dtw_cuda import SoftDTW

from library.config import config_dataset as cd
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

subj = 3

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Get dataset and model

# Get dataset
dataset_config = cd.get_moabb_dataset_config([2])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, _, test_dataset = pp.get_dataset_d2a(dataset_config)

# Get number of channels and number of time samples
C = train_dataset[0][0].shape[1]
T = train_dataset[0][0].shape[2]

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# Create the model and load the weights
model = hvEEGNet.hvEEGNet_shallow(model_config)
model.load_state_dict(torch.load('./examples/example_trained_weigths.pth', map_location = torch.device('cpu')))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Get n random eeg signal from test dataset
n_eeg_signal = 50
idx = torch.randint(len(test_dataset), (n_eeg_signal, 1)).squeeze()
x, _ = test_dataset[idx]

# Reconstruct EEG signals
x_r = model.reconstruct(x)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute reconstruction error with function inside the library

# Inside the module dtw analysis of the library there is a function that computed the dtw between two tensor. 
# Read the function documentation to more info about the computation and the input parameters.

print("Shape of tensor x = {}".format(x.shape))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x, x_r, device, average_channels = False, average_time_samples = False)
print("Reconstruction error with average_channels = False and average_time_samples = False. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x, x_r, device, average_channels = False, average_time_samples = True)
print("Reconstruction error with average_channels = False and average_time_samples = True. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x, x_r, device, average_channels = True, average_time_samples = False)
print("Reconstruction error with average_channels = True  and average_time_samples = False. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

recon_error = dtw_analysis.compute_recon_error_between_two_tensor(x, x_r, device, average_channels = True, average_time_samples = True)
print("Reconstruction error with average_channels = True  and average_time_samples = True. Shape of the output tensor = {}\n{}\n".format(recon_error.shape, recon_error))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute reconstruction error by hand

# If you want to use directly the softDTW functino here there is a brief snippet of code that show how to use it. The code is almost identical to that of the function compute_recon_error_between_two_tensor
# The only difference is that the function also implement the average along channels and by the number of time samples.

# Note that the dtw is computed between 1d signal so for an EEG signal of C channel you have to compare the channel one by one.

# Declare SoftDTW function
use_cuda = True if device == 'cuda' else False
softDTW_function = SoftDTW(use_cuda = use_cuda, normalize = False)

# Tensor to save the reconstruction error
recon_error = torch.zeros(x.shape[0], x_r.shape[2])

# Move input tensor to device
x = x_r.to(device)
x_r = x_r.to(device)

# Compute the DTW channel by channels
for i in range(x.shape[2]): # Iterate through EEG Channels
    x_ch = x[:, :, i, :].swapaxes(1, 2)
    x_r_ch = x_r[:, :, i, :].swapaxes(1, 2)
    # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
    # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
    
    # Compute reconstruction error, and move it to cpu
    recon_error[:, i] = softDTW_function(x_ch, x_r_ch).cpu()

# Technical notes :
# The person who implemented SoftDTW consider to work with batch of 1d signal, i.e. batch of shape B x T x D. In our case D, the depth dimension has size 1 so we consider batch of shape B x T x 1
# Here you find the github repo if you want more info : https://github.com/Maghoumi/pytorch-softdtw-cuda.
# With EEG usually you work with batch of shape B x 1 x C x T (with the dimension of shape 1 corresponding to the depth dimension) so I decide to iterate along the channels dimension.
# If you write x[:, :, j, :] you obtain a tensor of shape B x 1 x T. The swapaxes(1, 2), as the name suggest, changes the order of the axis creating a new tensor of shape B x T x 1, the shape required by softDTW.
# Note that writing x[:, :, j, :] means taking all channels of index j for each element in the batch. E.g. if the channel j is C3 you will take all the C3 channels of the batch.

# If you have a single EEG signal, i.e. a tensor of shape 1 x 1 x C x T, you can compute the dtw immediately, withouth iterating along the channels.
# To do this simply remove one dimension of size 1, obtaining a tensor of shape 1 x C x T and then use swapeaxes to change the shape in C x T x 1.
# In this way is that as if you had a batch with C elements and you can compute the error in a single step.

# Eventually you can also iterate along the batch elements and use the trick explained above to compute, at each iteration, the dtw between pair of EEG signals of shape C x T.
# I prefer to iterate along the EEG channels dimension because usually I have more than elements in the batch than EEG channels.
# But if you want to iterate along the batch element I will write below a brief example.

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Example of iteration along batch

# Tensor to save the reconstruction error
recon_error_2 = torch.zeros(x.shape[0], x_r.shape[2])

for i in range(x.shape[0]): # Iterate along batch elements
    # Get the i-th element of the batch
    # In this way x_i and x_r_i have shape 1 x C x T
    x_i = x[i]
    x_r_i = x_r[i]

    # Change the order of axis
    x_i = x_i.swapaxes(0, 1)
    x_r_i = x_r_i.swapaxes(0, 1)
    
    # Compute the reconstruction error for all the channels of the i-th element of the batch
    recon_error_2[i] = softDTW_function(x_i, x_r_i).cpu()
    # Note that in this case the output of softDTW_function has shape C, i.e. an array where each element is the dtw between the a channel of x and the corresponding channel of x_r
