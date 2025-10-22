"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Create a ChWi autoencoder.
The config of the models must be saved in a json file and the path can be passed as argument of the script.def
Otherwise the default path will be "json_config/ChWi_autoencoder_v1.json"
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import sys

import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from library.model import ChWi
from library.dataset import dataset_time as ds_time, preprocess as pp

from library.config import config_dataset as cd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create model

# Get json path
if len(sys.argv) > 1:
    model_config_path = sys.argv[1]
    model_weights_path = sys.argv[2] if len(sys.argv) == 3 else None
else :
    model_config_path = "json_config/ChWi_autoencoder_v1.json"
    model_weights_path = "./Saved Model/test_ChWi/model_40.pth"

# Get model config
with open(model_config_path, 'r') as j:
    model_config = json.loads(j.read())

# Create autoencoder
model = ChWi.ChWi_autoencoder(model_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Test it with fake signals

# Matrix with 33 eeg signal of length 1000. The dimension of size 1 is the depth dimension for the 1d conv
fake_eeg = torch.rand(33, 1, 1000)

# Forward pass
fake_eeg_r, z, mu, sigma = model(fake_eeg)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if model_weights_path is not None :
    # Load model weights
    model.load_state_dict(torch.load(model_weights_path, map_location = torch.device('cpu')))
    model.eval()

    # Get original dataset
    subj_list = [3]
    dataset_config = cd.get_moabb_dataset_config(subj_list)
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, _, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Get tensor with data
    train_data = train_dataset.data.squeeze()
    test_data = test_dataset.data.squeeze()

    # Create train and test ChWi dataset
    train_dataset = ds_time.EEG_Dataset_ChWi(train_data)
    test_dataset = ds_time.EEG_Dataset_ChWi(test_data)

    # Sample randomly from train test
    idx = np.random.randint(0, len(train_dataset))
    x = train_dataset[idx][0].unsqueeze(0)

    # Get reconstructed data and time axis array
    x_r, _, _, _ = model(x)
    t = np.linspace(2, 6, x.shape[-1])

    # Remove extra dimension 
    x = x.squeeze().detach()
    x_r = x_r.squeeze().detach()

    # Plot original and reconstructed signal
    fig, ax = plt.subplots(figsize = (12, 10))

    ax.plot(t, x, label = 'Original signal', color = 'grey')
    ax.plot(t, x_r, label = 'Reconstructed signal', color = 'black')

    ax.legend()
    ax.set_xlim([2, 4])
    ax.grid(True)

    fig.tight_layout()
    fig.show()

