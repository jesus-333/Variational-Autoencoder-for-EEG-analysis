"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Create a ChWi autoencoder.
The config of the models must be saved in a json file and the path can be passed as argument of the script.def
Otherwise the default path will be "json_config/ChWi_encoder.json"
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import json
import sys

import torch

from library.model import ChWi

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create model

# Get json path
if len(sys.argv) > 1:
    model_config_path = sys.argv[1]
else :
    model_config_path = "json_config/ChWi_encoder.json"

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
