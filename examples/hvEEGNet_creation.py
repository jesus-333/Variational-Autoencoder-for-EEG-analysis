"""
In this example you will see how to create an untrained hvEEGNet model.

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports the library modules

from library.config import config_model as cm
from library.model import hvEEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create config dictionary

# Parameter to specify
C = 22      # Number of EEG channels
T = 1000    # Number of time samples

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T)

# During the creation of hvEEGNet the size of the input data are needed to compute the size of some component of the model.
# Also C is necessary for the size of the second convolutional kernel (spatial filter)
# You can find an explanation of all the parameters used during the creation inside the config_model subpackage

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)
