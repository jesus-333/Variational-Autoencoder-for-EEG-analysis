
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports the library module

from library.config import config_model as cm
from library.model import hvEEGNet

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Create config dictionary

# Parameter to specify
type_decoder = 0
parameters_map_type = 0
C = 22
T = 1000

# Get model config
model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Create the model
model = hvEEGNet.hvEEGNet_shallow(model_config)
