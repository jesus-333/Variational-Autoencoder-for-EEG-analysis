# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Imports

from ..config import config_dataset as cd
from ..config import config_model as cm
from ..dataset import preprocess as pp
from ..training import train_generic

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def get_dataset_and_model(subj_list):
    dataset_config = cd.get_moabb_dataset_config(subj_list)

    C = 22
    if dataset_config['resample_data']: sf = dataset_config['resample_freq']
    else: sf = 250
    T = int((dataset_config['trial_end'] - dataset_config['trial_start']) * sf )
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

    # Create model (hvEEGNet)
    model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)
    model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
    model_config['use_classifier'] = False
    model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)

    return train_dataset, validation_dataset, test_dataset , model_hv
