# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports 

from library.training import wandb_training as wt

from library.config import config_model    as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_data = 2
subj_weight = 1

path_weights_hvEEGNet = 'Saved model/' # TODO

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

dataset_config, _, hvEEGNet_config = wt.get_config_dict_for_hvEEGNet_shallow([subj_data])
classifier_config = cm.get_config_classifier_v1()

model_config = dict(
    config_clf = classifier_config,
    config_hvEEGNet = hvEEGNet_config,
    path_weights = path_weights_hvEEGNet,
    use_only_mu_for_classification = True
)

train_config = ct.get_config_classifier()
train_config['wandb_training'] = True
train_config['model_artifact_name'] = 'classifier'

# model = wt.train_wandb('classifier_v1', dataset_config, train_config, model_config)
