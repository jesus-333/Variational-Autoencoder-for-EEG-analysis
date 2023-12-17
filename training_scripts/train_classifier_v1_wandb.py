# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports 

from library.training import wandb_training as wt

from library.config import config_model    as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_data = 2
subj_weights = 1

repetition = 1
epoch_trained = 80

path_weights_hvEEGNet = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj_weights, repetition, epoch_trained)

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
train_config['project_name'] = 'hvEEGNet_classifier'
train_config['model_artifact_name'] = 'classifier'

train_config['extra_info_training'] = dict(
    subj_data = subj_data,
    subj_weights = subj_weights,
    repetition = repetition,
    epoch_trained = epoch_trained,
    path_weights_hvEEGNet = path_weights_hvEEGNet
)

train_config['epochs'] = 3 
train_config['device'] = 'cpu'

model_name = 'classifier_v1'
model = wt.train_wandb(model_name, dataset_config, train_config, model_config)
