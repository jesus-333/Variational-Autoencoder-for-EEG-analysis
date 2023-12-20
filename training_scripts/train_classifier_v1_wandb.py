# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports 

from library.training import wandb_training as wt

from library.config import config_model    as cm
from library.config import config_training as ct

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Settings

subj_data = 6
subj_weights = 3
repetition_list = [1]
epochs = 800

freeze_encoder = False
use_only_mu_for_classification = False

epoch_trained = 80
hvEEGNet_training_repetition = 1 # Since for each subject we train hvEEGNet 20 times this parameter specifies the training run

path_weights_hvEEGNet = 'Saved Model/repetition_hvEEGNet_80/subj {}/rep {}/model_{}.pth'.format(subj_weights, hvEEGNet_training_repetition, epoch_trained)
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

dataset_config, _, hvEEGNet_config = wt.get_config_dict_for_hvEEGNet_shallow([subj_data])
classifier_config = cm.get_config_classifier_v1(use_only_mu_for_classification)

model_config = dict(
    config_clf = classifier_config,
    config_hvEEGNet = hvEEGNet_config,
    path_weights = path_weights_hvEEGNet,
    freeze_encoder = freeze_encoder   # If True freeze the encoder weights
)

train_config = ct.get_config_classifier()
train_config['wandb_training'] = True
train_config['project_name'] = 'hvEEGNet_classifier'
train_config['model_artifact_name'] = 'classifier_v1'

for repetition in repetition_list:

    train_config['extra_info_training'] = dict(
        subj_data = subj_data,
        subj_weights = subj_weights,
        hvEEGNet_training_repetition = hvEEGNet_training_repetition,
        hvEEGNet_epoch_trained = epoch_trained,
        path_weights_hvEEGNet = path_weights_hvEEGNet
    )
    
    train_config['name'] = "classifier_v1_Weights_S{}_Target_S{}_rep_{}".format(subj_weights, subj_data, repetition)
    
    train_config['notes'] = ""
    train_config['notes'] += "Training of classifier v1 with the weights of S{} and the data of S{}\n".format(subj_weights, subj_data)
    train_config['notes'] += "The weights are from hvEEGNet at the end of the training with {} epochs (repetition {})".format(epoch_trained, hvEEGNet_training_repetition)
    
    model_name = 'classifier_v1'
    model = wt.train_wandb(model_name, dataset_config, train_config, model_config)
