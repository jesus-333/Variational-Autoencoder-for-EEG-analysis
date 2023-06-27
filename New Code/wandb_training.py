"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script with the function to train the various network and save the results with the wandb framework
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

# Python library
import wandb

# Custom functions
import train_generic

# Config files
import config_model as cm
import config_dataset as cd 
import config_training as ct

"""
%load_ext autoreload
%autoreload 2

import sys
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def train_wandb_EEGNet(dataset_config, train_config, model_config):
    notes = train_config['notes']

    wandb_config = dict(
        dataset = dataset_config,
        train = dataset_config,
        model = model_config
    )

    with wandb.init(project = train_config['project_name'], job_type = "train", config = wandb_config, notes = notes) as run:
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)

        model = train_generic.train_and_test_model('EEGNet', dataset_config, train_config, model_config, model_artifact)
        
        return model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def main_EEGNet_classifier():
    subject_list = [1,2,3, 4,5,6,7,8,9]
    epochs_list = [300, 400]
    tmp_str = ""
    for subject in subject_list:
        # dataset_config['subjects_list'] = [subject]
        # train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
        # test_dataloader         = torch.utils.data.DataLoader(test_dataset, batch_size = train_config['batch_size'], shuffle = True)
        # model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
        # model_config['depth_first_layer'] = train_dataset[0][0].shape[0]
        for epochs in epochs_list:
            # import preprocess as pp
            # import metrics
            # import torch

            # tmp_str += "Subject {} - epochs {}\n".format(subject, epochs)
            
            # model = model = train_generic.get_untrained_model('EEGNet', model_config)
            # model.load_state_dict(torch.load('TMP_Folder/{}_{}/model_{}.pth'.format(subject, epochs, epochs)))
            # metrics_dict = metrics.compute_metrics(model, test_dataloader, 'cpu')
            # tmp_str += "\tAccuracy END : " + str( metrics_dict['accuracy']) + "\n"
            
            # model.load_state_dict(torch.load('TMP_Folder/{}_{}/model_BEST.pth'.format(subject, epochs)))
            # metrics_dict = metrics.compute_metrics(model, test_dataloader, 'cpu')
            # tmp_str +="\tAccuracy BEST: " + str( metrics_dict['accuracy']) + "\n\n"
            
            
            dataset_config = cd.get_moabb_dataset_config([subject])
            dataset_config['stft_parameters'] = cd.get_config_stft()
            
            train_config = ct.get_config_classifier()
            train_config['wandb_training'] = True
            train_config['notes'] = 'Subject {} - Epochs {}'.format(subject, epochs)
            train_config['epochs'] = epochs
            train_config['path_to_save_model'] = 'TMP_Folder/{}_{}/'.format(subject, epochs)
            
            C = 22
            T = 512 
            model_config = cm.get_config_EEGNet_stft_classifier(C, T, 22)
            
            model = train_wandb_EEGNet(dataset_config, train_config, model_config)
    
    return model

if __name__ == '__main__':
    model = main_EEGNet_classifier()
