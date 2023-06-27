"""
Created on Tue Jun 27 19:50:47 2023

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import wandb

import train_generic

# Config file
import config_dataset as cd
import config_model as cm
import config_training as ct


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train function

def train_sweep(config = None):
    # Get other config not included in the ones used for the sweep
    dataset_config = cd.get_moabb_dataset_config()
    
    train_config = ct.get_config_classifier()
    train_config['wandb_training'] = True
    C = 22
    T = 512 
    model_config = cm.get_config_EEGNet_stft_classifier(C, T, 22)
        
    with wandb.init(project = train_config['project_name'], job_type = "train", config = config, notes = train_config['notes']) as run:
        # Config from the sweep
        config = wandb.config
        print("Start Sweep")

        # "Correct" dictionaries with the parameters from the sweep
        correct_config(config, dataset_config)
        correct_config(config, train_config)
        correct_config(config, model_config)
        model_config['filter_2'] = model_config['filter_1'] * model_config['D']
        print("Update config with sweep parameters")
        
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {} model".format(train_config['model_artifact_name']),
                                        metadata = metadata)

        model = train_generic.train_and_test_model('EEGNet', dataset_config, train_config, model_config, model_artifact)
        
        # Log the artifact in wandb
        run.log_artifact(model_artifact)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Dictionary correction (i.e. add the parameter select by the sweep to the dictionaries used during training)

def correct_config(sweep_config, other_config):
    for key in other_config:
        if key in sweep_config:
            other_config[key] = sweep_config[key]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%%

def main_sweep():
    sweep_config = ct.get_config_sweep('accuracy_validation', 'maximize')
    
    # Used only to create the sweep 
    # train_config = ct.get_config_classifier()
    # sweep_id = wandb.sweep(sweep_config, project = train_config['project_name'])
    
    sweep_id = 'jesus_333/ICT4AWE_Extension/mp2qyzk8'
    
    wandb.agent(sweep_id = sweep_id, function = train_sweep)

if __name__ == '__main__':
    main_sweep()