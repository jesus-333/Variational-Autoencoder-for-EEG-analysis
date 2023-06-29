"""
Created on Tue Jun 27 19:50:47 2023

@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import wandb
import torch
import pprint
import os
import shutil

# Config file
import config_dataset as cd
import config_model as cm
import config_training as ct

import train_generic
import preprocess as pp
import support_function
import metrics

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
        update_config_with_sweep_parameters(config, dataset_config)
        update_config_with_sweep_parameters(config, train_config)
        update_config_with_sweep_parameters(config, model_config)
        model_config['filter_2'] = model_config['filter_1'] * model_config['D']
        print("Update config with sweep parameters")
        
        # Specific check for the STFT parameters
        if dataset_config['use_stft_representation']:
            update_config_with_sweep_parameters(config, dataset_config['stft_parameters'])
            # if config['window'] == ['gaussian', 1]: dataset_config['stft_parameters']['window'] = ('gaussian',1)
            # if config['window'] == ['gaussian', 2]: dataset_config['stft_parameters']['window'] = ('gaussian',2)
            if type(dataset_config['stft_parameters']['window']) == list: dataset_config['stft_parameters']['window'] = tuple(dataset_config['stft_parameters']['window'])
            
        
        # Update wandb config dictionary to save the complete config 
        tmp_config = dict(
            dataset_config = dataset_config,
            train_config = train_config,
            model_config = model_config
        )
        wandb.config.update(tmp_config)
                
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

def update_config_with_sweep_parameters(sweep_config, other_config):
    for key in other_config:
        if key in sweep_config:
            other_config[key] = sweep_config[key]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% 

def compute_metrics_from_sweep_EEGNet_classifier(sweep_id : str):
    # Get the runs from wandb Api
    api = wandb.Api()
    # Keep only the run with accuracy validation higher than 0.6
    runs = filter_list_of_runs(api.sweep(sweep_id).runs, 'accuracy_validation', 0.6)

    weight_path = support_function.get_sweep_path(sweep_id)
    tmp_str = ""
    for run in runs:
        dataset_config  = run.config['dataset_config']
        model_config    = run.config['model_config']
        train_config    = run.config['train_config']
        if type(dataset_config['stft_parameters']['window']) == list: dataset_config['stft_parameters']['window'] = tuple(dataset_config['stft_parameters']['window'])

        # Create dataset and dataloader
        train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
        train_dataloader        = torch.utils.data.DataLoader(train_dataset, batch_size = train_config['batch_size'], shuffle = True)
        validation_dataloader   = torch.utils.data.DataLoader(validation_dataset, batch_size = train_config['batch_size'], shuffle = True)
        test_dataloader         = torch.utils.data.DataLoader(test_dataset, batch_size = train_config['batch_size'], shuffle = True)
        
        # Create untrained model
        if dataset_config['use_stft_representation']:
            model_config['C'] = train_dataset[0][0].shape[0]
            model_config['T'] = train_dataset[0][0].shape[2]
            model_config['depth_first_layer'] = train_dataset[0][0].shape[0]
        model_config['input_size'] = train_dataset[0][0].unsqueeze(0).shape
        model_config['print_var'] = False
        model = train_generic.get_untrained_model('EEGNet', model_config)

        # Create temporary folder to download and save the weights of the network
        os.makedirs('TMP/', exist_ok = True)

        # Download the weight
        path_end = weight_path + str(train_config['epochs']) + '.pth' # Path for the files of the weights at the end of the training
        path_BEST = weight_path + 'BEST.pth' # Path for the files of the weights that give the minimum validation loss

        run.file(path_end).download('TMP/', exist_ok=True)
        run.file(path_BEST).download('TMP/', exist_ok=True)

        tmp_str += "Subject : " + str(dataset_config['subjects_list']) + "\n"

        model.load_state_dict(torch.load('TMP/' + path_end, map_location = 'cpu'))
        metrics_dict = metrics.compute_metrics(model, test_dataloader, 'cpu')
        tmp_str += "\tEND ACCURACY  : " + str(metrics_dict['accuracy']) + "\n"

        model.load_state_dict(torch.load('TMP/' + path_BEST, map_location = 'cpu'))
        metrics_dict = metrics.compute_metrics(model, test_dataloader, 'cpu')
        tmp_str += "\tBEST ACCURACY : " + str(metrics_dict['accuracy']) + "\n\n"

        print(tmp_str)
    
    print(tmp_str)

def filter_list_of_runs(list_of_runs, metric, min_value,):
    """
    Keep only the run not failed and with a minimum value in the specifeid metric
    """

    # Keep only the run completed (so no crash/failures)
    complete_runs = [run for run in list_of_runs if run.state == 'finished']
    
    # Keep only the run with a metric equals or higher the min_value
    filtered_runs =  [run for run in complete_runs if run.summary[metric] >= min_value]

    return filtered_runs

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%%

def main_sweep():
    sweep_config = ct.get_config_sweep('accuracy_validation', 'maximize')
    
    # Used only to create the sweep 
    # train_config = ct.get_config_classifier()
    # sweep_id = wandb.sweep(sweep_config, project = train_config['project_name'])
    
    # Bayes search, subject 2, 3 and 8
    # sweep_id = 'jesus_333/ICT4AWE_Extension/mp2qyzk8'
    
    # Random search, all subject
    sweep_id = 'jesus_333/ICT4AWE_Extension/wjim0nwt'
    
    wandb.agent(sweep_id = sweep_id, function = train_sweep)

if __name__ == '__main__':
    main_sweep()
