"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Script to train the model with the wandb framework
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import wandb
import torch

from metrics import compute_metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Principal function

def train_model_wandb(model, loader_list, train_config, wandb_config):
    with wandb.init(project = wandb_config['project_name'], job_type = "train", config = train_config, name = wandb_config['run_name']) as run:
        train_config = wandb.config

        # Setup optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr = config['lr'], 
                                      weight_decay = config['optimizer_weight_decay'])

        # Setup lr scheduler
        if train_config['use_scheduler'] == True:
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = train_config['gamma'])
        else:
            lr_scheduler = None
            
        # Setup artifact to save model
        model_artifact_name = train_config['model_artifact_name'] + '_trained'
        metadata = dict(training_config = dict(train_config))
        model_artifact = wandb.Artifact(model_artifact_name, type = "model",
                                        description = "Trained {}:{} model".format(train_config['model_artifact_name'], train_config['version']),
                                        metadata = metadata)
        
        # Print the training device
        if train_config['print_var']: print("Model trained on: {}".format(train_config['device']))

        # Train model
        wandb.watch(model, log = "all", log_freq = train_config['log_freq'])
        model.to(train_config['device'])
        train_cycle(model, optimizer, loader_list, model_artifact, train_config, lr_scheduler)

        # Save model after training
        add_model_to_artifact(model, model_artifact, "TMP_File/model_END.pth")
        run.log_artifact(model_artifact)
        
    return model

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train cycle function

def train_cycle(model, optimizer, loader_list, model_artifact, train_config, lr_scheduler = None):
    """
    Function with the training cycle
    """

    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in train_config: train_config['epoch_to_save_model'] = 1
    
    train_loader = loader_list[0]
    validation_loader = loader_list[1]

    log_dict = {}
    # Check the type of model
    for epoch in range(train_config['epochs']):
        # Save lr
        if train_config['use_scheduler']:
            log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        
        # Advance epoch for train set (backward pass) and validation (no backward pass)
        train_loss_list      = advance_epoch(model, optimizer, train_loader, train_config, True)
        validation_loss_list = advance_epoch(model, optimizer, validation_loader, train_config, False)
        
        # Update the log with the epoch loss
        update_log_dict_loss(train_loss_list, log_dict, 'train')
        update_log_dict_loss(validation_loss_list, log_dict, 'validation')

        # Measure Accuracy
        if train_config['measure_accuracy_during_training']:
            train_metrics_list = compute_metrics(model, train_loader)    
            validation_metrics_list = compute_metrics(model, validation_loader)

            update_log_dict_metrics(train_metrics_list, log_dict, 'train')
            update_log_dict_metrics(validation_metrics_list, log_dict, 'validation')

        # Save the model after the epoch
        # N.b. When the variable epoch is 0 the model is trained for an epoch when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        
        # Print loss 
        if train_config['print_var']:
            print("Epoch:{}".format(epoch))
            print(get_loss_string(log_dict))
        

def advance_epoch(model, optimizer, loader, train_config, is_train):
    """
    Function to advance a single epoch of the model
    """

    if is_train: model.train()
    else: model.eval()
    
    for sample_data_batch, sample_label_batch in loader:
        x = sample_data_batch.to(train_config['device'])
        true_label = sample_label_batch.to(train_config['device'])

        if is_train:
            # Zeros past gradients
            optimizer.zero_grad()
            
            # Networks forward pass
            x_r, mu, log_var, predict_label = model(x)
            
            # Loss evaluation
            loss_list = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, 
                                                train_config['use_shifted_VAE_loss'], 
                                                train_config['alpha'], train_config['beta'], train_config['gamma'], 
                                                train_config['L2_loss_type'])
            
            total_loss = loss_list[0]
            # total_loss, recon_loss, kl_loss, discriminator_loss
        
            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                # Forward pass
                x_r, mu, log_var, predict_label = model(x)
            
                # Loss evaluation
                loss_list = VAE_and_classifier_loss(x, x_r, mu, log_var, true_label, predict_label, 
                                                    train_config['use_shifted_VAE_loss'], 
                                                    train_config['alpha'], train_config['beta'], train_config['gamma'], 
                                                    train_config['L2_loss_type'])
        

        # cccumulate the  loss
        tot_loss += total_loss * x.shape[0]
        tot_recon_loss += loss_list[1] * x.shape[0]
        tot_kl_loss += loss_list[2] * x.shape[0]
        tot_discriminator_loss += loss_list[2] * x.shape[0]

    # End training cycle

    # Compute final loss
    tot_loss = tot_loss / len(loader.sampler)
    tot_recon_loss = tot_recon_loss / len(loader.sampler)
    tot_kl_loss = tot_kl_loss / len(loader.sampler)
    tot_discriminator_loss = tot_discriminator_loss / len(loader.sampler)
    
    return [tot_loss, tot_recon_loss, tot_kl_loss, tot_discriminator_loss]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other function

def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)


def update_log_dict_loss(loss_list, log_dict, label):
    tot_loss = loss_list[0]
    recon_loss = loss_list[1]
    kl_loss = loss_list[2]
    discriminator_loss = loss_list[3]

    log_dict["tot_loss_{}".format(label)] = tot_loss
    log_dict["recon_loss_{}".format(label)] = recon_loss
    log_dict["kl_loss_{}".format(label)] = kl_loss
    log_dict["discriminator_loss_{}".format(label)] = discriminator_loss


def update_log_dict_metrics(metrics_list, log_dict, label):
    # return accuracy, cohen_kappa, sensitivity, specificity, f1
    accuracy = metrics_list[0]
    cohen_kappa = metrics_list[1]
    sensitivity = metrics_list[2]
    specificity = metrics_list[3]
    f1 = metrics_list[4]

    log_dict['accuracy_{}'.format(label)] = accuracy
    log_dict['cohen_kappa_{}'.format(label)] = cohen_kappa
    log_dict['sensitivity_{}'.format(label)] = sensitivity
    log_dict['specificity_{}'.format(label)] = specificity
    log_dict['f1_{}'.format(label)] = f1


def get_loss_string(log_dict):
    tmp_loss = ""
    
    tmp_loss += "\t(TRAIN) Tot   Loss:" + str(log_dict['tot_loss_train']) + "\n"
    tmp_loss += "\t(TRAIN) Recon Loss:" + str(log_dict['recon_loss_train']) + "\n"
    tmp_loss += "\t(TRAIN) KL    Loss:" + str(log_dict['kl_loss_train']) + "\n"
    tmp_loss += "\t(TRAIN) Disc  Loss:" + str(log_dict['discriminator_loss_train']) + "\n"

    tmp_loss += "\t(VALID) Tot   Loss:" + str(log_dict['tot_loss_validation']) + "\n"
    tmp_loss += "\t(VALID) Recon Loss:" + str(log_dict['recon_loss_validation']) + "\n"
    tmp_loss += "\t(VALID) KL    Loss:" + str(log_dict['kl_loss_validation']) + "\n"
    tmp_loss += "\t(VALID) Disc  Loss:" + str(log_dict['discriminator_loss_validation']) + "\n"

    return tmp_loss


def compute_label(eeg_framework, loader):
    """
    Method create to compute the label in a dataloader with the eeg_framework class
    """


