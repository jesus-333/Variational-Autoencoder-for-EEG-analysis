import wandb
import torch


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
        train_cycle(model, loader_list, train_config)

        # Save model after training
        add_model_to_artifact(model, model_artifact, "TMP_File/model_END.pth")
        run.log_artifact(model_artifact)
        
    return model

#%% Train cycle function

def train_cycle(model, optimizer, loader_list, model_artifact, train_config, lr_scheduler = None):
    """
    Function with the training cycle
    """

    # Parameter used to save the model every x epoch
    if 'epoch_to_save_model' not in config: config['epoch_to_save_model'] = 1
    
    log_dict = {}
    # Check the type of model
    for epoch in range(train_config['epochs']):
        # Save lr
        if train_config['use_scheduler']:
            log_dict['learning_rate'] = optimizer.param_groups[0]['lr']
        
        train_loss = advance_epoch_wand()
        validation_loss = advance_epoch_wand()

        # Save the model after the epoch
        # N.b. When the variable epoch is 0 the model is trained for an epoch when arrive at this instructions.
        if (epoch + 1) % config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        

def advance_epoch_wand():
    """
    Function to advance a single epoch of the model
    """
    pass

#%% Other function
def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)
