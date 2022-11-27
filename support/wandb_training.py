import wandb
import torch

from support.support_training import VAE_and_classifier_loss

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
        
        train_loss      = advance_epoch_wand(model, optimizer, train_loader, train_config, True)
        validation_loss = advance_epoch_wand(model, optimizer, validation_loader, train_config, False)

        log_dict['train_loss']      = train_loss
        log_dict['validation_loss'] = validation_loss

        # Save the model after the epoch
        # N.b. When the variable epoch is 0 the model is trained for an epoch when arrive at this instructions.
        if (epoch + 1) % train_config['epoch_to_save_model'] == 0:
            add_model_to_artifact(model, model_artifact, "TMP_File/model_{}.pth".format(epoch + 1))
        
        # Log data on wandb
        wandb.log(log_dict)
        
        # Update learning rate (if a scheduler is provided)
        if lr_scheduler is not None: lr_scheduler.step()
        

def advance_epoch_wand(model, optimizer, loader, train_config, is_train):
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
                                                config['use_shifted_VAE_loss'], config['alpha'], config['beta'], config['gamma'], 
                                                config['L2_loss_type'])
            total_loss = loss_list[0]

            # total_loss, recon_loss, kl_loss, discriminator_loss
        
            # Backward/Optimization pass
            total_loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                pass


#%% Other function
def add_model_to_artifact(model, artifact, model_name = "model.pth"):
    torch.save(model.state_dict(), model_name)
    artifact.add_file(model_name)
    wandb.save(model_name)
