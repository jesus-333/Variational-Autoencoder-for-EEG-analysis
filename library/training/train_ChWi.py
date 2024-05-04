"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Train function of the Channel Wise Network
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

# Python library
import torch
import pprint

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    train_loss = 0
    recon_loss = 0
    kl_loss = 0
    clf_loss = 0

    for sample_data_batch, sample_label_batch in train_loader:
        # Move data to training device
        x = sample_data_batch.to(train_config['device'])

        # Zeros past gradients
        optimizer.zero_grad()
        
        # Networks forward pass
        x_r, z, mu, log_var  = model(x)

        # Add EEG channel dimension. It is necessary to compute the kullback, since the original function was written for batch with 4 dimensions
        # Note that ChWi use tensor of shape B x D x T. The loss function want tensor of shape B x D x C x T
        x = x.unsqueeze(2)
        x_r = x_r.unsqueeze(2)
        mu = mu.unsqueeze(2)
        log_var = log_var.unsqueeze(2)
               
        # Loss evaluation
        batch_train_loss = loss_function.compute_loss(x, x_r, mu, log_var,
                                                      predicted_label = None, true_label = None
                                                      )
    
        # Backward/Optimization pass
        batch_train_loss[0].backward()
        optimizer.step()

        # Accumulate the loss
        train_loss += batch_train_loss[0] * x.shape[0]
        recon_loss += batch_train_loss[1] * x.shape[0]
        kl_loss    += batch_train_loss[2] * x.shape[0]
        if train_config['use_classifier']: clf_loss += batch_train_loss[4] * x.shape[0]

    # Compute final loss
    train_loss = train_loss / len(train_loader.sampler)
    recon_loss = recon_loss / len(train_loader.sampler)
    kl_loss = kl_loss / len(train_loader.sampler)

    if log_dict is not None:
        log_dict['train_loss'] = float(train_loss)
        log_dict['train_loss_recon'] = float(recon_loss)
        log_dict['train_kl_loss'] = float(kl_loss)

        print("TRAIN LOSS")
        pprint.pprint(log_dict)
    
    return train_loss


def validation_epoch(model, loss_function, validation_loader, train_config, log_dict = None):
    # Set the model in training mode
    model.train()

    # Variable to accumulate the loss
    validation_loss = 0
    recon_loss = 0
    kl_loss = 0
    clf_loss = 0

    with torch.no_grad() :

        for sample_data_batch, sample_label_batch in validation_loader:
            # Move data to training device
            x = sample_data_batch.to(train_config['device'])
            
            # Networks forward pass
            x_r, z, mu, log_var  = model(x)

            # Add EEG channel dimension. It is necessary to compute the kullback, since the original function was written for batch with 4 dimensions
            # Note that ChWi use tensor of shape B x D x T. The loss function want tensor of shape B x D x C x T
            x = x.unsqueeze(2)
            x_r = x_r.unsqueeze(2)
            mu = mu.unsqueeze(2)
            log_var = log_var.unsqueeze(2)
                   
            # Loss evaluation
            batch_validation_loss = loss_function.compute_loss(x, x_r, mu, log_var,
                                                          predicted_label = None, true_label = None
                                                          )

            # Accumulate the loss
            validation_loss += batch_validation_loss[0] * x.shape[0]
            recon_loss += batch_validation_loss[1] * x.shape[0]
            kl_loss    += batch_validation_loss[2] * x.shape[0]
            if train_config['use_classifier']: clf_loss += batch_validation_loss[4] * x.shape[0]

        # Compute final loss
        validation_loss = validation_loss / len(validation_loader.sampler)
        recon_loss = recon_loss / len(validation_loader.sampler)
        kl_loss = kl_loss / len(validation_loader.sampler)

        if log_dict is not None:
            log_dict['validation_loss'] = float(validation_loss)
            log_dict['train_loss_recon'] = float(recon_loss)
            log_dict['train_kl_loss'] = float(kl_loss)

            print("VALIDATION LOSS")
            pprint.pprint(log_dict)
        
        return validation_loss
