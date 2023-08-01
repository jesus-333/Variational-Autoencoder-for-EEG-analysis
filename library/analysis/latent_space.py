import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from . import dtw_analysis 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_latent_space_dataset(model, dataset, config : dict):
    with torch.no_grad():
        # Convert dataset into PyTorch dataloader
        batch_size = config['batch_size'] if 'batch_size' in config else 32
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

        # Tensor to save the results
        hidden_space_embedding = torch.zeros(len(dataset), model.h_vae.hidden_space_size_flatten)
        labels = torch.zeros(len(dataset))

        if config['compute_recon_error']: recon_error = torch.zeros(len(dataset))

        i = 0
        for batch in dataloader:
            # Take the EEG Signal
            x = batch[0]
            labels[i * batch_size : (i +1) * batch_size] = batch[1]
            
            # Encode the input
            z, mu, log_var = model.encode(x, return_distribution = True)
            # Note that the z return from the methods is obtained witht the reparametrization trick
            # So It is like sampling from the distribution
            
            # Save the latent space embedding
            if config['sample_from_distribution']: # Save samples from the distribution
                hidden_space_embedding[i * batch_size : (i + 1) * batch_size] = z.flatten(1)
            else: # Save the mean vector (i.e. the aproximate location in the latent space)
               hidden_space_embedding[i * batch_size : (i + 1) * batch_size] = mu.flatten(1)

            if config['compute_recon_error']:
                x_r = model.reconstruct(x)
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                recon_error[i * batch_size : (i + 1) * batch_size] = dtw_analysis.compute_dtw_softDTWCuda(x, x_r, device)

            i += 1

        z_reduced = reduce_dimension_lanten_space(z, config)

        if config['compute_recon_error']:
            return z_reduced, recon_error
        else:
            return z_reduced

def reduce_dimension_lanten_space(z : torch.Tensor, config : dict):
    tsne = TSNE(
        n_components = 2,
        perplexity = config['perplexity'] if 'perplexity' in config else 30,
        n_iter = config['n_iter'] if 'n_iter' in config else 1000,
    )

    z_tsne = tsne.fit_transform(z)

    return z_tsne
