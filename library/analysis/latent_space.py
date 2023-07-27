import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_latent_space_dataset(model, dataset, config : dict):
    with torch.no_grad():
        # Convert dataset into PyTorch dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)

        # Tensor to save the results
        hidden_space_embedding = torch.zeros(len(dataset), model.h_vae.hidden_space_size_flatten)
        labels = torch.zeros(len(dataset))

        i = 0
        for batch in dataloader:
            # Take the EEG Signal
            x = batch[0]
            labels[i * 32 : (i +1) * 32] = batch[1]
            
            # Encode the input
            z, mu, log_var = model.encode(x, return_distribution = True)
            # Note that the z return from the methods is obtained witht the reparametrization trick
            # So It is like sampling from the distribution
            
            # Save the latent space embedding
            if config['sample_from_distribution']: # Save samples from the distribution
                hidden_space_embedding[i * 32 : (i + 1) * 32] = z.flatten(1)
            else: # Save the mean vector (i.e. the aproximate location in the latent space)
               hidden_space_embedding[i * 32 : (i + 1) * 32] = mu.flatten(1)

            i += 1

        z_reduced = reduce_dimension_lanten_space(z, config)
    

def reduce_dimension_lanten_space(z : torch.Tensor, config : dict):
    tsne = TSNE(
        n_components = 2,
        perplexity = config['perplexity'] if 'perplexity' in config else 30,
        n_iter = config['n_iter'] if 'n_iter' in config else 1000,
    )

    z_tsne = tsne.fit_transform(z)

    return z_tsne
