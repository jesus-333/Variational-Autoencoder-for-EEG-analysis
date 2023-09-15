import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import pdist as pdist
from scipy.spatial.distance import squareform as sf

from . import dtw_analysis 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_latent_space_dataset(model, dataset, config : dict, device = 'cpu'):
    with torch.no_grad():
        # Convert dataset into PyTorch dataloader
        batch_size = config['batch_size'] if 'batch_size' in config else 32
        dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = False)

        # Tensor to save the results
        # print(model.h_vae.hidden_space_size_flatten)
        hidden_space_embedding = torch.zeros(len(dataset), int(model.h_vae.hidden_space_size_flatten))
        labels = torch.zeros(len(dataset))

        if config['compute_recon_error']: recon_error = torch.zeros(len(dataset))
        if 'reduce_dimension' not in config: config['reduce_dimension'] = False
        
        model.to(device)

        i = 0
        for batch in dataloader:
            # Take the EEG Signal
            x = batch[0]
            labels[i * batch_size : (i + 1) * batch_size] = batch[1]
            
            # Encode the input
            z, mu, log_var = model.encode(x.to(device), return_distribution = True)
            # Note that the z return from the methods is obtained witht the reparametrization trick
            # So It is like sampling from the distribution
            
            # Save the latent space embedding
            if config['sample_from_distribution']: # Save samples from the distribution
                hidden_space_embedding[i * batch_size : (i + 1) * batch_size] = z.flatten(1)
            else: # Save the mean vector (i.e. the aproximate location in the latent space)
               hidden_space_embedding[i * batch_size : (i + 1) * batch_size] = mu.flatten(1)

            if config['compute_recon_error']:
                x_r = model.reconstruct(x.to(device))
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                recon_error[i * batch_size : (i + 1) * batch_size] = dtw_analysis.compute_dtw_softDTWCuda(x, x_r, device)

            i += 1

        if config['reduce_dimension']:
            z_reduced = reduce_dimension_lanten_space(hidden_space_embedding, config)
        else:
            z_reduced = hidden_space_embedding

        if config['compute_recon_error']:
            return z_reduced, recon_error
        else:
            return z_reduced

def reduce_dimension_lanten_space(z : torch.Tensor, config : dict):
    if config['reduction_methods'] == 'tsne':
        tsne = TSNE(
            n_components = 2,
            perplexity = config['perplexity'] if 'perplexity' in config else 30,
            n_iter = config['n_iter'] if 'n_iter' in config else 1000,
            verbose = 1
        )
        
        z_reduced = tsne.fit_transform(z)
    elif config['reduction_methods'] == 'pca':
        pca = PCA(n_components=2)   
        z_reduced = pca.fit_transform(z)
    elif config['reduction_methods'] == 'umap':
        print("Umap not yet implemented")
    else:
        raise ValueError("reduction_methods value not valid. The possible values are: tsne, pca, umap")

    return z_reduced 


def plot_latent_space(z_reduced_list : list, plot_config : dict, color = None):
    fig, axs = plt.subplots(1, len(z_reduced_list), figsize = plot_config['figsize'])
    plt.rcParams.update({'font.size': plot_config['fontsize']})

    for i in range(len(z_reduced_list)):
        z_reduced = z_reduced_list[i]
        ax = axs[i]

        ax.scatter(z_reduced[:, 0], z_reduced[:, 1],
                   s = plot_config['markersize'], c = color,
                   cmap = plot_config['colormap'], alpha = plot_config['alpha'])
        
        ax.set_title(plot_config['reduction_method_name_list'][i])

    fig.tight_layout()
    if plot_config['show_fig']: fig.show()


def clustering_evaluation(X, centers, labels):
    '''
    INPUT
    X       - data matrix for which to compute the proximity matrix
    centers - cluster centres from the clustering solution applied to X
    labels  - predicted labels from the clustering solution applied to X
    '''

    '''
    OUTPUT
    PM - proximity matrix computed on X (using euclidean distance metric)
    d  - average distance between pairs of objects in each cluster
    D  - inter-cluster distances
    '''

    K = len(np.unique(labels))

    METRIC = 'euclidean'
    PM = pdist(X, metric=METRIC)
    PM = sf(PM).round(2)

    # [NEW CORRECTED CODE!!] Intra-cluster distances (average over all pairwise distances) -----------------
    # You could alternatively compute this measure as the average distance of all points from its centroid.
    d = np.zeros(K)
    for k in range(K):
        ind = np.array( np.where(labels == k ) )
        for r in range(ind.size):
            d[k] = d[k] + np.sum( PM[ [ind[0][r]], [ind] ] )
    d[k] = d[k]/2                                          # not to consider pairs of pair-wise distance between objects twice (the PM is symmetric)
    d[k] = d[k]/( (ind.size*(ind.size-1)) / 2 )            # to compute the average among N*(N-1)/2 possible unique pairs
    # print("The intra-cluster distance of the three clusters are: ", d.round(2))
    # ------------------ CORRECTED CODE FOR INTER-CLUSTER DISTANCE --------------------------------

    # Inter-cluster distance
    D = pdist(centers, metric=METRIC)
    D = sf(D).round(2)

    # print("The inter-cluster distances are:\n |dist(C_0,C_1)| = %.2f \n |dist(C_0,C_2)| = %.2f \n |dist(C_1,C_2)| = %.2f " % (D[0,1].round(2), D[0,2].round(2), D[1,2].round(2)))

    return PM, d, D

