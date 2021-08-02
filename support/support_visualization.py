import torch
import numpy as np
import matplotlib.pyplot as plt

#%%


def visualizeHiddenSpace(vae, dataset, idx_hidden_space = (0,1), sampling = True, n_elements = 200, device = 'cpu', print_var = True):
    """
    Create a 2D plot with the first two element of the hidden space of the VAE
    Return two dictionaries containing the mu (mean) and the standard deviation for each element of the dataset. The elements are divided by class

    """
    vae = vae.to(device)
    
    mu_lists = {0:[], 1:[], 2:[], 3:[]}
    std_lists = {0:[], 1:[], 2:[], 3:[]}
    
    plt.figure(figsize = (10, 10))
    
    for i in range(n_elements):
        if(print_var and i % 3 == 0): print("Completition: {}".format(round(i/n_elements * 100, 2)))
        
        x_eeg = dataset[i][0].unsqueeze(0).to(device)
        label = int(dataset[i][1])
        
        z = vae.encoder(x_eeg)
        
        mu = z[0].cpu().squeeze().detach().numpy()
        
        # N.B. Since I obtain the logarimt of the variance from the VAE I moltiply for 0.5 = 1/2 to obtain the standard deviation
        std = torch.exp(0.5 * z[1]).cpu().squeeze().detach().numpy()
        
        if(sampling):
            x = np.random.normal(mu[idx_hidden_space[0]], std[idx_hidden_space[0]], 1)
            y = np.random.normal(mu[idx_hidden_space[1]], std[idx_hidden_space[1]], 1)
        else:
            x = mu[idx_hidden_space[0]]
            y = mu[idx_hidden_space[1]]
        
        if(label == 0): 
            plt.plot(x, y, 'ko')
        elif(label == 1): 
            plt.plot(x, y, 'ro')
        elif(label == 2): 
            plt.plot(x, y, 'yo')
        elif(label == 3): 
            plt.plot(x, y, 'bo')
            
        mu_lists[label].append(mu)
        std_lists[label].append(std)
        
        
    return mu_lists, std_lists
    