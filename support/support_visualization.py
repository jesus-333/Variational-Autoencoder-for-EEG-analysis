import torch
import numpy as np
import matplotlib.pyplot as plt

#%%


def computeHiddenSpaceRepresentation(vae, dataset, n_elements = 200, device = 'cpu', print_var = True):
    """
    Create two dictionaries containing the mu (mean) and the standard deviation for each element of the dataset. The elements are divided by class

    """
    vae = vae.to(device)
    
    mu_lists = {0:[], 1:[], 2:[], 3:[]}
    std_list = {0:[], 1:[], 2:[], 3:[]}
    
    plt.figure(figsize = (10, 10))
    
    if(n_elements <= 0): n_elements = len(dataset)
    
    for i in range(n_elements):
        if(print_var and i % 3 == 0): print("Completition: {}".format(round(i/n_elements * 100, 2)))
        
        x_eeg = dataset[i][0].unsqueeze(0).to(device)
        label = int(dataset[i][1])
        
        z = vae.encoder(x_eeg)
        
        mu = z[0].cpu().squeeze().detach().numpy()
        
        # N.B. Since I obtain the logarimt of the variance from the VAE I moltiply for 0.5 = 1/2 to obtain the standard deviation
        std = torch.exp(0.5 * z[1]).cpu().squeeze().detach().numpy()
        
        # if(label == 0): 
        #     plt.plot(x, y, 'ko', alpha = 0.4)
        # elif(label == 1): 
        #     plt.plot(x, y, 'ro')
        # elif(label == 2): 
        #     plt.plot(x, y, 'yo')
        # elif(label == 3): 
        #     plt.plot(x, y, 'bo')
            
        mu_lists[label].append(mu)
        std_list[label].append(std)
        
    
    for label in mu_lists.keys():
        mu_lists[label] = np.asarray(mu_lists[label])
        std_list[label] = np.asarray(std_list[label])
    
    return mu_lists, std_list

def visualizeHiddenSpace(mu_list, std_list, idx_hidden_space = (0,1), sampling = True, figsize = (10, 10), alpha = 0.8, s = 0.3):
    
    plt.figure(figsize = figsize)
    for label in mu_list.keys():
        mu = mu_list[label]
        std = std_list[label]
        
        if(sampling):
            x = np.random.normal(mu[:, idx_hidden_space[0]], std[:, idx_hidden_space[0]], mu.shape[0])
            y = np.random.normal(mu[:, idx_hidden_space[1]], std[:, idx_hidden_space[1]], mu.shape[0])
        else:
            x = mu[:, idx_hidden_space[0]]
            y = mu[:, idx_hidden_space[1]]
            
        print(x.shape)
        print(y.shape)
        if(label == 0): 
            plt.scatter(x, y, c = 'red', s = s, alpha = alpha)
        elif(label == 1): 
            plt.scatter(x, y, c ='blue', s = s, alpha = alpha)
        elif(label == 2): 
            plt.scatter(x, y, c = 'green', s = s, alpha = alpha)
        elif(label == 3): 
            plt.scatter(x, y, c = 'orange', s = s, alpha = alpha)