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
            
#%%

def plot_psd_V1(psd_original, psd_reconstructed, config):
    """
    Compare the PSD of the original signal and of the reconstructed signal in separate plot
    """
    
    fig, ax = plt.subplots(2, len(config['ch_list']), figsize = (15, 10))
    
    channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    channel_list = np.asarray(channel_list)
    
    for i in range(len(config['ch_list'])):
        idx_ch = channel_list == config['ch_list'][i]
    
        ax[0, i].plot(config['x_freq'], psd_original[idx_ch].squeeze(), 
                      color = config['color_list'][i])
        ax[1, i].plot(config['x_freq'], psd_reconstructed[idx_ch].squeeze(), 
                      color = config['color_list'][i])
        
        ax[0, i].set_xlabel('Frequency [Hz]')
        ax[1, i].set_xlabel('Frequency [Hz]')
        
        ax[0, i].set_ylabel('PSD')
        ax[1, i].set_ylabel('PSD')
                
        ax[0, i].set_title('Original - ' + config['ch_list'][i])
        ax[1, i].set_title('Reconstructed - ' + config['ch_list'][i])
    
    if 'font_size' in config: plt.rcParams.update({'font.size': config['font_size']})
    plt.tight_layout()
    
    
def plot_psd_V2(psd_original_list, psd_reconstructed_list, config):
    fig, ax = plt.subplots(1, len(config['ch_list']), figsize = (15, 10))
    
    channel_list = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    channel_list = np.asarray(channel_list)
    
    for i in  range(len(config['ch_list'])):
        for j in range(len(psd_original_list)):
            psd_original = psd_original_list[j]
            psd_reconstructed = psd_reconstructed_list[j]
            
            idx_ch = channel_list == config['ch_list'][i]
        
            ax[i].plot(config['x_freq'], psd_original[idx_ch].squeeze(), 
                          color = config['color_list'][i], label = 'Original')
            ax[i].plot(config['x_freq'], psd_reconstructed[idx_ch].squeeze(), 
                          color = config['color_list'][i], label = 'Reconstructed')
            
            ax[i].set_xlabel('Frequency')
            ax[i].set_xlabel('Frequency')
            
            ax[i].set_ylabel('PSD')
            ax[i].set_ylabel('PSD')
            
            
            ax[i].set_title(config['ch_list'][i])
            ax[i].set_title(config['ch_list'][i])
            
            ax[i].legend()
    
    if 'font_size' in config: plt.rcParams.update({'font.size': config['font_size']})
    plt.tight_layout()