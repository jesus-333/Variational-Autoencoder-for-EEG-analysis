"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Function to the creation the PyTorch dataset with EEG data trasformed through stft (or similar transformation like wavelet)
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

"""
%load_ext autoreload
%autoreload 2

import dataset_stft as ds_stft
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset_stft(Dataset):

    def __init__(self, data, labels, info : dict):
        """
        data = data used for the dataset. Must have shape [Trials x channels x frequency samples x time samples]
        labels = labels of each trial
        info = dictionary with extra info about the data
            channels_list = list with the name of the eeg channel
            subjects_list = List of the subject of the original dataset used to create this istance of the dataset
            t = time array with the value of the time instants after the stft
            f = frequency array with the value of the frequency bins after the stft    
        """

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()
        
        self.ch_list = info['channels_list'] 
        self.t = info['t'] 
        self.f = info['f'] 
        if 'subjects_list' in info: self.subjects_list = info['subjects_list']

        self.labels_dict_int_to_name = { 0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue'}
        self.labels_dict_name_to_int = { 'left' : 0, 'right' : 1, 'foot' : 2, 'tongue' : 3}
            
    def __getitem__(self, idx : int):
        return self.data[idx], self.labels[idx]    
    
    def __len__(self):
        return len(self.labels)
    
    def visualize_trial(self, idx_trial, ch, vmin = None, vmax = None):
        idx_ch = self.ch_list == ch
        trial = self.data[idx_trial, idx_ch].squeeze()
        
        if vmin is None: vmin = float(trial.min())
        if vmax is None: vmax = float(trial.max())

        plot_config = self.create_plot_config(vmin, vmax, ch)
        plot_config['title'] = "Trials n. {} - {} - Ch. {}".format(idx_trial, self.labels_dict_int_to_name[int(self.labels[idx_trial])], ch)
        self.__plot_stft(trial, plot_config)

    def visualize_average_trial(self, ch, label, vmin = None, vmax = None):
        if type(label) == str : label = self.labels_dict_name_to_int[label]

        idx_trial = self.labels == label
        idx_ch = self.ch_list == ch
        trial = self.data[idx_trial, idx_ch].mean(0)

        if vmin is None: 
            vmin = float(trial.min())
        else:
            trial[trial < vmin] = vmin
        if vmax is None: 
            vmax = float(trial.max())
        else:
            trial[trial > vmax] = vmax

        plot_config = self.create_plot_config(vmin, vmax, ch)
        plot_config['title'] = "Average for class {} and Ch. {}".format(self.labels_dict_int_to_name[label], ch)

        self.__plot_stft(trial, plot_config)

    def create_plot_config(self, vmin, vmax, ch):
        plot_config = dict(
            figsize = (15, 10),
            fontsize = 11,
            vmin = vmin,
            vmax = vmax,
            ch_to_plot = ch,
            cmap = 'Blues_r'
        )

        return plot_config
    
    def __plot_stft(self, stft_data, config):
        fig, ax = plt.subplots(1, 1, figsize = config['figsize'])
        plt.rcParams.update({'font.size': config['fontsize']})

        im = ax.pcolormesh(self.t, self.f, stft_data,
                           shading = 'gouraud', cmap = config['cmap'],
                           vmin = config['vmin'], vmax = config['vmax'])

        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        ax.title.set_text(config['title'])

        fig.colorbar(im)
        fig.tight_layout()
        plt.show()
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Check data

def main():
    import config_dataset as cd
    import preprocess as pp
    
    dataset_config = cd.get_moabb_dataset_config([3])
    dataset_config['stft_parameters'] = cd.get_config_stft()
    
    train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)
    
    print(train_dataset[0][0].shape)
    print(train_dataset.ch_list)
    
    train_dataset.visualize_average_trial('C3', 1)

if __name__ == "__main__":
    main()

#%% End file
