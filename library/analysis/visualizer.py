import torch
import numpy as np
import matplotlib.pyplot as plt

from ..dataset import preprocess as pp

from ..config import config_dataset as cd

from ..training.train_generic import get_untrained_model

"""
%load_ext autoreload
%autoreload 2
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class Visualizer():

    def __init__(self, config : dict):
        """
        Class to visualize different results about vEEGNet
        """ 
        self.train_dataset_list, self.validation_dataset_list, self.test_dataset_list = self.get_data(cd.get_moabb_dataset_config(config['subj_list']))

        self.model = get_untrained_model(config['model_name'], config['model_config'])
        self.device = config['device'] if 'device' in config else 'cpu'
        if 'path_weight' in config: self.update_model_weight(config['path_weight'])

        # Setting for visualization in time domain
        self.t_start = config['dataset_config']['trial_start'] if 'trial_start' in config['dataset_config'] else 2
        self.t_end = config['dataset_config']['trial_end'] if 'trial_end' in config['dataset_config'] else 4
        self.t_min_to_visualize = config['dataset_config']['trial_start'] if 'trial_start' in config['dataset_config'] else 2
        self.t_max_to_visualize = config['dataset_config']['t_max_to_visualize'] if 't_max_to_visualize' in config['dataset_config'] else 4

        # Various setting
        self.figsize = config['figsize'] if 'figsize' in config else (15, 10)
        self.fontsize = config['fontsize'] if 'fontsize' in config else 12
        self.save_fig = config['save_fig'] if 'save_fig' in config else False

    def get_data(self, subj_list):
        train_dataset_list = []
        validation_dataset_list = []
        test_dataset_list = []

        for subj in subj_list:
            # Get the data
            dataset_config = cd.get_moabb_dataset_config([subj])
            train_dataset, validation_dataset, test_dataset = pp.get_dataset_d2a(dataset_config)

            train_dataset_list.append(train_dataset)
            validation_dataset_list.append(validation_dataset)
            test_dataset_list.append(test_dataset)

        return train_dataset, validation_dataset, test_dataset

    def set_model(self, model_name : str, model_config : str):
        self.model = get_untrained_model(model_name,model_config)

    def set_dataset(self, subj_list):
        self.train_dataset_list, self.validation_dataset_list, self.test_dataset_list = self.get_data(cd.get_moabb_dataset_config(subj_list))

    def update_model_weight(self, path_weight : str):
        self.model.load_state_dict(torch.load(path_weight, map_location = self.device))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def plot_reconstructed_signal(self, idx_subj : int, idx_trial : int, idx_ch : int, path_weight = None):
        x, x_r, label = self.compute_reconstructed_signal(idx_subj, idx_trial, path_weight)

        t = np.linspace(self.t_start, self.t_end, x.shape[-1])
        idx_t = np.logical_and(t >= self.t_min_to_visualize, t <= self.t_max_to_visualize)

        x_plot = x.squeeze()[idx_ch, idx_t]
        x_r_plot = x_r.squeeze()[idx_ch, idx_t]

        plt.rcParams.update({'font.size': self.fontsize})
        fig, ax = plt.subplots(1, 1, figsize = self.figsize)
        
        ax.plot(t, x_plot, label = 'Original Signal')
        ax.plot(t, x_r_plot, label = 'Reconstruced Signal')
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Amplitude [microV]')

        ax.legend()
        ax.grid(True)
        
        fig.tigh_layout()
        fig.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def compute_reconstructed_signal(self, idx_subj : int, idx_trial : int, path_weight = None):
        if path_weight is not None: self.model.load_state_dict(torch.load(path_weight, map_location = self.device))
        
        label_dict = {0 : 'left', 1 : 'right', 2 : 'foot', 3 : 'tongue' }
        # TODO chiedere a giulia il canale della lingua
        # label_to_ch = {'left' : 11, 'right' : 7, 'foot' : 9, 'tongue' : -1 }

        with torch.no_grad():
            x = self.test_dataset_list[idx_subj][idx_trial][0]
            label = label_dict[int(self.test_dataset_list[idx_subj][idx_trial][1])]
            # idx_ch = label_to_ch[label]
            
            output = self.model(x.unsqueeze(0).to(self.device))
            x_r = output[0]

            return x.to('cpu'), x_r.to('cpu'), label.to('cpu')


