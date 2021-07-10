import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import sys

import torch
from torch import nn

#%%

class Pytorch_Dataset_HGD(torch.utils.data.Dataset):
    
    # Inizialization method
    def __init__(self, path, n_elements = -1, normalize_trials = False, binary_mode = -1):
        tmp_list = []
        
        # Read all the file in the folder and return them as list of string
        for element in os.walk(path): tmp_list.append(element)
        
        # print(path, tmp_list)
        self.path = tmp_list[0][0]
        self.file_list = tmp_list[0][2]
        
        self.path_list = []
        
        for i in range(len(self.file_list)): 
            file = self.file_list[i]
            self.path_list.append(path + file)
            
            if(i >= (n_elements - 1) and n_elements != -1): break
            
        self.path_list = np.asarray(self.path_list)
        
        # Retrieve dimensions
        tmp_trial = loadmat(self.path_list[0])['trial']
        self.channel = tmp_trial.shape[0]
        self.samples = tmp_trial.shape[1]
        
        # Set binary mode
        self.binary_mode = binary_mode
        
        if(normalize_trials):
            # Temporary set to false to allow to find max and min val
            self.normalize_trials = False
            self.max_val = self.maxVal()
            self.min_val = self.minVal()
            
        # Set to real value
        self.normalize_trials = normalize_trials  
        
        
    def __getitem__(self, idx):
        tmp_dict = loadmat(self.path_list[idx])
        
        # Retrieve and save trial 
        trial = tmp_dict['trial']
        trial = np.expand_dims(trial, axis = 0)
        if(self.normalize_trials): trial = self.normalize(trial)
        
        # Retrieve label. Since the original label are in the range 1-4 I shift them in the range 0-3 for the NLLLoss() loss function
        label = int(tmp_dict['label']) - 1
        if(self.binary_mode != -1): 
            if(label == self.binary_mode): label = 0
            else: label = 1
               
        # Convert to PyTorch tensor
        trial = torch.from_numpy(trial).float()
        label = torch.tensor(label).long()
        
        return trial, label
    
    
    def __len__(self):
        return len(self.path_list)
    
    
    def maxVal(self):
        max_ret = - sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_max = float(torch.max(el))
            if(tmp_max > max_ret): max_ret = tmp_max
            
        return max_ret
    
    
    def minVal(self):
        min_ret = sys.maxsize
        for i in range(self.__len__()):
            el = self.__getitem__(i)[0]
            tmp_min = float(torch.min(el))
            if(tmp_min < min_ret): min_ret = tmp_min
        
        return min_ret
    
    
    def normalize(self, x, a = 0, b = 1):
        x_norm = (x - self.min_val) / (self.max_val - self.min_val) * (b - a) + a
        return x_norm