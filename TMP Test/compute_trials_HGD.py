import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

from support_function_HGD import downloadDataset, computeTrialsHGD


#%%

idx_list = np.linspace(1, 3, 3)
resampling = True
envelope = False
idx_list = [3]

for idx in idx_list:
    
    # downloadDataset(int(idx))
    
    type_dataset = 'Train/'
    min_length = computeTrialsHGD(idx, type_dataset, resampling = resampling, envelope = envelope)
    print("END Train Set - Min length = {}\n".format(min_length))
    
    type_dataset = 'Test/'
    min_length = computeTrialsHGD(idx, type_dataset, resampling = resampling, envelope = envelope, min_length = min_length)
    print("END Test Set - Min length = {}\n".format(min_length))
    