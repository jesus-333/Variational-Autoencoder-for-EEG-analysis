"""
Compute inference time of hvEEGNet for various settings
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import torch
import numpy as np

from library.analysis import support
from library.config import config_dataset as cd 
import time

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def compute_inference_time(model, x, device, no_grad = True, n_average = 20):
    model.to(device)
    x = x.to(device)
    tot_time = 0
    time_list = []
    if no_grad:
        with torch.no_grad():
            for i in range(n_average):
                start = time.time()
                output = model(x)
                time_list.append(time.time() - start)
    else:
        for i in range(n_average):
            start = time.time()
            output = model(x)
            time_list.append(time.time() - start)
    
    return np.mean(time_list), np.std(time_list)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

subj = 3
dataset_config = cd.get_moabb_dataset_config([subj])
dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config)

x_1   = train_dataset[0][0].unsqueeze(0)
x_10  = train_dataset[0:10][0]
x_100 = train_dataset[0:100][0]
x_all = train_dataset[:][0]

time_1,   std_1   = compute_inference_time(model_hv, x_1, 'cpu')
time_10,  std_10  = compute_inference_time(model_hv, x_10, 'cpu')
time_100, std_100 = compute_inference_time(model_hv, x_100, 'cpu')
time_all, std_all = compute_inference_time(model_hv, x_all, 'cpu')

print("Inference time 1   samples (cpu):\t {}s ± {}s".format(time_1, std_1))
print("Inference time 10  samples (cpu):\t {}s ± {}s".format(time_10, std_10))
print("Inference time 100 samples (cpu):\t {}s ± {}s".format(time_100, std_100))
print("Inference time all samples (cpu):\t {}s ± {}s".format(time_all, std_all))

if torch.cuda.is_available():

    time_1,   std_1   = compute_inference_time(model_hv, x_1, 'cuda')
    time_10,  std_10  = compute_inference_time(model_hv, x_10, 'cuda')
    time_100, std_100 = compute_inference_time(model_hv, x_100, 'cuda')
    time_all, std_all = compute_inference_time(model_hv, x_all, 'cuda')

    print("Inference time 1   samples (gpu):\t {}s ± {}s".format(time_1, std_1))
    print("Inference time 10  samples (gpu):\t {}s ± {}s".format(time_10, std_10))
    print("Inference time 100 samples (gpu):\t {}s ± {}s".format(time_100, std_100))
    print("Inference time all samples (gpu):\t {}s ± {}s".format(time_all, std_all))
