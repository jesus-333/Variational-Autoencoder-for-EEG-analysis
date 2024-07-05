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
import time

from library.training import train_generic
from library.config import config_model as cm

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

n_elements_list = [1, 10, 100, 288]
C = 22
T = 1000

use_cuda = False

n_average = 20
no_grad = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_inference_time(model, x, device, no_grad = True, n_average = 20):
    model.to(device)
    x = x.to(device)
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
# Compute the average inference time

model_config = cm.get_config_hierarchical_vEEGNet(C, T, type_decoder = 0, parameters_map_type = 0)

time_average_list = []
time_std_list = []

for i in range(len(n_elements_list)) :
    n_elements = n_elements_list[i]
    print("Compute inference time for {} elements".format(n_elements))

    model_config['input_size'] = (1, 1, C, T)
    model_config['use_classifier'] = False
    model_hv = train_generic.get_untrained_model('hvEEGNet_shallow', model_config)

    x = torch.rand(n_elements, 1, C, T)

    if torch.cuda.is_available() and use_cuda:
        time_average, time_std = compute_inference_time(model_hv, x, 'cuda', no_grad = no_grad, n_average = n_average)
    else :
        time_average, time_std = compute_inference_time(model_hv, x, 'cpu', no_grad = no_grad, n_average = n_average)

    time_average_list.append(time_average)
    time_std_list.append(time_std)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Print the results
for i in range(len(n_elements_list)) :
    n_elements = n_elements_list[i]

    time_average = time_average_list[i]
    time_std = time_std_list[i]

    if torch.cuda.is_available() and use_cuda:
        print("Inference time {} samples (cuda):\t {}s ± {}s".format(n_elements, time_average, time_std))
    else :
        print("Inference time {} samples (cpu) :\t {}s ± {}s".format(n_elements, time_average, time_std))
