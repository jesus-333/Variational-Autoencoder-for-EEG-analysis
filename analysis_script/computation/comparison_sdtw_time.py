"""
Compare the computation time between the SDTW CUDA loss function (https://github.com/Maghoumi/pytorch-softdtw-cuda) (inside the library)
and the same function inside the tslearn library
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

import numpy as np
import torch
import time
import matplotlib.pyplot as plt

try :
    from tslearn.metrics import SoftDTWLossPyTorch
except :
    raise Exception('Please install tslearn library: https://tslearn.readthedocs.io/en/stable/installation.html')

from library.training import soft_dtw_cuda

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

batch_size_list = [1]
t_size_list = np.arange(10, 110, 10) * 10

repetitions = 5

use_cuda = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Compute time for both libraries

# Declare the loss functions
soft_dtw_loss_tslearn = SoftDTWLossPyTorch(gamma = 1)
soft_dtw_loss_cuda = soft_dtw_cuda.SoftDTW(use_cuda = use_cuda, gamma = 1)

# Matrix to save the results
time_tslearn = np.zeros((len(batch_size_list), len(t_size_list)))
time_cuda = np.zeros((len(batch_size_list), len(t_size_list)))

for i in range(len(batch_size_list)) : # Batch iteration
    batch_size = batch_size_list[i]
    for j in range(len(t_size_list)) : # time sample iteration
        print("Computing for batch size = {} and time size = {}".format(batch_size, t_size_list[j]))

        t_size = t_size_list[j]
        
        # Generate random data
        x = torch.rand(batch_size, t_size, 1)
        y = torch.rand(batch_size, t_size, 1)

        if use_cuda :
            x = x.cuda()
            y = y.cuda()
        
        tmp_list_tslearn = []
        tmp_list_cuda = []
        for _ in range(repetitions) :
            # Compute the time for tslearn
            start = time.time()
            sim = soft_dtw_loss_tslearn(x, y)
            tmp_list_tslearn.append(time.time() - start)
            
            # Compute the time for CUDA version
            start = time.time()
            sim = soft_dtw_loss_cuda(x, y)
            tmp_list_cuda.append(time.time() - start)

        time_tslearn[i, j] = np.mean(tmp_list_tslearn)
        time_cuda[i, j] = np.mean(tmp_list_cuda)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Plot the results

fig, ax = plt.subplots(1, 1, figsize = (15, 5))

for i in range(len(batch_size_list)) : # Batch iteration
    ax.plot(t_size_list, time_tslearn[i, :], label = 'tslearn, batch size = {}'.format(batch_size_list[i]))
    ax.plot(t_size_list, time_cuda[i, :], label = 'CUDA, batch size = {}'.format(batch_size_list[i]))

ax.set_xlabel('Time samples')
ax.set_ylabel('Time (s)')
ax.set_xlim([t_size_list[0], t_size_list[-1]])
if use_cuda : ax.set_yscale('log')
ax.grid()
ax.legend()

if use_cuda :
    ax.set_title('Comparison (GPU)')
else :
    ax.set_title('Comparison (CPU)')

fig.tight_layout()
fig.show()











