import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datasets.moabb import HGD
from braindecode.datautil.windowers import create_windows_from_events, create_fixed_length_windows


#%%
idx = 3

a = HGD(idx)

#%%
# windows_ds_1 = create_windows_from_events(a, trial_start_offset_samples=0, trial_stop_offset_samples=100,
#     window_size_samples=400, window_stride_samples=100,
#     drop_last_window=False)

# windows_ds_2 = create_windows_from_events(a, trial_start_offset_samples=0, trial_stop_offset_samples=100,
#     window_size_samples=800, window_stride_samples=100,
#     drop_last_window=False)



# for x, y, window_ind in windows_ds_1:
#     print(x.shape, y, window_ind)
#     # print(x.shape[0, -1])
#     break

# for x, y, window_ind in windows_ds_2:
#     print(x.shape, y, window_ind)
#     # print(x.shape[0, -1])
#     break

#%% Dataset transformation

for i in range(2): 
    # i = 0 ----> Train set
    # i = 1 ----> Test  set
    print("Set: ", i)
    
    dataset = a.datasets[i]
    
    dataframe_version = dataset.raw.to_data_frame()
    
    events = dataset.raw.info['events']
    
    numpy_version = dataframe_version.to_numpy()
   
    tmp_dict = {'trial': numpy_version, 'events':events}
    
    if(i == 0): savemat('Train/' + str(idx) + '.mat', tmp_dict)
    if(i == 1): savemat('Test/' + str(idx) + '.mat', tmp_dict)