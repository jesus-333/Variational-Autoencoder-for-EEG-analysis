import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import os

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datasets.moabb import HGD, MOABBDataset
import mne
from braindecode.datautil.windowers import create_windows_from_events, create_fixed_length_windows
import h5py

#%%
idx = 5
path = 'tmp_data/'

a = HGD(idx)

#%%

file_name = 'tmp_data/1_train.mat'
f1 = h5py.File(file_name,'r+')
f1_keys = list(f1.keys())
f1_special_key = [obj for obj in f1_keys if 'ch' not in obj and 'obj' not in obj]

for key in f1_special_key: print(f1[key].keys())


with h5py.File(file_name, "r") as h5file:
    a = h5file['nfo']['clab']
    
#%%
cnt = BBCIDataset(filename='tmp_data/1_train.mat', load_sensor_names=None)
cnt_dataset = cnt.load()

# a = cnt_dataset.to_data_frame()

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
    
    
    
#%% Test download

from moabb.datasets import Schirrmeister2017 as HGD
test_HGD = HGD()

test_HGD.data_path(1, 'tmp_data')


#%%
from braindecode.datasets.moabb import HGD, MOABBDataset
dataset = MOABBDataset(dataset_name="Schirrmeister2017", subject_ids=[1])