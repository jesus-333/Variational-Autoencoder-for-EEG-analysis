import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat, savemat
import scipy.signal
from scipy.signal import resample

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datasets.moabb import HGD
from braindecode.datautil.windowers import create_windows_from_events, create_fixed_length_windows


#%%

def downloadDataset(idx):
    
    # Download
    a = HGD(idx)
    
    # Divide and save train and test set
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
        
        
#%% 

def computeTrialsHGD(idx, type_dataset, resampling = False, envelope = False, min_length = -1):
    
    tmp_mat = loadmat(type_dataset +'/' + str(idx) + '.mat')
    
    # Recover trials and delete first row (time) and last row (empty)
    trials = tmp_mat['trial'].T
    trials = np.delete(trials, [0, -1], 0)
    
    
    # Retrieve events matrix
    events = tmp_mat['events']
    
    ignore_trial = np.ones(len(events))
    if(min_length == -1): min_length = 100000000
    
    # Search the shortes trial (in number of samples) and the trial to ignore
    for i in range(len(events) - 1):
        
        start_trial_1 = events[i, 0]
        start_trial_2 = events[(i + 1), 0]
        
        # Evaluata trials length
        length_trial = start_trial_2 - start_trial_1
        
        if(length_trial > 6000): 
            # Some trials have length around 7000 so I will ignore them
            ignore_trial[i] = 0
        else:
            # Update min trial length
            if(length_trial < min_length and min_length != -1): 
                min_length = length_trial
                print(min_length)
    
    # Counter to number the trials        
    counter = 0
    
    # Create path for the subject
    path = type_dataset + str(idx) + '/'
    if not os.path.exists(path):
        os.mkdir(path)
    
    # Cycle to create the trials:
    for i in range(len(events)):
        if(ignore_trial[i] != 0):
            # Compute trial
            if(resampling):
                start_trial_1 = events[i, 0]
                
                if(i != len(events) - 1): start_trial_2 = events[(i + 1), 0]
                else: start_trial_2 = -1
                
                # Extract trial
                tmp_trial = trials[:, start_trial_1:start_trial_2]
                
                # Matrix for the resampling trial
                resample_trial = np.zeros((tmp_trial.shape[0], min_length))
                
                # Resample to min length
                for j in range(len(trials)):
                    channel = tmp_trial[j, :]
                    channel_resampled = resample(channel, min_length)
                    resample_trial[j, :] = channel_resampled
                    
                # Substitute the matrix
                tmp_trial = resample_trial
                
            else:
                start_trial = events[i, 0]
                tmp_trial = trials[:, start_trial:(start_trial + min_length)]
                
            if(envelope):
                a = 1
            
            # Extract label
            tmp_label = events[i, 2]  
            
            # Save the new mat
            tmp_path = path + str(counter) + '.mat'
            tmp_dict = {'trial':tmp_trial, 'label':tmp_label}
            savemat(tmp_path, tmp_dict)
            
            counter += 1
            
    return min_length
            
def computeEnvelope(x, downsampling = 1):
    if(downsampling != 1): tmp_envelope = np.zeros([x.shape[0], int(x.shape[1]/downsampling)])
    else: tmp_envelope = np.zeros(x.shape)
    
    # Cycle through channels
    for j in range(x.shape[0]):
        CSP_channel = x[j, :]
        
        # Envelope evaluation
        analytic_signal = scipy.signal.hilbert(CSP_channel)
        amplitude_envelope = np.abs(analytic_signal)
        
        # Downsampling
        if(downsampling != 1):  amplitude_envelope = scipy.signal.decimate(amplitude_envelope, downsampling)
        
        # Save the envelope
        # tmp_envelope[j, :] = amplitude_envelope[0:tmp_envelope.shape[2]]
        tmp_envelope[j, :] = amplitude_envelope
        
    return tmp_envelope