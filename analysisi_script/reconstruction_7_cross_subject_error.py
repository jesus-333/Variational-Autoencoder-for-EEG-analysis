"""
Computation of the average reconstruction error for each subject for each epoch across repetition
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import torch
import pickle

from library.config import config_dataset as cd
from library.analysis import support

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

if len(sys.argv) > 1:
    tot_epoch_training = sys.argv[1]
    subj_to_use = sys.argv[2]
    model_name = sys.argv[3]
else:
    tot_epoch_training = 80
    subj_to_use = 2
    model_name = 'hvEEGNet_shallow'
    # model_name = 'vEEGNet'

subject_to_test_list = np.delete(np.arange(9) + 1, np.where(np.arange(9) + 1 == subj_to_use))
repetition_list = np.arange(20) + 1
epoch_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
epoch_list = [15, 20, 40]
use_test_set = False
batch_size = 96
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

for subj_to_test in subject_to_test_list:
    recon_loss_results = dict()
    
    print("Subj: ", subj_to_test)
    recon_loss_results[subj_to_test] = dict()
    
    valid_repetition_per_epoch = dict()
    for epoch in epoch_list: valid_repetition_per_epoch[epoch] = 0
    
    # Get the datasets
    dataset_config = cd.get_moabb_dataset_config([subj_to_test])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset , model_hv = support.get_dataset_and_model(dataset_config, model_name)
    
    for repetition in repetition_list:
        print("\tRep: ", repetition)
        if use_test_set == False:
            if subj_to_test == 4 and (repetition == 4 or repetition == 6): continue
            if subj_to_test == 5 and repetition == 19: continue
            if subj_to_test == 8 and repetition == 12: continue
        
        for epoch in epoch_list:
            if use_test_set: 
                dataset = test_dataset
                string_dataset = 'test'
            else: 
                dataset = train_dataset
                string_dataset = 'train'
    
            try:
                # Load model weight
                if model_name == 'hvEEGNet_shallow':
                    path_weight = 'Saved Model/repetition_hvEEGNet_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj_to_use, repetition, epoch)
                elif model_name == 'vEEGNet':
                    path_weight = 'Saved Model/repetition_vEEGNet_DTW_{}/subj {}/rep {}/model_{}.pth'.format(tot_epoch_training, subj_to_use, repetition, epoch)
                model_hv.load_state_dict(torch.load(path_weight, map_location = torch.device('cpu')))
                
                # Compute loss
                tmp_recon_loss = support.compute_loss_dataset(dataset, model_hv, device, batch_size) / 1000
            except:
                print("Fail to load weight subj {} epoch {} rep {}".format(subj_to_use, epoch, repetition))
                continue
            
            # Check if there are nan during the computation.
            if np.sum(np.isnan(tmp_recon_loss)) > 0:
                print("Trovato il nan per test subj {} con pesi subj {} (epoch {} rep {})".format(subj_to_test, subj_to_use, epoch, repetition))
                continue
            else:
                valid_repetition_per_epoch[epoch] += 1
    
            if epoch not in recon_loss_results[subj_to_test]:
                recon_loss_results[subj_to_test][epoch] = tmp_recon_loss
            else:
                recon_loss_results[subj_to_test][epoch] += tmp_recon_loss
    
            # Save the results for each repetition
            if model_name == 'hvEEGNet_shallow':
                path_save = 'Saved Results/repetition_hvEEGNet_{}/{}/subj {}/cross_subject/'.format(tot_epoch_training, string_dataset, subj_to_use)
            elif model_name == 'vEEGNet':
                path_save = 'Saved Results/repetition_vEEGNet_DTW_{}/{}/subj {}/cross_subject/'.format(tot_epoch_training, string_dataset, subj_to_use)
            os.makedirs(path_save, exist_ok = True)
    
            # path_save_pickle = path_save + 'recon_error_{}_rep_{}.pickle'.format(epoch, repetition)
            # pickle_out = open(path_save_pickle, "wb")
            # pickle.dump(tmp_recon_loss , pickle_out)
            # pickle_out.close()
    
            path_save_npy = path_save + 'recon_error_cross_subj_{}_to_{}_epoch_{}_rep_{}.npy'.format(subj_to_use, subj_to_test, epoch, repetition)
            np.save(path_save_npy, tmp_recon_loss)
    
    #%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # Average accross repetition and save the results
    
    for epoch in epoch_list:
        
        if model_name == 'hvEEGNet_shallow':
            path_save = 'Saved Results/repetition_hvEEGNet_{}/{}/subj {}/cross_subject/'.format(tot_epoch_training, string_dataset, subj_to_use)
        elif model_name == 'vEEGNet':
            path_save = 'Saved Results/repetition_vEEGNet_DTW_{}/{}/subj {}/cross_subject/'.format(tot_epoch_training, string_dataset, subj_to_use)
    
        # path_save_pickle = path_save + 'recon_error_{}_average.pickle'.format(epoch)
        # pickle_out = open(path_save, "wb")
        # pickle.dump(recon_loss_results[subj][epoch] / valid_repetition_per_epoch[epoch] , pickle_out)
        # pickle_out.close()
    
        path_save_npy = path_save + 'recon_error_subj_{}_to_{}_epoch_{}_average.npy'.format(subj_to_use, subj_to_test, epoch)
        np.save(path_save_npy, recon_loss_results[subj_to_test][epoch] / valid_repetition_per_epoch[epoch])
