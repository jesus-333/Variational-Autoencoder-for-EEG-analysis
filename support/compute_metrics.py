"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function to compute accuracy and other metrics
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports
import torch

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def compute_label(eeg_framework, loader):
    """
    Method create to compute the label in a dataloader with the eeg_framework class
    """
    
    eeg_framework.eval()

    true_label_list = []
    predict_label_list = []

    for batch in loader:
        # Get data and true labels
        x = batch[0]
        tmp_true_label = batch[1]
        
        # Compute predicted labels
        tmp_predict_label = eeg_framework.classify(x)
        
        # Save predicted and true labels
        true_label_list.append(tmp_true_label)
        predict_label_list.append(tmp_predict_label)
    
    # Convert list in tensor
    true_label = torch.cat(true_label_list)
    predict_label = torch.cat(predict_label_list)

