"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function to compute accuracy and other metrics
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports
import torch

from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, f1_score

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

    return true_label, predict_label


def compute_metrics(true_label, predict_label):
    accuracy    = accuracy_score(true_label, predict_label)
    cohen_kappa = cohen_kappa_score(true_label, predict_label)
    sensitivity = recall_score(true_label, predict_label)
    f1          = f1_score(true_label, predict_label)

    return accuracy, cohen_kappa, sensitivity, f1
