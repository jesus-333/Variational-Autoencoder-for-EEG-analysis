"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function to compute accuracy and other metrics
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, f1_score, confusion_matrix

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def compute_metrics(eeg_framework, loader, device):
    true_label, predict_label = compute_label(eeg_framework, loader, device)

    metrics_list = compute_metrics_from_labels(true_label, predict_label)

    return metrics_list

def compute_label(eeg_framework, loader, device):
    """
    Method create to compute the label in a dataloader with the eeg_framework class
    """
    
    eeg_framework.eval()
    eeg_framework.to(device)

    true_label_list = []
    predict_label_list = []

    for batch in loader:
        # Get data and true labels
        x = batch[0].to(device)
        tmp_true_label = batch[1]
        
        # Compute predicted labels
        tmp_predict_label = eeg_framework.classify(x)
        
        # Save predicted and true labels
        true_label_list.append(tmp_true_label)
        predict_label_list.append(tmp_predict_label)
        
    # Convert list in tensor
    true_label = torch.cat(true_label_list).cpu()
    predict_label = torch.cat(predict_label_list).cpu()

    return true_label, predict_label


def compute_metrics_from_labels(true_label, predict_label):
    accuracy    = accuracy_score(true_label, predict_label)
    cohen_kappa = cohen_kappa_score(true_label, predict_label)
    sensitivity = recall_score(true_label, predict_label, average = 'weighted')
    specificity = compute_specificity_multiclass(true_label, predict_label)
    f1          = f1_score(true_label, predict_label, average = 'weighted')

    return accuracy, cohen_kappa, sensitivity, specificity, f1


def compute_specificity_multiclass(true_label, predict_label, weight_sum = True):
    """
    Compute the average specificity 
    """

    binary_specificity_list = []
    weight_list = []

    for label in set(true_label):
        # Create binary label for the specific class
        tmp_true_label = (true_label == label).int()
        tmp_predict_label = (predict_label == label).int()
        
        # Compute specificity
        binary_specificity_list.append(compute_specificity_binary(tmp_true_label, tmp_predict_label))
        
        # (OPTIONAL) Count the number of example for the specific class
        if weight_sum: weight_list.append(int(tmp_true_label.sum()))
        else: weight_list.append(1)

    return np.average(binary_specificity_list, weights = weight_list)


def compute_specificity_binary(true_label, predict_label):
    # Get confusion matrix
    cm = confusion_matrix(true_label, predict_label)
    
    # Get True Negative and False positive
    TN = cm[1, 1]
    FP = cm[1, 0]
    
    # Compute specificity
    specificity = TN / (TN + FP)

    return specificity
