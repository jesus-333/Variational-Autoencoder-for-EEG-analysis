"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the function to compute accuracy and other metrics
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Imports

import numpy as np
import torch
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score, recall_score, f1_score, confusion_matrix, multilabel_confusion_matrix
import os
import wandb

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Function to compute the metrics 

def compute_metrics(model, loader, device):
    true_label, predict_label = compute_label(model, loader, device)

    metrics_list = compute_metrics_from_labels(true_label, predict_label)

    return metrics_list

def compute_label(model, loader, device):
    """
    Method create to compute the label in a dataloader with the model class
    """
    
    model.eval()
    model.to(device)

    true_label_list = []
    predict_label_list = []

    for batch in loader:
        # Get data and true labels
        x = batch[0].to(device)
        tmp_true_label = batch[1]
        
        # Compute predicted labels
        tmp_predict_label = model.classify(x)
        
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
    # confusion_matrix = multilabel_confusion_matrix(true_label, predict_label)
    confusion_matrix = compute_multiclass_confusion_matrix(true_label, predict_label)

    return accuracy, cohen_kappa, sensitivity, specificity, f1, confusion_matrix


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

def compute_multiclass_confusion_matrix(true_label, predict_label):
    # Create the confusion matrix
    confusion_matrix = np.zeros((4, 4))
    
    # Iterate through labels
    for i in range(len(true_label)):
        # Get the true and predicts labels
        # Notes that the labels are saved as number from 0 to 3 so can be used as index
        tmp_true = true_label[i]
        tmp_predict = predict_label[i]
        
        confusion_matrix[tmp_true, tmp_predict] += 1
    
    # Normalize between 0 and 1
    confusion_matrix /= len(true_label)
    
    
    return confusion_matrix

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%% Function to compute the metrics given data/model

def compute_metrics_given_path(model, loader_list, path, device = 'cpu'):
    """
    Function to compute the metrics given a path containing the pth file with the weights of the network.
    For each pth file load the weight and computer the metrics
    """

    file_list = os.listdir(path)
    
    metrics_per_file = dict(
        accuracy = [],
        cohen_kappa = [], 
        sensitivity = [],
        specificity = [],
        f1 = [], 
        confusion_matrix = []
    )
    for file in file_list:
        print(file)

        # Crete path to the file 
        complete_path = path + '/' + file
        
        # Load weights
        model.load_state_dict(torch.load(complete_path, map_location=torch.device('cpu')))

        for loader in loader_list:
            accuracy, cohen_kappa, sensitivity, specificity, f1, confusion_matrix = compute_metrics(model, loader, device)
            
            metrics_per_file['accuracy'].append(accuracy)
            metrics_per_file['cohen_kappa'].append(cohen_kappa)
            metrics_per_file['sensitivity'].append(sensitivity)
            metrics_per_file['specificity'].append(specificity)
            metrics_per_file['f1'].append(f1)
            metrics_per_file['confusion_matrix'].append(confusion_matrix)

    return metrics_per_file
