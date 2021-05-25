# -*- coding: utf-8 -*-
"""


@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Add support function folder to path
import sys
sys.path.insert(1, 'C:/Users/albi2/Documents/GitHub/Deep-Learning-For-EEG-Classification/support')

from jesus_network_V2 import convertArrayInTupleList

#%%

def getParameters(activation_function = 2, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    # C = 22
    # T = 512
    
    # F_1 = 8
    # D = 4
    # F_2 = 32
    # # F_2 = F_1 * D
    
    kernel_1 = (1, 33)
    kernel_2 = (C, 1)
    kernel_3 = (1, 16)
    kernel_4 = (1, 1)
    
    # kernel_1 = (1, 8)
    # kernel_2 = (C, 1)
    # kernel_3 = (1, 4)
    # kernel_4 = (1, 1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 4
    parameters["layers_ff"] = 1
    
    # parameters["activation_list"] = [2, 2, 2, 2, 9, 9]
    # parameters["activation_list"] = [-1, activation_function, -1, activation_function, 9, 9]
    parameters["activation_list"] = [12, activation_function, -1, activation_function, 9, 9]
    
    parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4]
    
    parameters["filters_list"] = [1, F_1, F_1 * D, F_1 * D, F_2]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    parameters["padding_list"] = [(0, int(kernel_1[1]/2)), [0,0], (0, int(kernel_3[1]/2)), [0,0]]
    
    # parameters["CNN_normalization_list"] = [True, True, False, True]
    # parameters["CNN_normalization_list"] = [True, True, True, True]
    
    # parameters["dropout_list"] = [-1, 0.5, -1, 0.5, -1]
    parameters["dropout_list"] = [-1, 0.5, -1, 0.5, -1, -1]
       
    parameters["pooling_list"] = [-1, [1, (1,4)], -1, [1, (1,8)]]
    
    parameters["groups_list"] = [1, F_1, F_1 * D, 1]
    
    parameters["bias_list"] = [False, False, False, False, True, True]
    
    parameters["neurons_list"] = [4]


    return parameters


#%%

def getParametersBasic():
    activation_function = 3
    C = 22
    T = 512
    
    F_1 = 8
    D = 2
    F_2 = 16
    
    parameters = getParameters(activation_function, C, T, F_1, D, F_2)
    
    return parameters

#%%

def getParametersHGD(activation_function = 3, C = 128, T = 3500):
    parameters = getParameters(activation_function, C, T)
    
    return parameters
    
#%%

def getParametersGlobalPool(activation_function = 3, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    parameters = getParameters(activation_function, C, T,  F_1, D, F_2)
    
    parameters["pooling_list"] = [-1, [1, (1,4)], -1, [2, (1,8)]]
    parameters["dropout_list"] = [-1, 0.5, -1, -1, -1, -1]
    
    return parameters

def getParametersGlobalPoolCW(activation_function = 3, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    kernel_1 = (1, 32)
    kernel_3 = (1, 16)
    kernel_4 = (1, 1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 3
    parameters["layers_ff"] = 1
    
    # parameters["activation_list"] = [2, 2, 2, 2, 9, 9]
    parameters["activation_list"] = [activation_function, activation_function, activation_function, 10, 10]
    
    parameters["kernel_list"] = [kernel_1, kernel_3, kernel_4]
    
    parameters["filters_list"] = [1, F_1, F_1 * D, F_2]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    parameters["padding_list"] = [(0, int(kernel_1[1]/2)), (0, int(kernel_3[1]/2)), [0,0]]
    
    parameters["CNN_normalization_list"] = [True, False, True]
    
    # parameters["dropout_list"] = [-1, 0.5, -1, 0.5, -1]
    parameters["dropout_list"] = [-1, -1, -1, -1, -1]
       
    parameters["pooling_list"] = [-1, -1, [2, (1,8)]]
    
    parameters["groups_list"] = [1, F_1, 1]
    
    parameters["bias_list"] = [False, False, False, True, True]
    
    parameters["neurons_list"] = [4]

    
    # parameters["pooling_list"] = [-1, [1, (1,4)], -1, [2, (1,8)]]
    # parameters["dropout_list"] = [-1, 0.5, -1, -1, -1, -1]
    
    return parameters
    
#%%

def getParametersAlexEEG(activation_function = 3, C = 22, T = 512, F_1 = 60, D = 2, F_2 = 120):
    # C = 22
    # T = 512
    
    # F_1 = 8
    # D = 4
    # F_2 = 32
    # # F_2 = F_1 * D
    
    kernel_1 = (1, 16)
    kernel_2 = (1, 8)
    kernel_3 = (C, 1)
    kernel_4 = (1, 8)
    kernel_5 = (1, 4)
    kernel_6 = (1, 1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 6 
    parameters["layers_ff"] = 1
    
    ac = activation_function
    parameters["activation_list"] = [ac, ac, ac, ac, ac, ac, 9, 9]
    parameters["activation_list"] = [-1, -1, ac, -1, -1, ac, 9, 9]
    
    parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6]
    
    parameters["filters_list"] = [1, F_1, F_1, F_1 * D, F_1 * D, F_1 * D, F_2]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    
    parameters["padding_list"] = [(0, int(kernel_1[1]/2)), (0, int(kernel_2[1]/2)), [0,0], (0, int(kernel_4[1]/2)), (0, int(kernel_5[1]/2)), [0,0]]
    
    parameters["CNN_normalization_list"] = [True, True, True, True, False, True]
    # parameters["CNN_normalization_list"] = [True, True, True, True]
    
    # parameters["dropout_list"] = [-1, 0.5, -1, 0.5, -1]
    parameters["dropout_list"] = [-1, 0.5, 0.5, -1, 0.5, 0.5, -1, -1]
       
    parameters["pooling_list"] = [-1, [1, (1,2)], [1, (1,4)], -1, [1, (1,4)], [1, (1,8)]]
    
    # parameters["groups_list"] = [1, 1, F_1, F_1 * D, 1, 1]
    
    parameters["bias_list"] = [False, False, False, False, False, False, True, True]
    
    parameters["neurons_list"] = [4]


    return parameters

#%%


    
    