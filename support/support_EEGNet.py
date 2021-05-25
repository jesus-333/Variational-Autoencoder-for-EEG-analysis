"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)
"""

#%% Imports

from DynamicNet import convertArrayInTupleList

#%% EEGNet (Dataset 2a resampled at 128Hz)

def getParameters(activation_function = 2, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    
    kernel_1 = (1, 33)
    kernel_2 = (C, 1)
    kernel_3 = (1, 16)
    kernel_4 = (1, 1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 4
    parameters["layers_ff"] = 1
    
    parameters["activation_list"] = [-1, activation_function, -1, activation_function, 9, 9]
    
    parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4]
    
    parameters["filters_list"] = [1, F_1, F_1 * D, F_1 * D, F_2]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    parameters["padding_list"] = [(0, int(kernel_1[1]/2)), [0,0], (0, int(kernel_3[1]/2)), [0,0]]
    
    parameters["CNN_normalization_list"] = [True, True, False, True]
    
    parameters["dropout_list"] = [-1, 0.5, -1, 0.5, -1, -1]
       
    parameters["pooling_list"] = [-1, [1, (1,4)], -1, [1, (1,8)]]
    
    parameters["groups_list"] = [1, F_1, F_1 * D, 1]
    
    parameters["bias_list"] = [False, False, False, False, True, True]
    
    parameters["neurons_list"] = [4]


    return parameters



#%% Encoder EEGNet

def getParametersEncoder(activation_function = 2, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    
    kernel_1 = (1, 33)
    kernel_2 = (C, 1)
    kernel_3 = (1, 16)
    kernel_4 = (1, 1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 4
    parameters["layers_ff"] = 0
    
    parameters["activation_list"] = [-1, activation_function, -1, activation_function]
    
    parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4]
    
    parameters["filters_list"] = [1, F_1, F_1 * D, F_1 * D, F_2]
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    parameters["padding_list"] = [(0, int(kernel_1[1]/2)), [0,0], (0, int(kernel_3[1]/2)), [0,0]]
    
    parameters["CNN_normalization_list"] = [True, True, False, True]
    
    parameters["dropout_list"] = [-1, 0.5, -1, 0.5]
       
    parameters["pooling_list"] = [-1, [1, (1,4)], -1, [1, (1,8)]]
    
    parameters["groups_list"] = [1, F_1, F_1 * D, 1]
    
    parameters["bias_list"] = [False, False, False, False]
    
    parameters["add_flatten_layer"] = False
    
    return parameters

#%% Decoder EEGNet

def getParametersDecoder(activation_function = 2, C = 22, T = 512, F_1 = 8, D = 2, F_2 = 16):
    
    kernel_1 = (-1, -33)
    kernel_2 = (-128, -1)
    kernel_3 = (-1, -16)
    kernel_4 = (-1, -1)
    
    parameters = {}
    
    parameters["h"] = C
    parameters["w"] = T
    
    parameters["layers_cnn"] = 4
    parameters["layers_ff"] = 0
    
    parameters["activation_list"] = [-1, activation_function, -1, activation_function]
    parameters["activation_list"].reverse()
    
    parameters["kernel_list"] = [kernel_1, kernel_2, kernel_3, kernel_4]
    parameters["kernel_list"].reverse()
    
    parameters["filters_list"] = [1, F_1, F_1 * D, F_1 * D, F_2]
    parameters["filters_list"].reverse()
    parameters["filters_list"] = convertArrayInTupleList(parameters["filters_list"])
    
    parameters["padding_list"] = [(0, abs(int(kernel_1[1]/2))), [0,0], (0, abs(int(kernel_3[1]/2))), [0,0]]
    parameters["padding_list"].reverse()
    
    parameters["CNN_normalization_list"] = [True, True, False, True]
    parameters["CNN_normalization_list"].reverse()
    
    parameters["dropout_list"] = [-1, 0.5, -1, 0.5]
    parameters["dropout_list"].reverse()
    
    parameters["groups_list"] = [1, F_1, F_1 * D, 1]
    parameters["groups_list"].reverse()
    
    parameters["bias_list"] = [False, False, False, False]
    parameters["bias_list"].reverse()
    
    parameters["add_flatten_layer"] = False
    
    return parameters

    
    