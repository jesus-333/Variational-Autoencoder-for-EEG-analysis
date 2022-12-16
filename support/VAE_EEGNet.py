import torch
from torch import nn
import numpy as np

import matplotlib.pyplot as plt

from scipy.io import savemat, loadmat

from support_EEGNet import getParametersEncoder, getParametersDecoder
from DynamicNet import DynamicNet

#%% EEGNet VAE class

class EEGNetVAE(nn.Module):
    
    def __init__(self, C, T, hidden_space_dimension, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        self.encoder = EEGNetEncoder(C, T, hidden_space_dimension, False, tracking_input_dimension)
        output_shape_conv_encoder = self.encoder.conv_encoder.output_shape
        
        # shape_original_input = (C, T)
        # self.decoder = EEGNetDecoder(hidden_space_dimension, self.encoder.flatten_neurons, output_shape_conv_encoder, shape_original_input, False, tracking_input_dimension)
        
        tracking_input_dimension_list = self.encoder.conv_encoder.tracking_input_dimension_list
        self.decoder = EEGNetDecoderV2(hidden_space_dimension, self.encoder.flatten_neurons, tracking_input_dimension_list)
        
        if(print_var): print("Number of trainable parameters (VAE) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x):
        mu, log_var = self.encoder(x)
        
        z = self.reparametrize(mu, log_var)
        
        x_mean, x_std = self.decoder(z)
        
        return x_mean, x_std, mu, log_var
    
    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the stochastic part of the model
        
        sigma = torch.exp(0.5 * log_var)
        z = torch.randn(size = (mu.size(0),mu.size(1)))
        z = z.type_as(mu) # Setting z to be .cuda when using GPU training 
        
        return mu + sigma*z
    
#%% EEGNet Encoder class
    
class EEGNetEncoder(nn.Module):
    
    def __init__(self, C, T, hidden_space_dimension, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        # Convolutional section
        parameters = getParametersEncoder(C = C, T = T)
        self.conv_encoder = DynamicNet(parameters, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        
        # Fully connect section
        self.flatten_neurons = self.conv_encoder.output_shape[1] * self.conv_encoder.output_shape[2] * self.conv_encoder.output_shape[3]
        self.fc_encoder = nn.Linear(self.flatten_neurons, 2 * hidden_space_dimension)
        
        
    def forward(self, x):
        x = self.conv_encoder(x)
        x = x.view([x.shape[0], -1])
        x = self.fc_encoder(x)
        
        mu = x[:, 0:int((x.shape[1]/2))]
        log_var = x[:, int((x.shape[1]/2)):]
        
        return mu, log_var
    
#%% EEGNet Decoder class (with DynamicNet)

class EEGNetDecoder(nn.Module):
    
    def __init__(self, hidden_space_dimension, flatten_layer_dimension, conv_shape, shape_original_input, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        # Fully connect layer
        self.fc_decoder = nn.Linear(hidden_space_dimension, flatten_layer_dimension)
        
        # Convolutional layer
        parameters = getParametersDecoder(C = conv_shape[2], T = conv_shape[3])
        self.decoder = DynamicNet(parameters, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        self.shape_input_conv_decoder = list(conv_shape)
        self.shape_input_conv_decoder[0] = -1
        
        # Upsample layer
        self.upsample_layer = nn.Upsample(size = shape_original_input)
        
        
    def forward(self, z):
        x = self.fc_decoder(z)
        x = torch.reshape(x, self.shape_input_conv_decoder)
        
        x = self.decoder(x)
        
        x = self.upsample_layer(x)
        
        return x

#%% EEGNet Decoder class (classical implementation)

class EEGNetDecoderV2(nn.Module):
    
    def __init__(self, hidden_space_dimension, flatten_layer_dimension, tracking_input_dimension_list):
        super().__init__()
        
        # Fully connect layer
        self.fc_decoder = nn.Linear(hidden_space_dimension, flatten_layer_dimension)
    
        self.shape_input_conv_decoder = list(tracking_input_dimension_list[-1][1])
        self.shape_input_conv_decoder[0] = -1
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Convolutional section of the decoder 
        drop_1 = nn.Dropout(p = 0.5)
        upsample_layer_1 = nn.Upsample(size = tracking_input_dimension_list[-2][1][2:4])
        act_1 = nn.ELU()
        batch_norm_1 = nn.BatchNorm2d(num_features = 16)
        cnn_layer_1 = nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (1,1), bias = False)
        
        cnn_layer_2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 16, kernel_size = (1,32), padding=(0, 16), groups = 16, bias = False)
        
        drop_3 = nn.Dropout(p = 0.5)
        upsample_layer_3 = nn.Upsample(size = tracking_input_dimension_list[1][1][2:4])
        act_3 = nn.ELU()
        batch_norm_3 = nn.BatchNorm2d(num_features = 16)
        cnn_layer_3 = nn.ConvTranspose2d(in_channels = 16, out_channels = 8, kernel_size = (tracking_input_dimension_list[0][1][2], 1), groups = 8, bias = False)
        
        batch_norm_4 = nn.BatchNorm2d(num_features = 8)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # List of module (use for debugging)
        module_list = nn.ModuleList()
        module_list.append(drop_1)
        module_list.append(upsample_layer_1)
        module_list.append(act_1)
        module_list.append(batch_norm_1)
        module_list.append(cnn_layer_1)
        
        module_list.append(cnn_layer_2)
        
        module_list.append(drop_3)
        module_list.append(upsample_layer_3)
        module_list.append(act_3)
        module_list.append(batch_norm_3)
        module_list.append(cnn_layer_3)

        module_list.append(batch_norm_4)
        # module_list.append(cnn_layer_4)
        
        self.conv_decoder = nn.Sequential(*module_list)
        
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
        # Layer to compute the mean
        self.mean_output_layer = nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = (1, 64), padding=(0, 32), bias = False)
        
        # Evaluate the final dimensio of the output
        dumb_shape = [i for i in self.shape_input_conv_decoder] # Copy the shape in a new list (i.e. the original list is not modified)
        dumb_shape[0] = 1
        dumb_output_shape = self.conv_decoder(torch.rand(dumb_shape)).shape

        # Layer to compute the std 
        self.std_output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 8, out_channels = 1, kernel_size = (1, 64), padding=(0, 32),bias = False),
            # nn.Linear(dumb_output_shape[-1], 1),
            nn.Linear(512, 1),
            nn.Flatten(),
            nn.Linear(22, 1)
        )


    def forward(self, z):
        x = self.fc_decoder(z)
        x = torch.reshape(x, self.shape_input_conv_decoder)
        
        x = self.conv_decoder(x)

        x_mean = self.mean_output_layer(x)
        x_std = self.std_output_layer(x)
        # x_std = torch.ones(1).to(x_mean.device)
        
        return x_mean, x_std
    
#%% Classifier

class CLF_V1(nn.Module):
    
    def __init__(self, n_input_neurons):
        super().__init__()
        
        # ORIGINAL STRUCTURE
        self.classifier = nn.Sequential(
            nn.Linear(n_input_neurons, 64),
            nn.ELU(),
            nn.Linear(64, 4),
            nn.LogSoftmax(dim = 1)
        )
        
        # self.classifier = nn.Sequential(
        #     nn.Linear(n_input_neurons, 4),
        #     nn.LogSoftmax(dim = 1)
        # )
        
    def forward(self, x): 
        return self.classifier(x)
        

#%% VAE + Classifier

class EEGFramework(nn.Module):
    
    def __init__(self, C, T, hidden_space_dimension, use_reparametrization_for_classification = False, print_var = False, tracking_input_dimension = False):
        super().__init__()
        
        # VAE Definition
        self.vae = EEGNetVAE(C = C, T = T, hidden_space_dimension = hidden_space_dimension, print_var = print_var, tracking_input_dimension = tracking_input_dimension)
        
        # Classifier (Discriminator) definition        
        if(use_reparametrization_for_classification): self.classifier = CLF_V1(hidden_space_dimension)
        else: self.classifier = CLF_V1(hidden_space_dimension * 2)
        
        self.use_reparametrization_for_classification = use_reparametrization_for_classification
        
        self.trainable_parameter_VAE = sum(p.numel() for p in  self.vae.parameters() if p.requires_grad)
        self.trainable_parameter_classifier = sum(p.numel() for p in  self.classifier.parameters() if p.requires_grad)
        
        if(print_var): print("Number of trainable parameters (Classifier) = ", sum(p.numel() for p in  self.classifier.parameters() if p.requires_grad), "\n")
        if(print_var): print("Number of trainable parameters (VAE + Classifier) = ", sum(p.numel() for p in self.parameters() if p.requires_grad), "\n")
        
    def forward(self, x):
        # Foraward pass through VAE
        x_r_mean, x_r_std, mu, log_var = self.vae(x)
                
        # (OPTIONAL) Reparametrization for classification
        if(self.use_reparametrization_for_classification): z = self.vae.reparametrize(mu, log_var)
        else: z = torch.cat((mu, log_var), dim = 1)
        
        # Classification of VAE output
        label = self.classifier(z)
        
        # Return reconstructed input, mu and log variance vectors and label of the input.
        return x_r_mean, x_r_std, mu, log_var, label

    def classify(self, x, return_as_index = True):
        mu, log_var = self.vae.encoder(x)
        z = torch.cat((mu, log_var), dim = 1)
        label = self.classifier(z)

        if return_as_index:
            predict_prob = torch.squeeze(torch.exp(label).detach())
            label = torch.argmax(predict_prob, dim = 1)

        return label
