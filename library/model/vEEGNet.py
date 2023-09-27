"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Implementation of vEEGNet model using PyTorch
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#%% Imports
import torch
from torch import nn

from . import EEGNet, MBEEGNet, Decoder_EEGNet, support_function

"""
%load_ext autoreload
%autoreload 2
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

class vEEGNet(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

        self.check_model_config(config)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Create Encoder

        # Convolutional section
        if config["type_encoder"] == 0:
            self.cnn_encoder = EEGNet.EEGNet(config['encoder_config']) 
        elif config["type_encoder"] == 1:
            self.cnn_encoder = MBEEGNet.MBEEGNet(config['encoder_config']) 
        
        # Get the size and the output shape after an input has been fed into the encoder
        # This info will also be used during the encoder creation
        n_input_neurons, decoder_ouput_shape = self.encoder_shape_info(config['encoder_config']['C'], config['encoder_config']['T'])

        # Layer to "sample" from the latent space
        self.sample_layer = support_function.sample_layer(decoder_ouput_shape, config, config['hidden_space'])
        self.parameters_map_type = config['parameters_map_type']

        if self.parameters_map_type == 0: self.hidden_space = n_input_neurons
        elif self.parameters_map_type == 1: self.hidden_space = config['hidden_space'] # Note that in this case the variable is not used and is put here only as a placehoder to avoid error with old code

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  

        # (OPTIONAL) classifier
        self.use_classifier = config['use_classifier']
        if self.use_classifier:

            self.clf = nn.Sequential(
                nn.Linear(self.hidden_space * 2, config['n_classes']),
                nn.LogSoftmax(dim = 1)
            )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Create Decoder
        # Note that the config used for the encoder  are also used for the decoder (stuff like the size of convolutional and pooling kernel)
        
        # Information specific for the creation of the decoder
        config['encoder_config']['dimension_reshape'] = decoder_ouput_shape
        config['encoder_config']['hidden_space'] = self.hidden_space
        config['encoder_config']['parameters_map_type'] = config['parameters_map_type']
        
        # For the decoder we use the same type of the encoder
        # E.g. if the encoder is EEGNet also the decoder will be EEGNet
        if config["type_encoder"] == 0:
            if config['type_decoder'] == 0:
                self.decoder = Decoder_EEGNet.EEGNet_Decoder_Upsample(config['encoder_config']) 
            elif config['type_decoder'] == 1:
                self.decoder = Decoder_EEGNet.EEGNet_Decoder_Transpose(config['encoder_config']) 
        elif config["type_encoder"] == 1:
            # TODO Implement MBEEGNet decoder 
            self.decoder = MBEEGNet(config['encoder_config']) 

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    def forward(self, x):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -                 
        # Encoder section
        x = self.cnn_encoder(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        # Reparametrization and sampling
        # Note that the reparametrization trick is done by inside the sample_layer
        z, z_mean, z_log_var = self.sample_layer(x)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Decoder
        x_r = self.decoder(z)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  
        # Classifier
        
        # Note that since the input of the classifier are vector their size is "Batch x n_elements"
        # So the concatenation will be along the dimension with index 1

        if self.use_classifier: 
            z = torch.cat([z_mean, z_log_var], dim = 1)
            if self.parameters_map_type == 0: z = z.flatten(1)
            predicted_label = self.clf(z)

            return x_r, z_mean, z_log_var, predicted_label
        else:
            return x_r, z_mean, z_log_var
        
    def reconstruct(self, x, no_grad = True):
        if no_grad:
            with torch.no_grad():
                output = self.forward(x)
        else:
            output = self.forward(x)
                
        return output[0]

    def generate(self):
        # Sample laten space (normal distribution)
        z = torch.randn(1, self.hidden_space)
        
        # Generate EEG sample
        x_g = self.decoder(z)

        return x_g

    def encoder_shape_info(self, C, T):
        """
        Compute the total number of neurons for the feedforward layer
        Compute the shape of the input after pass through the convolutional encoder

        Note that the computation are done for an input with batch size = 1
        """
        # Create fake input
        x = torch.rand(1, 1, C, T)

        # Pass the fake input inside the encoder
        x = self.cnn_encoder(x)

        # Compute the number of neurons needed for the feedforward layer
        input_neurons = len(x.flatten())
        
        # Get the shape at the output of the convolutional encoder
        # The dimension in position 0 is the batch dimension and it is set to -1 to ignore it during the reshape
        encoder_ouput_shape = list(x.shape)
        encoder_ouput_shape[0] = -1

        return input_neurons, encoder_ouput_shape

    def check_model_config(self, config : dict):
        # Check type encoder
        if config["type_encoder"] == 0: print("EEGNet encoder selected")
        elif config["type_encoder"] == 1: print("MBEEGNet encoder selected")
        else: raise ValueError("type_encoder must be 0 (EEGNET) or 1 (MBEEGNet)")

        # Check type decoder 
        if config["type_decoder"] == 0: print("Upsample decoder selected")
        elif config["type_decoder"] == 1: print("Transpose decoder selected")
        else: raise ValueError("type_decoder must be 0 (Upsample) or 1 (Transpose)")

    def classify(self, x, return_as_index = True):
        """
        Directly classify an input by returning the label (return_as_index = True) or the probability distribution on the labels (return_as_index = False)
        """
        
        x_r, z_mean, z_log_var, label, = self.forward(x)

        if return_as_index:
            predict_prob = torch.squeeze(torch.exp(label).detach())
            label = torch.argmax(predict_prob, dim = 1)

        return label
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 


