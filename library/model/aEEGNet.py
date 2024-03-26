"""
Implementation of a EEGNet + Attention
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torch import nn

import support_function as sf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class aEEGNet(nn.Module):

    def __init__(self, config : dict):
        super().__init__()

    def forward(self, x):
        return x

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class self_attention_module(nn.Module):

    def __init__(self, config : dict):
        """
        Compute the self-attention for a matrix of EEG signal. 
        To be precise the module require in input a tensor of shape B x D x C x T, with B = batch shape, D = # of depth map, C = # of EEG channels, T = # of EEG samples
        For each depth map the self attention is computed independently
        """
        super().__init__()
        
        # Create the head for the query/key/value
        self.q_head, self.k_head, self.v_head = sf.create_head(config)
        
        # Other variable
        self.normalize_qk = config['normalize_qk']
        self.v_head_output_length = config['v_head_output_length']
        
    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """
        Args :
            x : Input tensor of shape B x D x C x T, with B = batch shape, D = # of depth map, C = # of EEG channels, T = # of EEG samples (For each depth map the self attention is computed independently)
        Returns : 
            z_matrix : A matrix with shape B x D x C x self.v_head_output_length, where each depth map contains the computations of all the self-attention over the EEG channels
        """
        
        # Check the input
        if len(x.shape != 4) :
            raise ValueError("x shape is wrong. The shape of x must be B x D x C x T")
        
        # Create matrix to save results
        z_matrix = torch.zeros(x.shape[0], x.shape[1], x.shape[2], self.v_head_output_length)

        # Compute attention
        for i in range(x.shape[1]) :
            # Take the i-th depth map
            # Note that with this operation x_depth_map has shape B x C x T
            x_depth_map = x[:, i]

            # Compute query, key and value
            q = self.q_head(x_depth_map)
            k = self.k_head(x_depth_map)
            v = self.v_head(x_depth_map)
            
            # Compute and save attention values
            z_matrix[:, i] = sf.compute_attention_values(q, k, v, self.normalize_qk)

        return z_matrix


class external_attention_module(nn.Module):

    def __init__(self, config : dict):
        """
        Compute the attention of a matrix of EEG signal with an external query array.
        To be precise the module require in input two tensor :
        - 1 tensor of shape B x D x C x T, with B = batch shape, D = # of depth map, C = # of EEG channels, T = # of EEG samples
        - 1 tensor of shape B x N with B = batch shape and N = length of external query array
        For each depth map the self attention is computed independently
        """
        super().__init__()
        
        # Create the head for the key/value
        _, self.k_head, self.v_head = sf.create_head(config)

        # Create head for query
        if config['use_head_for_q'] : self.q_head = nn.Linear(config['external_query_input_length'], config['qk_output_length'])
        else : self.q_head = nn.Identity()
        
        # Other variable
        self.normalize_qk = config['normalize_qk']
        self.v_head_output_length = config['v_head_output_length']
        
    def forward(self, x : torch.Tensor, external_query : list[torch.Tensor]) -> torch.Tensor :
        """
        Args :
            x : Input tensor of shape B x D x C x T, with B = batch shape, D = # of depth map, C = # of EEG channels, T = # of EEG samples (For each depth map the self attention is computed independently)
            external_query : Tensor of shape B x 1 x N to use as query array for the attention computation. The dimension of size 1 is needed for the torch.bmm function
        Returns :
            z_matrix : A matrix with shape B x D x C x self.v_head_output_length, where each depth map contains the computations of all the self-attention over the EEG channels
        """
        
        # Check the input
        if len(x.shape != 4) :
            raise ValueError("x shape has a wrong number of dimension. The shape of x must be B x D x C x T")
        if len(external_query.shape != 3) :
            raise ValueError("external_query shape has a wrong number of dimension. The shape of external_query must be B x 1 x N")
        if external_query.shape[1] != 1 :
            raise ValueError("external_query shape is wrong. The shape must be B x 1 x N. Current shape is B x {} x N".format(external_query.shape[1]))
        
        # Create matrix to save results
        z_matrix = torch.zeros(x.shape[0], x.shape[1], 1, self.v_head_output_length)

        # Compute attention
        for i in range(x.shape[1]) :
            # Take the i-th depth map
            # Note that with this operation x_depth_map has shape B x C x T
            x_depth_map = x[:, i]

            # Compute query, key and value
            q = self.q_head(external_query)
            k = self.k_head(x_depth_map)
            v = self.v_head(x_depth_map)
            
            # Compute and save attention values
            z_matrix[:, i] = sf.compute_attention_values(q, k, v, self.normalize_qk)

        return z_matrix

