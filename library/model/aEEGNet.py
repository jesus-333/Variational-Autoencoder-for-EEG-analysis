"""
Implementation of a EEGNet + Attention
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch
from torch import nn

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class aEEGNet(nn.Module):

	def __init__(self, config : dict):
		super().__init__()

	def forward(self, x):
		return x

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class attention_module(nn.Module):

	def __init__(self, config : dict):
		super().__init__()

        if config['use_ff_for_qkv'] :
            pass
            # self.ff_query = nn.Linear(config[])
        else:


	def forward(self, x, external_query = None):
		return x

