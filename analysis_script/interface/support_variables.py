"""
Contains a list of variables that can be imported from scripts

@author: Alberto Zancanaro (Jesus)
@organization: Luxembourg Centre for Systems Biomedicine (LCSB)
@contact : alberto.zancanaro@uni.lu
@date: October 2025
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import matplotlib
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

COLORMAPS = np.sort(list(matplotlib.colormaps))

subj = [2, 5]
SUBJ_ARRAY = [f"Subj {i}" for i in subj]

CHANNELS_D2A = np.array(['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz'])
