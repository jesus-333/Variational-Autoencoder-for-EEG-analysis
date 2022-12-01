"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain function used to config the wandb sweep and during the sweep train
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

import wandb

import config_file as cf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Train function

def train_sweep(config = None):

    with wandb.init(project = config['wandb_config']['project_name'], job_type = "train_sweep", config = config, name = config['wandb_config']['run_name']) as run:
        pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Other function

def get_uniform_distribution(min:float, max:float) -> dict:
    tmp_dict = dict(
        distribution = 'uniform',
        min = min,
        max = max
    )

    return tmp_dict

