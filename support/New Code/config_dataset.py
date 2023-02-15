"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config related to dataset download and preprocess
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_moabb_dataset_config():
    dataset_config = dict(
        # Frequency filtering settings
        filter_data = False,
        fmin = 0,
        fmax = 125,
        # Resampling settings
        resample_data = True,
        resample_freq = 128,
        # Other
        n_classes = 4,
        subjects_list = [1,2,3,4,5,6,7,8,9],
        normalize_trials = True,
        percentage_split = 0.9,
    )

    return dataset_config

def get_artifact_dataset_config(type_dataset, folder_to_save = 'v2'):
    dataset_config = dict(
        # Version
        type_dataset = type_dataset,
        # Frequency filtering settings
        filter_data = True,
        fmin = 0,
        fmax = 125,
        # Resampling settings
        resample_data = True,
        resample_freq = 256,
        # Other
        folder_to_save = folder_to_save
    )

    return dataset_config
