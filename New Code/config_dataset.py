"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config related to dataset download and preprocess
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_moabb_dataset_config(subjects_list = [1,2,3,4,5,6,7,8,9]):
    dataset_config = dict(
        # Frequency filtering settings
        filter_data = False,
        filter_type = 0, # 0 Bandpass, 1 lowpass, 2 highpass
        fmin = 0.5, # Used in bandpass and highpass
        fmax = 50, # Used in bandpass and lowpass
        filter_method = 'iir',
        iir_params = dict(ftype = 'cheby2', order = 20, rs = 30),
        # Resampling settings
        resample_data = False,
        resample_freq = 128,
        # Other
        use_stft_representation = True,
        n_classes = 4,
        subjects_list = subjects_list,
        normalize_trials = True, # TODO CHECK
        percentage_split_train_test = 0.7,  # For ALL the data select the percentage for training and for test. -1 means to use the original division in train and test data
        percentage_split_train_validation = 0.9, # For ONLY the training data select the percentage for train and for validation
        return_channels = True,
        trial_start = 2, # Time (in seconds) when the trial starts
        trial_end = 7, # Time (in seconds) when the trial end
        normalization_type = 1, # 0 = no normalization, 1 = ERS normalization
        use_moabb_segmentation = False,
    )

    return dataset_config

def get_config_stft():
    config = dict(
        sampling_freq = 250,
        nperseg = 50,
        noverlap = 40,
        window = 'hann',
    )

    return config

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
