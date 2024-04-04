"""
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

Contain the config related to dataset download and preprocess
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_moabb_dataset_config(subjects_list = [1,2,3,4,5,6,7,8,9], use_stft_representation = False):
    """
    Configuration for the download of dataset 2a through moabb
    """

    dataset_config = dict(
        # Frequency filtering settings
        filter_data = False,    # If True filter the data
        filter_type = 0,        # 0 Bandpass, 1 lowpass, 2 highpass (used only if filter_data is True)
        fmin = 0.5,             # Used in bandpass and highpass (used only if filter_data is True)
        fmax = 50,              # Used in bandpass and lowpass (used only if filter_data is True)
        filter_method = 'iir',  # Filter settings (used only if filter_data is True)
        iir_params = dict(ftype = 'cheby2', order = 20, rs = 30), # Filter settings (used only if filter_data is True and filter_method is iir)

        # Resampling settings
        resample_data = False,  # If true resample the data
        resample_freq = 128,    # New sampling frequency (used only if resample_data is True)

        # Trial segmentation
        trial_start = 2,    # Time (in seconds) when the trial starts. Keep this value
        trial_end = 6,      # Time (in seconds) when the trial end. Keep this value
        use_moabb_segmentation = False,

        # Split in train/test/validation
        seed_split = 42,                        # Seed for the random function used for split the dataset. Used for reproducibility
        percentage_split_train_test = -1,       # For ALL the data select the percentage for training and for test. -1 means to use the original division in train and test data
        percentage_split_train_validation = 0.9, # For ONLY the training data select the percentage for train and for validation

        # Other
        n_classes = 4,                  # Number of labels. For datset 2a is equal to 4. (IGNORE)
        subjects_list = subjects_list,  # List of the subjects of dataset 2a to download

        # Stft settings (IGNORE)(NOT USED)
        use_stft_representation = use_stft_representation,
        channels_list = ['C3', 'Cz', 'C4'], # List of channel to transform with STFT. Ignore.
        normalize = 0, # If different to 0 normalize the data during the dataset creation. Ignore and kept to 0
        train_trials_to_keep = None, # Boolean list with the same length of the training set, BEFORE THE DIVISION with training and validation, that indicate with trial kept for the training.
        # normalization_type = 1, # 0 = no normalization, 1 = ERS normalization (NOT IMPLEMENTED)
    )
    
    if dataset_config['use_stft_representation']: 
        dataset_config['stft_parameters'] = get_config_stft()
    else:
        del dataset_config['channels_list']

    return dataset_config

def get_config_stft():
    config = dict(
        sampling_freq = 250,
        nperseg = 50,
        noverlap = 40,
        # window = ('gaussian', 1),
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
