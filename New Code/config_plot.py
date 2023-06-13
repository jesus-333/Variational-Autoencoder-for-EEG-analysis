
def get_config_plot_preprocess_random_trial():
    config = dict(
        # - - - - - - - - - - 
        # Config related to data
        n_trial_to_plot = 3,
        ch_to_plot = 'C3',
        t_start = 0, # Time in seconds when the eeg recording starts
        t_end = 6, # Time in seconds when the eeg recording end 
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 12,
        save_plot = True,
    )

    return config


def get_config_plot_preprocess_ERS():
    config = dict(
        # Figure config
        figsize = (18, 12),
        # Data config
        ch_to_plot = ['C3', 'Cz', 'C4'],
        # label_to_plot = [0,1,2]
        label_to_plot = [1,2,3],
        save_plot = True,
        fontsize = 12,
    )

    return config


def get_config_plot_preprocess_average_stft():
    config = dict(
        # - - - - - - - - - - 
        # Config related to data
        ch_to_plot = 'C3',
        t_start = 2, # Time in seconds when the eeg recording starts
        t_end = 6, # Time in seconds when the eeg recording end 
        band_start = 8,
        band_end = 13,
        n_trials_to_average = 20,
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 12,
        save_plot = True,
    )

    return config
