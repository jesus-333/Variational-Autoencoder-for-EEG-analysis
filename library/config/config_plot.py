
def get_config_plot_preprocess_random_trial():
    config = dict(
        # - - - - - - - - - - 
        # Config related to data
        n_trials_to_plot = 3,
        ch_to_plot = 'C3',
        t_start = 0, # Time in seconds when the eeg recording starts
        t_end = 6, # Time in seconds when the eeg recording end 
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 12,
        show_fig = True,
        save_plot = True,
    )

    return config

def get_config_plot_preprocess_ERS():
    config = dict(
        # Data config
        ch_to_plot = ['C3', 'Cz', 'C4'],
        label_to_plot = [0,1,2],
        # label_to_plot = [1,2,3],
        # Figure config
        figsize = (18, 12),
        fontsize = 12,
        show_fig = True,
        save_plot = True,
    )

    return config


def get_config_plot_preprocess_average_stft():
    config = dict(
        # - - - - - - - - - - 
        # Config related to data
        ch_to_plot = 'C3',
        t_start = 0, # Time in seconds when the eeg recording starts
        t_end = 6, # Time in seconds when the eeg recording end 
        band_start = 8,
        band_end = 13,
        n_trials_to_average = 20,
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 12,
        show_fig = True,
        save_plot = True,
    )

    return config


def get_config_latent_plot_representation():
    config = dict(
        sample_from_distribution = True,
        # - - - - - - - - - - 
        # tsne config
        perplexity = 30,
        n_iter = 1000,
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 12,
        show_fig = True,
        save_plot = False,
    )

    return config

def get_config_set_of_trials():
    config = dict(
        # - - - - - - - - - - 
        # Trials info
        idx_start = 166,
        idx_end = 172,
        idx_ch = 6,
        trial_length = 4,
        # - - - - - - - - - - 
        # Figure config
        figsize = (15, 10),
        fontsize = 15,
        add_trial_line = True
    )
    
    return config