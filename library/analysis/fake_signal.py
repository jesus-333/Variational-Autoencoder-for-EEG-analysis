import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
#%%

def generate_sinusoide(config : dict):
    fs = config['f_sampling']
    phase = config['phase'] if 'phase' in config else 0
    amplitude = conifg['amplitude'] if 'amplitude' in config else 1
    t = np.linspace(config['t_start'], config['t_end'], int(fs * (config['t_end'] - config['t_start'])))
    
    x = amplitude * np.sin(2 * np.pi * config['f_sine'] * t + phase)

    return x, t

    

