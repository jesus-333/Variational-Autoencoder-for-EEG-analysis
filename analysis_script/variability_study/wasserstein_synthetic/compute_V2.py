"""
Compute the wasserstein distance between the distribution encoded by the latent space

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

import json
import numpy as np
from scipy.stats import wasserstein_distance_nd

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Load distribution_settings
path_distribution_settings = 'analysis_script/variability_study/wasserstein_synthetic/distribution_settings.json'
with open(path_distribution_settings, 'r') as j:
    distribution_settings = json.loads(j.read())

# MAtrix to save the data
distance_values = np.zeros((len(distribution_settings), len(distribution_settings)))

n_samples = 300
dist_dimension = 100

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Function to generate samples

def generate_samples(n_samples : int, dist_dimension : int, dist_type : str, parameters : dict) -> np.array :
    samples = np.zeros((n_samples, dist_dimension))

    for i in range(n_samples) :
        if dist_type == 'gauss' :
            samples[i] = np.random.normal(loc = parameters['mean'], scale = parameters['std'], size = dist_dimension)
        elif dist_type == 'exponential' :
            samples[i] = np.random.exponential(scale = parameters['mean'], size = dist_dimension)
        elif dist_type == 'uniform' :
            samples[i] = np.random.uniform(low = parameters['low'], high = parameters['high'], size = dist_dimension)
        else :
            raise ValueError("dist_type must be gauss, exponential or uniform")

    dist_description = '{} - '.format(dist_type)
    if dist_type == 'gauss' :
        dist_description += 'mean = {} - std = {}'.format(parameters['mean'], parameters['std'])
    elif dist_type == 'exponential' :
        dist_description += 'mean = {}'.format(parameters['mean'])
    elif dist_type == 'uniform' :
        dist_description += 'low = {} - high = {}'.format(parameters['low'], parameters['high'])

    return samples, dist_description


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute distances

dist_description_list = []

i = 0
for dist_1 in distribution_settings :
    print(dist_1)

    u_samples, dist_description = generate_samples(n_samples, dist_dimension, dist_1[0:-2], distribution_settings[dist_1])
    dist_description.append(dist_description)
    
    j = 0
    for dist_2 in distribution_settings :
        print('\t', dist_2)

        v_samples, _ = generate_samples(n_samples, dist_dimension, dist_2[0:-2], distribution_settings[dist_2])

        # Compute distance between distribution
        distance_between_distribution = wasserstein_distance_nd(u_samples, v_samples)

        distance_values[i, j] = distance_between_distribution

        j += 1

    i += 1
