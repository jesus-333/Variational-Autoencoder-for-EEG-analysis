"""
Compute the wasserstein distance between the distribution encoded by the latent space

@author : Alberto (Jesus) Zancanaro
@organization : University of Padua
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Import libraries

from scipy.stats import wasserstein_distance_nd
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Settings

average_dist_1 = 0
average_dist_2 = 0

std_dist_1 = 1
std_dist_2 = 1

dist_type_1 = 'guass'
dist_type_2 = 'gauss'

n_samples = 300
dist_dimension = 100

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Generate samples

u_samples = np.zeros((n_samples, dist_dimension))
v_samples = np.zeros((n_samples, dist_dimension))

for i in range(n_samples) :
    if dist_type_1 == 'gauss' :
        u_samples[i] = np.random.normal(loc = average_dist_1, scale = std_dist_1, size = dist_dimension)
    elif dist_type_1 == 'poisson' :
        u_samples[i] = np.random.poisson()
    elif dist_type_1 == 'rayleigh' :
        u_samples[i] = np.random.rayleigh(scale = std_dist_1, size = dist_dimension)

    if dist_type_2 == 'gauss' :
        v_samples[i] = np.random.normal(loc = average_dist_2, scale = std_dist_2, size = dist_dimension)
    elif dist_type_2 == 'poisson' :
        v_samples[i] = np.random.poisson()
    elif dist_type_2 == 'rayleigh' :
        v_samples[i] = np.random.rayleigh(scale = std_dist_2, size = dist_dimension)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Compute wasserstein distance
        
# Compute distance between distribution
distance_between_distribution = wasserstein_distance_nd(u_samples, v_samples)
print("The distance between distribution is {}".format(distance_between_distribution))
