# -*- coding: utf-8 -*-
"""x
@author: Alberto Zancanaro (Jesus)
@organization: University of Padua (Italy)

"""

#%%

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from Tensorflow_VAE_Tutorial import CVAE

#%%

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

#%%
optimizer = tf.keras.optimizers.Adam(1e-4)
epochs = 10

# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

