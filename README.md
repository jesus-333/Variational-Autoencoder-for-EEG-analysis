# Variational Autoencoder for EEG analysis
 Variational Autoencoder for EEG analysis

Branch with the code used for the paper : **hvEEGNet: exploiting hierarchical VAEs on EEG data for neuroscience applications**

# Introduction
The code is organized as a python package inside the **library** folder. Inside **library** there are several submodules, each one dedicated to a specific purpose. Therefore the structure of the import will have the following syntax. 
```python
from library.subpackage import stuff_from_subpackage
```

The complete list of subpackages is :
- *model* : contains the definition of all the models developed. More information in the [model README](library/README/README_model.md)
- *training* : contains function to train the various model. More information in the [training README](library/README/README_training.md). If you use wandb there are version of the training scripts with support for this awesome library. If you don't use wandb I highly recommend you try using it.
- *config* : contains the configuration for the creation of models and training. More info in the [model REAMDE](library/README/README_model.md) and the [training README](library/README/README_training.md)
- *dataset*: contains functions used to download dataset and to perform some basic preprocess. More info in the [dataset README](library/README/README_dataset.md)



