# Variational Autoencoder for EEG analysis
Branch used for personal experiments
 
This repository contains the code used for different pubblications. Different branch corresponds to the code used for specific articles or/and up to a specific moment in time. Below there is a list of branches with a short description for each one. The complete list of articles is at the bottom of the readme.
Index of the README.
* [List of branches](#list-of-branches)
* [List of papers](#list-of-papers)
* [Code General info](#code-general-info)


## List of branches
- main : default brach
- [code_ICT4AWE_2023](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/code_ICT4AWE_2023) : Code used for [[1]][vEEGNet_ver1] and [[2]][vEEGNet_ver2_preprint]. Contains the code for vEEGNet-ver1 and vEEGNet-ver2
- [Before_rewriting](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/Before_rewriting) : The branch contains the codes before a complete rewriting and reorganization of the repository
- [backup_after_dwt_implementation](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/backup_after_dwt_implementation) : As the name suggests, it contains the codes after a backup of the repository after the implementation of DTW loss function.
- [hvEEGNet_paper](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/hvEEGNet_paper) : Code used for [[3]][hvEEGNet_preprint]. Contains the code for vEEGNet-ver3 and hvEEGNet
- [jesus-experiment](https://github.com/jesus-333/Variational-Autoencoder-for-EEG-analysis/tree/jesus-experiment) : Branch for personal experiments

## Code General Info
The code is organized as a python package inside the **library** folder. Inside **library** there are several submodules, each one dedicated to a specific purpose. Therefore the structure of the import will have the following syntax. 
```python
from library.subpackage import stuff_from_subpackage
```

The complete list of subpackages is :
- *model* : contains the definition of all the models developed. More information in the [model README](library/README/README_model.md)
- *training* : contains function to train the various model. More information in the [training README](library/README/README_training.md). If you use wandb there are version of the training scripts with support for this awesome library. If you don't use wandb I highly recommend you try using it.
- *config* : contains the configuration for the creation of models and training. More info in the [model REAMDE](library/README/README_model.md) and the [training README](library/README/README_training.md)
- *dataset*: contains functions used to download dataset and to perform some basic preprocess. More info in the [dataset README](library/README/README_dataset.md)

## List of papers
<details>
  <summary>Click to expand!</summary>
 
  - [[1]][vEEGNet_ver1] Zancanaro, A., Zoppis, I., Manzoni, S., & Cisotto, G. (2023). vEEGNet: A New Deep Learning Model to Classify and Generate EEG. In Proceedings of the 9th International Conference on Information and Communication Technologies for Ageing Well and e-Health, ICT4AWE 2023, Prague, Czech Republic, April 22-24, 2023 (Vol. 2023, pp. 245-252). Science and Technology Publications.
  - [[2]][vEEGNet_ver2_preprint] Zancanaro, A., Cisotto, G. Zoppis, I., & Manzoni, S. (2023). vEEGNet: A New Deep Learning Model to Classify and Generate EEG., vEEGNet: learning latent representations to reconstruct EEG raw data via variational autoencoders (under review) ([preprint][vEEGNet_ver2_preprint] on ResearchGate)
  - [[3]][hvEEGNet_preprint]  Cisotto, G., Zancanaro, A., Zoppis, I., & Manzoni, S. (2023). hvEEGNet: exploiting hierarchical VAEs on EEG data for neuroscience applications (under review) ([preprint][hvEEGNet_preprint] on ResearchGate)
  
</details>

[vEEGNet_ver1]: https://www.scitepress.org/Papers/2023/119908/119908.pdf
[vEEGNet_ver2_preprint]: https://www.researchgate.net/publication/375867809_vEEGNet_learning_latent_representations_to_reconstruct_EEG_raw_data_via_variational_autoencoders
[hvEEGNet_preprint]: https://www.researchgate.net/publication/375868326_hvEEGNet_exploiting_hierarchical_VAEs_on_EEG_data_for_neuroscience_applications