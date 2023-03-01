# Training General information

Each model has its own related traing files called `train_model_name.py`. Each one of this files have the following functions (customized according to the model):
* `train_and_test_model(dataset_config, train_config, model_config, model_artifact)`: This is the primary function to use for training and testing the model. It requires 3 config dictionaries:
	* `dataset_config`: contain the settings related to the data and dataset (e.g. train/validation split, preprocess info like filtering frequencies etc). For more information about dataset read the [README_dataset](README_dataset).
	* `train_config`: contain the settings used during the training. See section [Training config for each model](#training-config-for-each-model) for more information about the parameters used in the training of each model.
	* `model_config`: contain the config related to the model you want to train. For more information about models read [README_model](README_model.md). 	
* `train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler = None)`: Function used to train the model. If you do not want to use the `train_and_test_model` (maybe because you prefer to use your own data) use this function to train the model. The input are the following:
	* `model`: PyTorch model to train.
	* `loss_function`: loss function used during the training.
	* `optimzer`: optimizer (e.g. ADAM) to used during the training.
	* `loader_list`: list with 2 [Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#iterate-through-the-dataloader). The first Dataloader must contain the training data and the second one the validation data.
* `test( ... )`: Test the model with the test data. The test and the input 
* `train_epoch( ... )`: execute a single training epoch during training (i.e. forward and backward pass, optimization step).
* `validation_epoch( ... )`: execute a validation epoch (i.e. compute loss, and and possibly other metrics, for the validation data but DO NOT perform the backward pass and the optimization step).
* `check_train_config( ... )`: function used to check if there are problem with the training config

Note that you are supposed to use the first 3 function of this list (and possibly `check_train_config()` if you want to check your train config). You are not supposed to use `train_epoch()` and `validation_epoch()`. 
Also the use of `train_and_test_model()` is encouraged over using the individual `train` and `test` functions. This is because the `train_and_test_model` function provides an integrated and complete solution for downloading data, creating a model, training it and testing it

# Standard Training 

## Python/IPython shell
After open a python/ipython shell you simply declare the dictionary used for training

### Example with MBEEGNet

```python

dataset_config  = dict( ... )
train_config    = dict( ... )
model_config    = dict( ... )

train_MBEEGNEt.train_and_test_model(dataset_config, train_config, model_config)

```

Note that we already provide files with functions that return complete config dictionary for dataset, train and model. 
You could simply call this functions without having to write the entire parameter dictionaries by yourself. Also if you need to change parameters you could simply modify these functions.

Here, there is an example of training using the config functions:
```python
import config_model as cm
import config_dataset as cd
import config_training as ct

import train_MBEEGNet

C = 22
T = 512 

dataset_config = cd.get_moabb_dataset_config()
train_config = ct.get_config_MBEEGNet_training()
model_config = cm.get_config_MBEEGNet_classifier(C, T, 4)

train_MBEEGNEt.train_and_test_model(dataset_config, train_config, model_config)

```

# Train with wandb

If you have install the wandb python library you can use function inside `wandb_trainig.py` script to train the network and keep track of the traing results through wandb

# Training config for each model
