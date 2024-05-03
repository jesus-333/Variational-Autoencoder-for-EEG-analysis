# Table of Contents
* [General informatin about training](#training-general-information): General information about train the model in this repository
* [How to train a model](#how-to-train-a-model): Example on how to train a model
	* [Python/IPython shell](#python-ipython-shell)	
	<!-- * [Python IDE](#python-ide) -->
	* [Shell](#shell)
* [Integration with wandb](#integration-with-wandb): How to use wandb to track your training
* [Training config for each model](#training-config-for-each-model): Training parameters specific for each model

# Training General information

Each model has its own related traing files called `train_model_name.py` (e.g. for the MBEEGNet model there is `train_MBEEGNet.py`). Each one of this files have the following functions (customized according to the model):
* `train_and_test_model(dataset_config, train_config, model_config, model_artifact)`: This is the function to use for training and testing the model. It requires 3 config dictionaries:
	* `dataset_config`: contain the settings related to the data and dataset (e.g. train/validation split, preprocess info like filtering frequencies etc). For more information about dataset read the [README_dataset](README_dataset.md).
	* `train_config`: contain the settings used during the training. See section [Training config for each model](#training-config-for-each-model) for more information about the parameters used in the training of each model.
	* `model_config`: contain the config related to the model you want to train. For more information about models read [README_model](README_model.md). 	
* `train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler = None)`: Function used to train the model. If you do not want to use the `train_and_test_model` (maybe because you prefer to use your own data) use this function instead. The input are the following:
	* `model`: PyTorch model to train.
	* `loss_function`: loss function used during the training.
	* `optimzer`: optimizer (e.g. ADAM) to used during the training.
	* `loader_list`: list with 2 [Dataloader](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#iterate-through-the-dataloader). The first Dataloader must contain the training data and the second one the validation data.
	* `train_config`: dictionary with parameters to used during the training. See section [Training config for each model](#training-config-for-each-model) for more information.
* `test( ... )`: Test the model with the test data. NOT FULLY IMPLEMENTED FOR ALL MODEL. 
* `train_epoch( ... )`: execute a single training epoch during training (i.e. forward and backward pass, optimization step).
* `validation_epoch( ... )`: execute a validation epoch (i.e. compute loss, and and possibly other metrics, for the validation data but DO NOT perform the backward pass and the optimization step).
* `check_train_config( ... )`: function used to check if there are problem with the training config

Note that you are supposed to use the first 3 functions of this list (and possibly `check_train_config()` if you want to check your train config). You are not supposed to use `train_epoch()` and `validation_epoch()`. 
Also the use of `train_and_test_model()` is encouraged over using the individual `train` and `test` functions. This is because the `train_and_test_model` function provides an integrated and complete solution for downloading data, creating a model, training it and testing it

# How to train a model 

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
You could simply call this functions without having to write the entire parameter dictionaries by yourself. If you need to change parameters you could simply use these functions to get a full dictionary and modify only the parameter(s) you need.

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
model_config['temporal_kernel_3'] = (1, 8)

train_MBEEGNEt.train_and_test_model(dataset_config, train_config, model_config)

```

<!-- ## Python IDE -->
<!-- For train the models in a Python IDE (e.g. spyder, pycharm), simply open the training script you need (e.g. `train_MBEEGNet.py`) and run it. The configs used for dataset, training and model will be those specified in the files `config_model.py`, `config_dataset.py` and `config_training.py` -->


## Shell
For train the models from a shell (e.g. fish, Windows power shell etc) simply run the command:
```
python path/to/file/train_model_name.py
```

For example, if you opened the shell in the folder containing the file `train_MBEEGNet.py` you could simply run the command `python train_MBEEGNet.py` from the shell.

Also in this case, the configs used for dataset, training and model will be those specified in the files `config_model.py`, `config_dataset.py` and `config_training.py`

# Integration with wandb

If you have install the [wandb](https://wandb.ai/) python library you can use the functions inside `wandb_trainig.py` script to train the network and keep track of the traing results through wandb.
For each model there is a function called `train_wandb_model_name()` (e.g. for MBEEGNet `train_wandb_MBEEGNet`). The call of the function is the same of any `train_and_test_model()` and require the same input argument (i.e. `dataset_config`, `train_config`, `model_config`)

Here, there is an example with MBEEGNet:
```python
import config_model as cm
import config_dataset as cd
import config_training as ct

import wandb_trainig

C = 22
T = 512 

dataset_config = cd.get_moabb_dataset_config()
train_config = ct.get_config_MBEEGNet_training()
model_config = cm.get_config_MBEEGNet_classifier(C, T, 4)

wandb_trainig.train_wandb_MBEEGNet(dataset_config, train_config, model_config)
```

To install wandb you can use one of the following commands:
* pip
```
pip install wandb
```
* conda
```
conda install -c conda-forge wandb
```

# Training config for each model

⚠️ Important disclaimer ⚠️

Most of the training parameters remain valid for all models. The training parameters differences between the various models usually are some specific hyperparameter (e.g. the hyperparameters that multiply the loss of the vae)

## MBEEGNet (Classifier)

### Training config
```python
config = dict(
	# Training settings
	batch_size = 30,                    
	lr = 1e-3,                          # Learning rate (lr)
	epochs = 500,                       # Number of epochs to train the model
	use_scheduler = False,              # Use the lr scheduler (exponential lr scheduler)
	lr_decay_rate = 0.995,              # Parameter of the lr exponential scheduler
	optimizer_weight_decay = 1e-2,      # Weight decay of the optimizer

	# Support stuff (device, log frequency etc)
	device = "cuda" if torch.cuda.is_available() else "cpu",  # device (i.e. cpu/gpu) used to train the network. 
	epoch_to_save_model = 5,				  # Save the weights of the network every n epochs
	path_to_save_model = 'TMP_Folder',			  # Path where to save the model. If the path does not exist the function save the weights in the folder you are currently in
	measure_metrics_during_training = True,			  # Measure accuracy and other metric during training
	print_var = True,					  # Print information in the console during the training
	
	# (OPTIONAL) wandb settings
	wandb_training = False,             	# If True track the model during the training with wandb
	project_name = "MBEEGNet",		# Name of the wandb project where the runs are saved
	model_artifact_name = "MBEEGNet_test",	# Name of the artifact used to save the models
	log_freq = 1,				# Specifies how often to log data to wandb (e.g. 1 = every epoch, 2 = every to epoch etc)
	notes = "",
)
```

### Other notes
The model expects input in 4 dimensions, batch size x 1 x eeg channels x time samples. This is because convolutions are done via [Conv2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).

⚠️ The dimension with the 1 it's the dimension used for the convolutions channels (which have nothing to do with EEG channels!!!)
