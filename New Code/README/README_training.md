# Training General information

Each model has its own related traing files called `train_model_name.py`. Each one of this files have the following functions (customized according to the model)
    - `train_and_test_model`:
    - `train`
    - `test`
    - `train_epoch`
    - `validation_epoch`
    - `check_train_config`


# Standard Training 

## Python/IPython shell
After open a python/ipython shell you simply declare the dictionary used for training

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
