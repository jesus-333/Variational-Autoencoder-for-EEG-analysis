# Table of Contents
* [Requirements](#requirements): List of all the python library required to run the scripts of this repository
* [List of files](#list-of-files): Section with a brief description of all the files and functions of the repository (TO COMPLETE)

# Requirements
* Python: Any python distribution. Scripts can be launched from both shell and IDE. 
* PyTorch: used for models definition and training. See [PyTorch website](https://pytorch.org/) for installation information.
* moabb: the acronym stands for Mother of all BCI Benchmark. It is used to download the dataset. Link to [Github repo][moabb_github] and [website][moabb_website]. You could install the library through pip with the followin command: `pip install MOABB`. This is optional if you want to work with your own data.
* wandb: Use to track the training of models. See [here](https://docs.wandb.ai/quickstart) for installation information. If could still train the model without wandb but the library is still needed for code structure and the import on some files.

# List of files
This readme contain the complete list of all the files and functions in the repository

* `EEGNet.py`: Implementation of EEGNet ([ArXiv][EEGNet_Arxiv], [Journal][EEGNet_Journal]). 
    * `EEGNet`: (Class) Implementation of EEGNet
* `MBEEGNet.py`: Implementation of MBEEGNet ([mdpi][MBEEGNet_mdpi], [pubmed][MBEEGNet_pubmed]). 
    * `MBEEGNet`: (Class) Implementation of MBEEGNet
    * `MBEEGNet_Classifier`: (Class) Implementation of MBEEGNet + Classification layer
* `config_model.py`: Contain the definition for the dictionary config of the various model
* `support_function.py`: Contain minor function
    * `get_activation`: (Function) Function that returns a list of activation function in PyTorch
    * `get_dropout`: (Function) Function that return a dropout or a 2d dropout
    * `count_trainable_parameters`: count the number of trainable parameter in a PyTorch module

<!-- Reference Link -->
[EEGNet_Journal]: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
[EEGNet_Arxiv]: https://arxiv.org/abs/1611.08024
[MBEEGNet_mdpi]: https://www.mdpi.com/2079-6374/12/1/22
[MBEEGNet_pubmed]: https://pubmed.ncbi.nlm.nih.gov/35049650/
[moabb_github]: https://github.com/NeuroTechX/moabb
[moabb_website]: http://moabb.neurotechx.com/docs/index.html
