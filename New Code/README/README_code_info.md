# Table of Contents
* [Requirements](#requirements): List of all the python library required to run the scripts of this repository
* [List of files](#list-of-files): Section with a brief description of all the files and functions of the repository (TO COMPLETE)

# Requirements
* Python: Any python distribution. Scripts can be launched from both shell and IDE. 
* PyTorch: used for model definition and training. See [PyTorch website](https://pytorch.org/) for installation information.
* moabb: the acronym stands for Mother of all BCI Benchmark. It is used to download the dataset. Link to [Github repo][moabb_github] and [website][moabb_website]. You could install the library through pip with the followin command: `pip install MOABB`
* wandb (OPTIONAL): if you have install wandb you can use it to track the training of your model. See [here](https://docs.wandb.ai/quickstart) for installation information.

# List of files
This readme contain the complete list of all the files and functions in the repository

* `EEGNet.py`: Implementation of EEGNet ([ArXiv][EEGNet_Arxiv], [Journal][EEGNet_Journal]). 
    * `EEGNet`: (Class) Implementation of EEGNet
    * `get_activation_function`: (function) Support function that return a list of activation function in PyTorch
    * `count_trainable_parameters` : (function) Support function that return the number of trainable parameters of a PyTorch layer
* `MBEEGNet.py`: Implementation of MBEEGNet ([mdpi][MBEEGNet_mdpi], [pubmed][MBEEGNet_pubmed]). 
    * `MBEEGNet`: (Class) Implementation of MBEEGNet
    * `MBEEGNet_Classifier`: (Class) Implementation of MBEEGNet + Classification layer
* `config_model.py`: Contain the definition for the dictionary config of the various model

<!-- Reference Link -->
[EEGNet_Journal]: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
[EEGNet_Arxiv]: https://arxiv.org/abs/1611.08024
[MBEEGNet_mdpi]: https://www.mdpi.com/2079-6374/12/1/22
[MBEEGNet_pubmed]: https://pubmed.ncbi.nlm.nih.gov/35049650/
[moabb_github]: https://github.com/NeuroTechX/moabb
[moabb_website]: http://moabb.neurotechx.com/docs/index.html
