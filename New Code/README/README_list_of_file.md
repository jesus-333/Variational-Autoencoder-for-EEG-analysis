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
