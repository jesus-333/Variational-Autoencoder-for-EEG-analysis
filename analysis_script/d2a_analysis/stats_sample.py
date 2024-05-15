"""
Compute the average and the standard deviation for all the data of dataset 2a for each subject and session
"""
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

from library.config import config_dataset as cd
from library.analysis import support

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

subj_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
string_to_print = ""

for i in range(len(subj_list)):
    subj = subj_list[i]
    
    # Get dataset
    dataset_config = cd.get_moabb_dataset_config([subj])
    dataset_config['percentage_split_train_validation'] = -1 # Avoid the creation of the validation dataset
    train_dataset, validation_dataset, test_dataset, _ = support.get_dataset_and_model(dataset_config, model_name = 'hvEEGNet_shallow')
    
    # Get data
    train_data = train_dataset.data.flatten()
    test_data = test_dataset.data.flatten()

    string_to_print += "Stats for S{}\n".format(subj)
    string_to_print += "\tSession 1 : Mean = {:.2f}   Std = {:.2f}\n\n".format(train_data.mean(), train_data.std())

print(string_to_print)
