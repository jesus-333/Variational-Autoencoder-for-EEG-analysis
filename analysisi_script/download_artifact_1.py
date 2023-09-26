"""
Computation of the reconstruction error for each trial
"""

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.insert(0, parent_directory)

import numpy as np
import wandb

#%% - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Parameters

# Number of total epoch of training for the model of the specific artifact
if len(sys.argv) > 1:
    tot_epoch = sys.argv[1]
    subj_to_download = sys.argv[2]
else:
    tot_epoch = 80
    subj_to_download = 6

version_list = np.arange(140, 238 + 1)
artifact_name = 'jesus_333/ICT4AWE_Extension/hvEEGNet_shallow_trained'
root_to_save_model = 'Saved Model/repetition_hvEEGNet_{}/'.format(tot_epoch)

run = wandb.init()

repetition_list = dict()

for i in range(len(version_list)):
    version = version_list[i]
    print(version)
    try:

        artifact = run.use_artifact('{}:v{}'.format(artifact_name, version), type='model')

        # Get the name of the run and split it by the char _
        # Note that the name of the run are something in the form hvEEGNet_shallow_DTW_subj_x_epoch_y_rep_z
        # With the split I obtained a list ['hvEEGNet', 'shallow', 'DTW', 'subj', 'x', 'epoch', 'y', 'rep', 'z']
        run_name = artifact.logged_by().name.split("_")

        subj  = int(run_name[4])
        epoch = int(run_name[6])
        rep   = int(run_name[8])

        # Take only the
        if epoch == tot_epoch and subj == subj_to_download:
            artifact_dir = artifact.download()
            if rep not in repetition_list: repetition_list[rep] = []

            repetition_list[rep].append(subj)

            path_to_save_model = '{}/subj {}/rep {}/'.format(root_to_save_model, subj, rep)
            os.makedirs(path_to_save_model, exist_ok = True)

            for file in artifact.files():
                old_model_weight_path = "{}/{}".format(artifact_dir, file.name)
                new_model_weight_path = "{}/{}".format(path_to_save_model, file.name)
                os.rename(old_model_weight_path, new_model_weight_path)

    except:
        print("Version {} has no artifact".format(version))


# Remove the log file of the run
os.system("rm -r wandb")
os.system("rm -r artifacts")

# Check that all the repetition have all the 9 subject
subj_list = [1,2,3,4,5,6,7,8,9]
for rep in repetition_list:
    subj_list_rep = repetition_list[rep]

    for subj in subj_list_rep:
        if subj not in subj_list_rep:
            print("Subject {} miss for repetition {}".format(subj, rep))
