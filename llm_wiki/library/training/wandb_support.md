# `library/training/wandb_support.py`

## Purpose
Small helpers for interacting with Weights & Biases: attaching files to an artifact, a one-off script to re-download and rename specific historical Ofner2017 training artifacts, and a helper to merge classification metrics into a training `log_dict` for W&B logging.

## Key functions
- `add_file_to_artifact(artifact, file_name)` — adds `file_name` to a W&B `artifact` object and also calls `wandb.save(file_name)` so the raw file is uploaded to the run as well as the artifact. Used throughout `train_generic.train` and `wandb_training.py` whenever a checkpoint is saved.
- `download_artifacts_Ofner2017(version_list: list)` — opens a `wandb.init()` run, and for each artifact version in `version_list` downloads `jesus_333/hvEEGNet_Ofner2017/hvEEGNet_Ofner2017_trained:v{version}`, reads `metadata['training_config']['name_training_run']`, and renames the downloaded folder to that name — with three hardcoded manual overrides (`version == 11/18/19`) for runs whose W&B name was fixed by hand after the fact and no longer matches the stored metadata.
- `update_log_dict_metrics(metrics_dict, log_dict, label)` — copies `accuracy`, `cohen_kappa`, `sensitivity`, `specificity`, `f1` from `metrics_dict` into `log_dict` under keys like `accuracy_{label}` — **incomplete**: the `specificity` and `f1` local variables are read from `metrics_dict` but only `accuracy_{label}`, `cohen_kappa_{label}`, `sensitivity_{label}` are actually written into `log_dict`; the function is truncated/cut off (the file ends right after the `sensitivity_{label}` line) and never adds `specificity_{label}` or `f1_{label}`.

## Dependencies
- External: `wandb`, `os`.

## Notes
- `update_log_dict_metrics` silently drops `specificity` and `f1` from what gets logged to W&B despite computing/reading them — likely an unfinished edit rather than intentional; a future agent touching metrics logging should extend this function to add the missing two `log_dict` entries if W&B dashboards are expected to show them.
- `download_artifacts_Ofner2017` is a one-off migration/bookkeeping script tied to specific hardcoded W&B artifact versions and project name (`jesus_333/hvEEGNet_Ofner2017`) — not a general-purpose utility.
