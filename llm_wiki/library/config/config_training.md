# `library/config/config_training.py`

## Purpose
Factory functions for the `train_config` dict consumed by `library/training/train_generic.py` and the model-specific `train_*.py` / `wandb_training.py` files, plus one example wandb-sweep config. Covers classifier training (EEGNet/MBEEGNet) and vEEGNet/hvEEGNet (VAE) training.

## Key functions

### `get_config_classifier() -> dict`
Used to train EEGNet and MBEEGNet classifiers.
- Training: `batch_size=20`, `lr=1e-3`, `epochs=300`, `use_scheduler=False`, `lr_decay_rate=0.999` (exponential scheduler), `optimizer_weight_decay=1e-2`.
- Support: `device` (`'cuda'` if available else `'cpu'`), `epoch_to_save_model=5`, `path_to_save_model='TMP_Folder'`, `measure_metrics_during_training=True`, `repetition=1` (number of times to repeat the training), `use_classifier=True` (comment: "Do NOT CHANGE and keep True. Needed for metric computation in the train function (train_generic.py)"), `print_var=True`.
- wandb (optional): `wandb_training=False`, `project_name='hvEEGNet_extension'`, `model_artifact_name='artifact_name'`, `log_freq=1`, `name_training_run=None` (wandb assigns a random name if `None`), `notes=""`, `debug=True` (used to delete debug runs from wandb).

### `get_config_vEEGNet_training() -> dict`
Used for vEEGNet and (via `get_config_hierarchical_vEEGNet_training`) hvEEGNet training.
- Training: `batch_size=30`, `lr=1e-2`, `epochs=3`, `use_scheduler=True`, `lr_decay_rate=0.999`, `optimizer_weight_decay=1e-2`.
- Loss weighting: `alpha=1` (reconstruction-error multiplier), `beta=1` (KL multiplier), `gamma=1` (classification-error multiplier, only relevant if a classifier is attached — independent of `gamma_dtw` despite the similar name), `recon_loss_type=1` (0=L2, 1=SDTW, 2=SDTW-Divergence, 3=Block-SDTW, 4=Block-SDTW-Divergence), `block_size=128` (only if `recon_loss_type` is 3 or 4), `edge_samples_ignored=0` (samples ignored at trial edges during loss computation), `average_channels=False`, `average_time_samples=False`, `gamma_dtw=1` (soft-DTW soft-min steepness; closer to 0 → closer to true min).
- Support: `device`, `epoch_to_save_model=5`, `path_to_save_model='TMP_Folder'`, `use_classifier=False` (ignore unless model has a classifier), `measure_metrics_during_training=True` (ignored unless classifier present), `print_var=True`.
- wandb (optional): `wandb_training=False`, `project_name='TMP_Project'`, `model_artifact_name='TMP_NAME'`, `log_freq=1`, `name_training_run=None`, `notes=""`, `debug=True`.

### `get_config_hierarchical_vEEGNet_training() -> dict`
Thin wrapper: returns `get_config_vEEGNet_training()` unchanged — docstring notes hvEEGNet training requires the same config shape as standard vEEGNet.

### `get_config_sweep_EEGNet_classifier(metric_name, metric_goal) -> dict`
Builds a wandb sweep config for EEGNet-classifier hyperparameter search. Raises `ValueError` unless `metric_goal` is exactly `'maximize'` or `'minimize'`. Returns: `method='random'`, `metric={'name': metric_name, 'goal': metric_goal}`, and a `parameters` dict with search spaces for `epochs` (`[200,300,500]`, with a debug alternative `[1,2,3]` commented out), `lr_decay_rate` (`[1, 0.995]`), `batch_size` (`[30,40,50]`), `subjects_list` (one list per d2a subject, 1–9), `channels_list` (two alternatives: 3-channel motor-cortex set or a larger ~22-channel set), `window` (`['hann', ('gaussian',1), ('gaussian',2)]`), `kernel` (`[(3,3),(5,5),(7,7),(9,9)]`), `filter_1` (`[8,16,32]`), `D` (`[2,4,8,16]`), `prob_dropout` (`[0.3,0.4,0.5]`), `use_dropout_2d` (`[True, False]`).

## Dependencies
- Internal: none.
- External: `torch` (only to detect CUDA availability for the `device` key).

## Notes
- `use_classifier` and `measure_metrics_during_training` appear in both `get_config_classifier` and `get_config_vEEGNet_training` but mean "this run is classifier-driven" in the former and "ignore unless a classifier is attached to the VAE" in the latter — don't assume the same semantics when porting configs between model families.
- `debug=True` is the default in both training configs; a real (non-debug) training run must explicitly flip this to `False` or the corresponding wandb run may be deleted/flagged as debug.
- This README is cross-referenced by `library/README/README_training.md`, which documents the broader `train_and_test_model` / `train` / `test` API that consumes these configs — read that file together with this one for the full training workflow.
- `get_config_sweep_EEGNet_classifier`'s `channels_list` and `window` sweep values hardcode d2a-specific channel names (`'C3'`, `'Cz'`, etc.) and STFT window choices — reusing this sweep for a different dataset requires editing those lists.
