# `library/training/wandb_training.py`

## Purpose
The W&B-integrated training entry points referenced by `library/README/README_training.md` as `train_wandb_model_name()`. Wraps `train_generic`'s model-agnostic training with a `wandb.init(...)` context, config logging, and model-artifact upload — this is the file users are pointed to when they want tracked training instead of calling `train_generic` directly. Also contains several `main_*`/`get_config_dict_for_*` convenience functions that assemble ready-to-use config dicts for specific models.

## Key functions
### `train_wandb_V1(model_name: str, dataset_config: dict, train_config: dict, model_config: dict) -> model`
- For the "repo downloads its own data" path (same dataset-loading flow as `train_generic.train_and_test_model`). Opens a `wandb.init` run, builds a model artifact (`train_config['model_artifact_name'] + '_trained'`), sets `train_config['wandb_training'] = True`, calls `train_generic.train_and_test_model(model_name, dataset_config, train_config, model_config, model_artifact)`, logs the artifact, returns the trained model.

### `train_wandb_V2(model_name, train_config, model_config, train_dataset, validation_dataset, dataset_config=None) -> model`
- For "bring your own dataset" use (per its docstring: datasets must implement the `EEG_dataset` interface from `library.dataset.dataset_time`). Validates config via `check_config.check_model_config_hvEEGNet` (if `'hvEEGNet'` is in `model_name`) and `check_config.check_train_config`. Builds dataloaders directly from the given datasets, constructs the model via `train_generic.get_untrained_model`, builds optimizer/scheduler itself (duplicating logic in `train_generic.train_and_test_model` rather than reusing it), then calls the lower-level `train_generic.train(...)` (not `train_and_test_model`) inside a `wandb.init` context, logging the artifact at the end.
- Has a commented-out finetuning block (`# TODO add proper finetuning`) showing an abandoned attempt to load pretrained weights before training (`path_weight = 'Saved Model/hvEEGNet_d2a_block/S3_block_250/model_50.pth'`).

### `main_EEGNet_classifier()`
- Example/manual-run entry: builds configs for subject 3 and calls `train_wandb_V1('EEGNet', ...)`.

### `get_config_dict_for_vEEGNet(subj_list: list) -> (dataset_config, train_config, model_config)`
- Assembles ready-to-use configs for vEEGNet: MOABB dataset config for the given subjects, vEEGNet training config, and model config with hardcoded `C=22`, `hidden_space_dimension=64`, `type_encoder=0`, `type_decoder=0`; computes `T` from `trial_start`/`trial_end`/sampling frequency. Syncs `train_config['measure_metrics_during_training']`/`['use_classifier']` from `model_config['use_classifier']`.

### `main_vEEGNet_default(subj_list: list)` / `main_vEEGNet(dataset_config, train_config, model_config)`
- `main_vEEGNet_default` builds configs via `get_config_dict_for_vEEGNet` then calls `main_vEEGNet`, which just calls `train_wandb_V1('vEEGNet', ...)`. Docstring notes `main_vEEGNet` (taking configs directly) is also the one used from Google Colab, since Colab notebooks can't easily edit other local files to change config functions.

### `get_config_dict_for_hvEEGNet_shallow(subj_list: list) -> (dataset_config, train_config, model_config)`
- Same pattern as the vEEGNet version but for `hvEEGNet_shallow` (hardcoded `type_decoder=0`, `parameters_map_type=0`).

### `main_hvEEGNet_shallow(dataset_config, train_config, model_config)`
- Calls `train_wandb_V1('hvEEGNet_shallow', ...)`.

### `get_config_dict_for_classifier_v1(subj: int, path_weight_hvEEGNet: str)`
- **Incomplete function**: builds `dataset_config`, `hvEEGNet_config` (discarding its own `train_config`), a `classifier_config`, and starts building a `train_config` for the classifier (`model_artifact_name = 'classifier'`) — but the function has **no `return` statement**, so it currently returns `None` regardless of caller expectations. Appears to be mid-edit/unfinished (likely intended to eventually load `path_weight_hvEEGNet` into the hvEEGNet encoder and return configs for training the classifier on top).

## Dependencies
- Internal: `library.training.train_generic`, `library.check_config`, `library.config.{config_model, config_dataset, config_training}`.
- External: `wandb`, `torch`.

## Notes
- `train_wandb_V2` duplicates optimizer/scheduler construction logic that already exists inside `train_generic.train_and_test_model` — if `train_and_test_model`'s optimizer setup changes (e.g. switching optimizers, adding gradient clipping), this function must be updated in parallel or it will silently diverge.
- `get_config_dict_for_classifier_v1` is unfinished (no `return`); do not assume it works — check before using it to build a classifier training config.
- The module's `if __name__ == '__main__':` block only has `pass` and a commented-out call to `main_EEGNet_classifier()` — running this file directly currently does nothing.
