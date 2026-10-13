# `library/training/train_generic.py`

## Purpose
The central, model-agnostic training orchestrator. It dispatches to the right model constructor, loss function, and per-model `train_epoch`/`validation_epoch` pair based on a `model_name` string, then runs the full training loop (epochs, checkpointing, optional W&B logging). This is the lowest common denominator that `wandb_training.py` and the dataset-specific `training_scripts/*.py` build on top of.

## Key functions

### `train_and_test_model(model_name, dataset_config, train_config, model_config, model_artifact=None) -> model`
- End-to-end pipeline when using the repo's own D2a-download path: loads train/validation/test datasets via `library.dataset.preprocess.get_dataset_d2a(dataset_config)`, builds `DataLoader`s, instantiates the model via `get_untrained_model`, builds the loss via `get_loss_function`, creates an `AdamW` optimizer and optional `ExponentialLR` scheduler, then calls `train(...)`.
- Reads from `dataset_config`: `use_stft_representation` (if true, patches `model_config['C']`, `['T']`, `['depth_first_layer']` from the actual data shape).
- Reads from `train_config`: `batch_size`, `device`, `lr`, `optimizer_weight_decay`, `use_scheduler`, `lr_decay_rate`, `path_to_save_model`, `wandb_training`, `log_freq`.
- The `test(...)` call at the end is commented out with a `# TODO`.

### `train(model, loss_function, optimizer, loader_list, train_config, lr_scheduler=None, model_artifact=None) -> model`
- The actual epoch loop, usable directly if you already have a model/dataloaders (bypassing `train_and_test_model`, e.g. for custom/external datasets — see `wandb_training.train_wandb_V2`).
- Calls `check_config.check_train_config(train_config, model_artifact)` first.
- Picks the correct `train_epoch`/`validation_epoch` functions via `get_train_and_validation_function(model)`.
- Each epoch: runs train + validation epoch, saves `model_BEST.pth` whenever validation loss improves, saves a periodic snapshot every `train_config['epoch_to_save_model']` epochs, optionally steps the LR scheduler, optionally computes classification metrics (`measure_metrics_during_training` + `use_classifier`), optionally prints progress, optionally logs everything to W&B (`wandb_training` flag) including uploading saved checkpoints as artifact files.
- Checkpoints are saved to `{train_config['path_to_save_model']}/model_BEST.pth` and `model_{epoch}.pth` via plain `torch.save(model.state_dict(), ...)`.

### `test(model, test_loader, config)`
- Prints metrics (via `library.metrics.compute_metrics`) both for the current (end-of-training) weights and after reloading `model_BEST.pth`. Not wired into `train_and_test_model` (see TODO above) and only partially used elsewhere.

### `get_untrained_model(model_name: str, model_config: dict)`
- String-keyed factory. Recognized `model_name` values: `'EEGNet'`, `'MBEEGNet'`, `'vEEGNet'`, `'hvEEGNet_shallow'`, `'hvEEGNet_V2'`, `'classifier_v1'`, `'ChWi_autoencoder'`. Raises `ValueError` for anything else (the error message text only lists the first four, which is stale/incomplete relative to the actual `elif` chain).

### `get_loss_function(model_name, config=None)`
- `'EEGNet'`/`'MBEEGNet'`/`'classifier_v1'` → `torch.nn.NLLLoss()`.
- `'vEEGNet'`/`'ChWi_autoencoder'` → `loss_function.vEEGNet_loss(config)`.
- `'hvEEGNet_shallow'`/`'hvEEGNet_V2'` → `loss_function.hvEEGNet_loss(config)`.
- Same `ValueError` pattern as above for unknown names.

### `get_train_and_validation_function(model)`
- Dispatches on `str(type(model))` substring matching (not `isinstance`) to pick the right `(train_epoch, validation_epoch)` pair from `train_EEGNet`, `train_vEEGNet`, `train_hvEEGNet`, `train_classifier`, `train_ChWi`. Both `EEGNet` and `MBEEGNet` models use `train_EEGNet`'s functions (shared classifier-style epoch loop). Both `hvEEGNet_shallow` and `hvEEGNet_V2` use `train_hvEEGNet`'s functions.

## Dependencies
- Internal: `library.training.wandb_support`, `library.metrics`, `library.training.loss_function`, `library.dataset.preprocess` (imported in a `try/except` — if this submodule fails to import, `train_and_test_model` will not work but the rest of the module still loads), `library.config.{config_model,config_dataset,config_training}`, `library.check_config`, `library.model.{EEGNet,MBEEGNet,vEEGNet,hvEEGNet,hvEEGNet_V2,classifier,ChWi}`, and the sibling `train_*` modules (`train_EEGNet`, `train_vEEGNet`, `train_hvEEGNet`, `train_classifier`, `train_ChWi`).
- External: `torch`, `wandb`, `pprint`, `os`, `sys`.

## Notes
- This file is the single place that enumerates "all models the training pipeline knows about" — when adding a new model you must update `get_untrained_model`, `get_loss_function`, and `get_train_and_validation_function` together, plus add a new `train_<model>.py` if its epoch loop differs.
- Dispatch by stringified `type(model)` in `get_train_and_validation_function` is fragile (depends on exact class path text, e.g. `'EEGNet.EEGNet'` is checked but the actual classifier class used elsewhere is `EEGNet_Classifier` — the substring check works because `'EEGNet.EEGNet'` is still a substring of `'...EEGNet.EEGNet_Classifier'`, but this is easy to break by renaming/moving classes).
- `test()` is effectively dead code in the main flow (call commented out in `train_and_test_model`); it hardcodes the `'model_BEST.pth'` filename and moves everything to `'cpu'`.
