# `library/check_config.py`

## Purpose
Validation/normalization layer for the plain-dictionary configs used throughout the repo (dataset config, training config, model config, federated-server config). Functions here mutate a config dict in place: they fill in missing keys with defaults (printing a message when they do so) and raise `ValueError` when a required combination of keys is missing or invalid. This is the de facto schema definition for the config dictionaries since there is no dataclass/pydantic model anywhere in the package.

## Key classes
None — this file is purely functional.

## Key functions

### `check_config_dataset(config: dict) -> None`
- Mutates `config` in place (no return value).
- Defaults `filter_data` to `False` if absent; if `True`, requires `fmin`/`fmax` to be present.
- Defaults `resample_data` to `False`; if `True`, requires a positive `resample_freq`.
- Defaults `use_moabb_segmentation` to `True`.
- Defaults `n_samples_to_use` to `-1` (meaning "use all").
- Defaults `trial_start`/`trial_end` to `-1`/`-1` if either is missing.
- **Bug**: the bounds check `if config['trial_start'] > 0 and config['trial_end'] > config['trial_end']` compares `trial_end` to itself — this condition is always `False`, so it never actually catches `trial_end <= trial_start`. The error message text also says "trial_end must be higher than trial_end" (typo, should say `trial_start`).
- Converts `config['train_trials_to_keep'] == []` to `None`.
- If `use_stft_representation` is present and `True`, requires a `stft_parameters` sub-dict.
- Has two dead commented-out blocks (forcing `use_moabb_segmentation` to `False`, and a `normalization_type` default) left in as historical notes.

### `check_train_config(train_config: dict, model_artifact=None) -> None`
- Defaults `epoch_to_save_model` to `1`.
- Defaults `wandb_training` to `False`. A commented-out check used to require `model_artifact` when wandb training was enabled — currently disabled.
- Defaults `path_to_save_model` to `"."` with a warning, otherwise creates the directory via `os.makedirs(..., exist_ok=True)` as a side effect.
- Defaults `measure_metrics_during_training` to `False`.
- Defaults `print_var` to `True`.
- `model_artifact` parameter is accepted but currently unused (the only use of it is inside the disabled commented-out block).

### `check_model_config_EEGNet(model_config: dict) -> None`
- Works around TOML's lack of `null`/`None` support: if `p_kernel_1`/`p_kernel_2` are `int` and `<= -1`, converts them to `None` (this is how the pooling layer in `EEGNet` is told to be skipped — see `library/model/EEGNet.py`).
- Converts the list-valued kernel keys (`c_kernel_1`, `c_kernel_2`, `c_kernel_3`, `p_kernel_1`, `p_kernel_2`) from `list` to `tuple` because TOML loads lists as Python lists, but PyTorch conv/pool layers expect `tuple` kernel sizes.

### `check_model_config_hvEEGNet(model_config: dict) -> None`
- Calls `check_model_config_EEGNet` on `model_config['encoder_config']`.
- Forces `encoder_config['flatten_output']` to `False` (hvEEGNet requires the encoder's spatial output, not a flattened vector), printing a notice if it had to override a `True` value.
- Validates `parameters_map_type` is in `[0, 3]`, raising `ValueError` otherwise.

### `check_server_config(server_config: dict) -> None`
- Used for the federated-learning server config (see `library/training/federated/server.py`).
- Defaults a missing `notes` key... but the code actually writes to `server_config['wandb_config']['notest']` (note the typo `notest` instead of `notes`) rather than fixing the checked key — likely a bug, since the config will still appear to be missing `notes` after this call.
- Similarly, if `name_training_run` is missing/empty, it sets `server_config['wandb_config']['name_training_run ']` (with a trailing space in the key, also likely an unintentional typo-bug) instead of `name_training_run`.
- Requires `path_to_save_model` and `model_config` to be present, raising `ValueError` otherwise.
- Delegates to `check_model_config_hvEEGNet(server_config['model_config'])`.

## Dependencies
- Internal: none directly imported (pure dict manipulation); conceptually paired with `library/config/config_dataset.py`, `library/config/config_training.py`, `library/config/config_model.py`, and `library/training/federated/server.py`, whose output dictionaries these functions validate/normalize.
- External: `os` (standard library) only.

## Notes
- This file is the closest thing to a schema/contract for the config dictionaries. When adding a new config key that needs a default or validation, this is the place to do it.
- Several copy-paste/typo bugs noted above (`notest` instead of `notes`, `name_training_run ` with trailing space, self-comparison in the trial bounds check) — worth fixing if these config paths are touched again, but left as-is here since fixing behavior wasn't requested.
- Designed to work hand-in-hand with configs loaded from TOML files (see `training_scripts/config/*.toml`), hence the `int -> None` and `list -> tuple` conversions in `check_model_config_EEGNet`.
