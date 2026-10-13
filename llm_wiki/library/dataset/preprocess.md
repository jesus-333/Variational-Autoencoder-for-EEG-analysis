# `library/dataset/preprocess.py`

## Purpose
Second stage of the data pipeline: turns raw trial tensors (from `download.py`) into ready-to-train PyTorch `Dataset` objects (`dataset_time.EEG_Dataset` or `dataset_stft.EEG_Dataset_stft`), and implements time-frequency preprocessing (STFT, baseline removal / event-related synchronization). Also contains dataset-specific "get_dataset_*" orchestration functions and a handful of exploratory/visualization scripts.

## Key functions

### `baseline_removal(trials_matrix, config: dict)`
- Pipeline helper: computes STFT (`compute_stft`) then event-related synchronization (`compute_ERS`) on top of it. Returns `(stft_trials_matrix_ERS, t_trial, f)`.

### `compute_stft(trials_matrix, config: dict)`
- Computes `scipy.signal.stft` independently for every trial and every channel in `trials_matrix` (shape `trials x channels x time`), using `config['stft_parameters']` (`sampling_freq`, `nperseg`, `window`, `noverlap`).
- Squares the magnitude of the complex STFT output (power spectrum): `np.power(np.abs(tmp_stft), 2)`.
- Intends to restrict the output to only the frequencies that survived filtering (band/low/high-pass via `config['filter_type']`), but see Notes — this restriction is dead code.
- Returns `(stft_matrix[:, :, idx_freq, :], t, f[idx_freq])`.

### `compute_ERS(stft_trials_matrix, t, f)`
- Computes event-related synchronization/desynchronization relative to a fixed rest period `0.5s <= t < 1.75s` (hardcoded), by calling `compute_ERS_single_channels` per trial/channel.
- Returns `(stft_trials_matrix_ERS, t)`.

### `compute_ERS_single_channels(stft_channel, idx_rest)`
- For one channel's STFT (`freq x time`): computes the mean power over the rest period, then for every time bin computes `((power - rest_mean) / rest_mean) * 100` — i.e. percent change relative to baseline rest power.

### `get_dataset_d2a(config: dict)`
- Full D2a pipeline: downloads train/test data via `download.get_D2a_data`, stores the discovered channel list back into `config['channels_list']`.
- Expands a channel-depth dimension (`np.expand_dims(data, 1)`) so shape becomes `N x 1 x C x T` (the depth-of-1 is required because the models use `Conv2d`).
- Optional `config['train_trials_to_keep']`: subsets training trials by index.
- Optional `config['use_stft_representation']`: converts to time-frequency domain via `compute_stft`.
- Optional `config['percentage_split_train_test']` (if `!= -1`): re-merges the original train+test sets and re-splits them according to this fraction (via `support_function.get_idx_to_split_data`), overriding moabb's native split.
- Optional train/validation split via `config['percentage_split_train_validation']`.
- Wraps the final arrays in `dataset_stft.EEG_Dataset_stft` (if STFT used) or `dataset_time.EEG_Dataset` (otherwise), passing `config['normalize']` to the time-domain dataset.
- Returns `(train_dataset, validation_dataset, test_dataset)` — `validation_dataset` is `None` if no validation split was requested.

### `get_dataset_TUAR(config: dict)`
- Downloads/segments TUAR via `download.get_TUAR_data`, expands the channel-depth dim, splits into train/test (`config['percentage_split_train_test']`, required to be strictly between 0 and 1 or raises `ValueError`) and train/validation (`config['percentage_split_train_validation']`, same constraint).
- TUAR has no real labels, so **random integer labels in `[0,4)` are generated** for train/test/validation "only for compatibility" with downstream training code that expects a label tensor.
- Wraps in `dataset_time.EEG_Dataset` for all three splits.

### `get_dataset_SEED(path_filename: str, trials_length_in_seconds: int)`
- Loads a SEED dataset `.mat` file (`scipy.io.loadmat`), which contains several raw recordings of shape `C x T` at a hardcoded 200 Hz.
- Discards the 3 MATLAB-format metadata keys (identified by `'__' in key`), slices every recording into fixed-length non-overlapping trials of `sampling_freq * trials_length_in_seconds` samples, discarding any trailing partial trial.
- Returns a single `N x C x T'` numpy array.

### `get_dataset_SEED_split_in_train_test_validation(path_filename, trials_length_in_seconds, percentage_split_train_test, percentage_split_train_validation=-1, random_seed=42)`
- Loads SEED data via `get_dataset_SEED`, expands the depth dimension, and splits into train/test (required to be strictly between 0 and 1) and optionally train/validation.
- Returns `(data_train, data_test, data_validation)` — note the **unusual argument order** (test before validation) relative to other functions in this file.

### `download_preprocess_and_visualize()`, `show_filter_effect()`, `show_filter_effect_on_ERS()`, `show_after_before_ERS()`
- Standalone exploratory/demo scripts (not part of the reusable API) that download D2a data, compute STFT/ERS, and call plotting functions from `library.plot.preprocess_plot` to visually inspect filter and baseline-removal effects. Each hardcodes a single test subject by overwriting a list literal (e.g. `subjects_list = [1,2,...]` immediately followed by `subjects_list = [3]`), clearly leftover from manual interactive debugging.

## Dependencies
- Internal: `library.config.config_dataset` (`cd`), `library.config.config_plot` (`cp`), `library.plot.preprocess_plot` (`pp_plot`), `library.dataset.download`, `library.dataset.dataset_time` (`ds_time`), `library.dataset.dataset_stft` (`ds_stft`), `library.dataset.support_function` (`sf`).
- External: `matplotlib.pyplot` (with a fallback to the `agg` backend if the default backend import fails — for headless/server environments), `numpy`, `scipy.signal`, `scipy.io`, `moabb.datasets`.

## Notes
- **Dead code bug in `compute_stft`**: the frequency-filtering logic computes `idx_freq` based on `config['filter_type']`, but immediately afterward the line `idx_freq = np.ones(len(f)) == 1` unconditionally overwrites it with "keep everything". So no frequency bins are ever actually dropped by this function, regardless of `filter_data`/`filter_type` config. Same dead-code pattern appears in `compute_ERS` (`k_to_avoid` is computed conditionally then immediately reset to `[]`).
- `compute_ERS`'s rest-period window (`t >= 0.5 and t < 1.75`) is hardcoded, not config-driven — assumes a specific trial timing convention (consistent with the D2a cue-based paradigm) and will silently produce nonsensical ERS values if applied to a dataset with different trial timing.
- Leftover `# load_ext autoreload` IPython magic comments at the top of the file indicate this was originally developed/run interactively (Spyder/IPython), not purely as a library.
- There's a likely bug in `get_dataset_SEED_split_in_train_test_validation`: in the `else` branch when `percentage_split_train_validation` is invalid, `data_train, data_validation = data_train[idx_train], data_train[idx_test]` uses `idx_test` (from the earlier train/test split) to index into `data_train`, which is almost certainly a copy-paste mistake (should probably be `idx_validation`). Flag this if touching SEED validation-split logic.
- `get_dataset_d2a` and `get_dataset_TUAR` both assume `config['normalize']` / `percentage_split_*` keys exist; there's no defensive `.get()` — always pass a fully-populated config (see `library/config/config_dataset.py`).
- The TUAR random-label behavior means **any classifier accuracy/metric computed on a TUAR-wrapped dataset is meaningless**; TUAR is intended for the autoencoder/reconstruction pipeline (hvEEGNet), not classification.
