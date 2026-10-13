# `library/dataset/download.py`

## Purpose
Central module for obtaining raw EEG data for every public dataset supported by the repository. It wraps the [moabb](http://moabb.neurotechx.com/) library (and, for one dataset, a direct Google Drive download) to fetch data, split it into train/test, select the right EEG channels, and convert labels to numerical class indices. This is the first stage of the data pipeline, feeding into `library/dataset/preprocess.py`.

## Key functions

### `get_moabb_data_automatic(dataset, paradigm, config: dict, type_dataset: str)`
- The "automatic" download path: lets moabb's `paradigm.get_data()` handle trial segmentation and baseline removal.
- Applies config-driven resampling (`resample_data`/`resample_freq`), filtering (`filter_data`/`fmin`/`fmax`), trial window (`trial_start`/`trial_end`) and number of classes before calling `paradigm.get_data(dataset=dataset, subjects=config['subjects_list'])`.
- Calls `get_idx_train_or_test` to select train/test/full rows, then `convert_label` and `get_dataset_channels`.
- Returns `(data, labels, ch_list)` with `data` shaped `N x C x T`.

### `get_idx_train_or_test(dataset, info: dict, type_dataset: str)`
- Returns a boolean/index array selecting which trials belong to train/test/full, based on dataset-specific metadata columns moabb stores in `info` (`session` or `run`).
- Per-dataset logic (all hardcoded by checking `str(type(dataset))`):
  - **BNCI2014_001 / BNCI2014001 (D2a)**: `info['session']` == `'0train'`/`'1test'`.
  - **Zhou2016**: `info['run']` == `'0'`/`'run_0'` (train) or `'1'`/`'run_1'` (test) — handles a moabb naming-convention change across versions.
  - **Lee2019_MI**: same idea but on `info['session']` with `'0'`/`'session_0'` and `'1'`/`'session_1'`.
  - **Ofner2017**: 10 recording sessions per subject (ids `0`-`9`); first 5 = train, last 5 = test, or all with `type_dataset='full'`.
  - **Schirrmeister2017**: `info['run']` == `'0train'`/`'1test'`.
  - **BI2014a, Weibo2014, Cho2017, GrosseWentrup2009**: no native train/test split, so the function just takes the first half of trials as train and the second half as test (or all trials with `'full'`).
  - Anything else raises `ValueError('Dataset not supported')`.

### `get_moabb_data_handmade(dataset, config, type_dataset)`
- Alternative to the automatic path: downloads raw continuous recordings via `dataset.get_data(subjects=...)` and performs trial segmentation manually (via `get_trial_handmade`) instead of relying on moabb's paradigm.
- Handles two different moabb session-naming conventions (`'session_T'/'session_E'` vs `'0train'/'1test'`).
- For D2a (`BNCI2014_001`) only, optionally restricts to specific channels via `get_idx_ch`/`config['channels_list']`, otherwise defaults to the first 22 channels.
- Returns `(trials_per_subject, labels_per_subject, ch_list)` where the first axis is subject.

### `get_trial_handmade(raw_data, config)`
- Iterates over each "run" in a subject's raw MNE data, extracts event onsets via `mne.find_events`, optionally filters the continuous signal (`filter_RawArray`), and segments it into trials (`divide_by_event`).
- Updates `config['sampling_freq']` as a side effect (reads it from the MNE raw object).
- Returns `(trials_matrix, labels, ch_names)`.

### `filter_RawArray(raw_array_mne, config)`
- Applies one of 4 filter types to an MNE `RawArray`/`RawEDF`, selected by `config['filter_type']`: `0`=bandpass (`fmin`/`fmax`), `1`=lowpass (`fmax`), `2`=highpass (`fmin`), `3`=notch (`notch_freq`). Uses `config['filter_method']` and `config['iir_params']`.

### `divide_by_event(raw_run, events, config)`
- Slices the continuous run data between consecutive event sample indices to build individual trials, then further restricts each trial to the window `[sampling_freq * trial_start, sampling_freq * trial_end]`.
- Scales the result by `1e6` (converts volts to microvolts — "the original signal is in microvolt" per the comment, so this reconciles raw MNE's volt convention).

### `get_idx_ch(ch_list_dataset, config)`
- Builds an integer index array mapping `config['channels_list']` (desired channel names) onto their positions in `ch_list_dataset` (the dataset's native channel order). O(n*m) nested loop, no vectorization.

### `get_data_subjects_train(subjects_list: list)`
- Convenience wrapper: builds a default D2a dataset config via `config_dataset.get_moabb_dataset_config`, downloads `BNCI2014_001` train data via the handmade path. Hardcoded to D2a only.

### `get_D2a_data(config, type_dataset)`
- Dataset-specific entry point for BCI Competition IV 2a. Tries `mb.BNCI2014_001()` then falls back to `mb.BNCI2014001()` (moabb renamed the class across versions; both are tried defensively).
- Uses `mp.MotorImagery()` paradigm.
- If `config['use_moabb_segmentation']` is True, uses the automatic path and keeps only the first 22 channels (D2a has 25; the last 3 are EOG channels, not EEG) and notes the automatic moabb segmentation produced 281 trials instead of the expected 288 in some versions (a known quirk, see Notes).
- Otherwise uses the handmade path, flattens the subject dimension, and remaps labels from `1..4` to `0..3` (PyTorch class-index convention) by subtracting 1.
- Optional `config['use_fewer_trials']`: if > 0, randomly subsamples that many trials using `config['seed_split']`.
- Returns `(data, labels.squeeze(), ch_list)`.

### `get_Zhou2016(config, type_dataset)`
- Zhou et al. 2016 motor imagery dataset via moabb (`mb.Zhou2016()`, `MotorImagery` paradigm), automatic path.
- Drops channels `[0,1]` (VEOU/VEOUL, EOG) and the last channel (STIM) from the returned channel list — note the comment says these are already excluded from `data` by moabb's paradigm, so this only fixes up the channel-name list to match.

### `get_Weibo2014(config, type_dataset)`
- Weibo 2014 dataset, `MotorImagery` paradigm. Drops the last 3 channel names (VEO, HEO, STIMO14 — again, already excluded from the data array).

### `get_Cho2017(config, type_dataset)`
- Cho et al. 2017 dataset, `LeftRightImagery` paradigm. Drops the last 5 channel names (EMG1-4, Stim).

### `get_Ofner2017(config, type_dataset)`
- Ofner 2017 dataset, `MotorImagery` paradigm. Rescales data by `1e-6` (raw values are in the 10^6-10^7 range — opposite direction from the `1e6` scaling in `divide_by_event`, so data magnitude conventions differ between the "automatic" and "handmade" download paths). Drops the last 36 channel names.

### `get_GrosseWentrup2009(config, type_dataset)`
- GrosseWentrup 2009 dataset, `LeftRightImagery` paradigm. No channel trimming.

### `get_Lee2019_MI(config, type_dataset)`
- Lee 2019 motor imagery dataset, `LeftRightImagery` paradigm. Drops last 5 channels (EMG1-4, STI 014).

### `get_Schirrmeister2017(config, type_dataset)`
- Schirrmeister 2017 (High-Gamma) dataset, `MotorImagery` paradigm. No channel trimming (the `ch_list = ch_list` line is a no-op).

### `get_BI2014a(config, type_dataset)`
- Huebner et al. 2018 P300 dataset (`mb.BI2014a()`, `P300` paradigm). Drops the last channel name.

### `get_TUAR_data(config: dict)`
- Handles the TUAR (Temple University Artifact) dataset, which is **not downloaded via moabb** but from a curated Google Drive folder (a subset of the full TUAR corpus) via `download_TUAR()`.
- Checks whether `data/TUAR_dataset` already exists locally; downloads only if missing or `config['force_download']` is True.
- Reads an EDF file directly with `mne.io.read_raw_edf(config['path_file'])`.
- Optionally restricts/reorders channels (`config['channels_to_use']`), resamples, filters.
- Segments the continuous recording into fixed-length epochs using `mne.make_fixed_length_epochs` with `config['segment_duration']` and `config['segment_overlap_duration']`.
- Optional `config['samples_to_use']` truncates the number of resulting segments (marked `# TODO (?)` in source).
- Returns `(data_segmented, data_raw, ch_list)` — note this returns the raw MNE object itself (not labels), unlike the moabb dataset functions.

### `download_TUAR()`
- Downloads the TUAR subset from a hardcoded Google Drive share link into `data/`.

### `download_from_google_drive(link_drive: str, path_save: str)`
- Lazily imports `googledriver.download_folder` (import deferred so the optional dependency is only required if TUAR is actually used) and downloads the folder; creates `path_save` if missing. Raises `ImportError` with an install hint if `googledriver` is not installed.

### `get_dataset_channels(dataset)`
- Returns the channel-name list for a given moabb dataset/raw object, with per-dataset-type branching (again matched by `str(type(dataset))`) because the internal nested-dict key layout moabb uses to store per-subject/session/run raw data differs by dataset and by moabb version (keys like `'0train'` vs `'0'`, `'session_0'` vs `'run_0'`, etc.).
- Supports: `BNCI2014_001`, `RawEDF` (TUAR), `Zhou2016`/`BI2014a`/`Weibo2014`/`Cho2017`/`GrosseWentrup2009`, `Ofner2017`, `Lee2019_MI`, `Schirrmeister2017`. Anything else raises `ValueError`.

### `convert_label(raw_labels, use_BCI_D2a_label=True)`
- Converts moabb's string/raw labels into a numeric vector.
- If `use_BCI_D2a_label=True` (default — but every call site in this file passes `False`): maps the 4 known D2a class names (`left_hand`, `right_hand`, `feet`, `tongue`) to fixed integers 1-4.
- If `False`: uses `np.unique(raw_labels)` to discover classes dynamically and assigns them indices `0..k-1` in sorted order, printing the mapping as it goes.

## Dependencies
- Internal: `library.config.config_dataset` (as `cd`, only used by `get_data_subjects_train`), `library.check_config` (`check_config_dataset`, validates the config dict before use).
- External: `numpy`, `mne` (raw EEG I/O, filtering, event detection, epoching), `moabb.datasets` (as `mb`) and `moabb.paradigms` (as `mp`) for dataset access, `os` (path checks), and a lazily-imported optional `googledriver` package for TUAR.

## Notes
- Every dataset-dispatch function (`get_idx_train_or_test`, `get_dataset_channels`) matches on `str(type(dataset))` substring checks rather than `isinstance`, because moabb's dataset class names/paths have changed across library versions — this is a deliberate but fragile compatibility shim; expect it to need updates if moabb is upgraded.
- Scaling conventions are inconsistent across functions: `divide_by_event` (handmade path) multiplies by `1e6`, while `get_Ofner2017` (automatic path) multiplies by `1e-6`. A future agent changing scaling behavior must check both paths.
- `get_D2a_data` notes a known moabb quirk in a comment: automatic segmentation sometimes yields 281 trials instead of the expected 288.
- The TUAR dataset is fundamentally different from the others: it is not a moabb dataset, has no labels (random labels are generated downstream in `preprocess.get_dataset_TUAR` "only for compatibility"), and is fetched from a manually curated Google Drive link rather than the official TUAR source (which requires separate registration/access).
- `get_data_subjects_train` is a thin, D2a-only convenience function; most real usage goes through `get_D2a_data` or the dataset-specific `get_*` functions directly.
- Several docstrings are incomplete placeholders (e.g. `get_Weibo2014`, `get_Ofner2017`, `get_GrosseWentrup2009`, `get_Lee2019_MI`, `get_Schirrmeister2017` all have `"""Get the dataset presented by ..."""` with no citation filled in, unlike `get_Zhou2016`/`get_Cho2017`/`get_BI2014a` which do cite a paper/DOI).
