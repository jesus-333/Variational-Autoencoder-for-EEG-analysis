# `library/config/config_dataset.py`

## Purpose
Factory functions that build the plain-`dict` configuration objects consumed by `library/dataset/download.py` and `library/dataset/preprocess.py` to download, filter, resample, segment and split EEG datasets (BCI Competition IV 2a via MOABB, the "artifact" datasets, and TUAR). These dicts are the de-facto schema for dataset configuration across the whole repo — there is no dataclass/validation layer, just conventions enforced by `library/check_config.py`.

## Key functions

### `get_moabb_dataset_config(subjects_list=[1..9], use_stft_representation=False) -> dict`
Config for downloading the BCI Competition IV 2a (d2a) dataset through MOABB. Keys:
- **Frequency filtering**: `filter_data` (bool, default `False`), `filter_type` (0=bandpass, 1=lowpass, 2=highpass, 3=notch — notch only implemented for d2a), `fmin`/`fmax` (Hz, used for bandpass/highpass/lowpass, only if `filter_data`), `notch_freq` (only if `filter_type == 3`), `filter_method` (`'iir'`), `iir_params` (dict: `ftype='cheby2'`, `order=20`, `rs=30`; only if `filter_method == 'iir'`).
- **Resampling**: `resample_data` (bool), `resample_freq` (Hz, default 128, only if `resample_data`).
- **Trial segmentation**: `trial_start`/`trial_end` (seconds, default 2/6 — comment says "keep this value"), `use_moabb_segmentation` (bool, lets MOABB do the trial splitting instead).
- **Train/test/validation split**: `seed_split` (42), `percentage_split_train_test` (-1 means keep the dataset's original train/test division), `percentage_split_train_validation` (0.9, applied only to the training portion), `use_fewer_trials` (-1 disables; a positive int randomly subsamples that many trials).
- **Other**: `n_classes` (4 for d2a), `subjects_list` (which d2a subjects to download).
- **STFT (marked IGNORE/NOT USED in comments)**: `use_stft_representation`, `channels_list` (deleted from the dict entirely if `use_stft_representation` is `False`), `normalize` (kept at 0), `train_trials_to_keep` (optional boolean mask over the pre-split training set), a commented-out `normalization_type`.
- If `use_stft_representation` is `True`, a nested `stft_parameters` key is added via `get_config_stft()`.

### `get_config_stft() -> dict`
STFT parameters: `sampling_freq=250`, `nperseg=50`, `noverlap=40`, `window='hann'` (a Gaussian-window alternative is commented out).

### `get_artifact_dataset_config(type_dataset, folder_to_save='v2') -> dict`
Minimal config for the "artifact" dataset variant: `type_dataset`, `filter_data=True`, `fmin=0`, `fmax=125`, `resample_data=True`, `resample_freq=256`, `folder_to_save`.

### `get_TUAR_dataset_config(path_file: str, force_download: bool = True) -> dict`
Config for the TUAR (Temple University Artifact) dataset. Keys: `path_file`, `force_download`, `channels_to_use` (hardcoded list of 26 TUAR channel names; an earlier 22-channel version is left commented out above it), `samples_to_use=400`, the same filtering block as above (`filter_data=False`, `filter_type=0`, `fmin=0.5`, `fmax=50`, `filter_method='iir'`, `iir_params`), resampling (`resample_data=True`, `resample_freq=250`), segmentation (`segment_duration=4`, `segment_overlap_duration=3`), split settings (`seed_split=42`, `percentage_split_train_test=0.5`, `percentage_split_train_validation=0.9`), and `normalize=-1`.

## Dependencies
- Internal: none (pure config, no imports from other `library` modules).
- External: none (no imports at all; `dict(...)` literals only).

## Notes
- No validation happens in this file — `library/check_config.py` is the place that checks these dicts for consistency before use.
- Several comments explicitly say "IGNORE"/"NOT USED"/"Ignore and kept to 0" for the STFT-related and `normalize` fields in `get_moabb_dataset_config` — they exist for legacy/experimental reasons and should not be relied upon for new work.
- `channels_list` is conditionally deleted from the returned dict (not just left at a default) when STFT is unused — code consuming this config must not assume the key always exists.
- The two channel lists in `get_TUAR_dataset_config` differ only slightly (the active one adds `'EEG F8-REF'` and swaps some channels) — the old list is kept commented out directly above, a pattern repeated elsewhere in the repo instead of using version control history.
- `get_config_stft`'s signature takes no dataset size parameters, so `sampling_freq`/`nperseg`/`noverlap` must be manually kept consistent with whatever `resample_freq` is chosen elsewhere.
