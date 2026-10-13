# `library/dataset/dataset_time.py`

## Purpose
Defines the PyTorch `Dataset` classes used for time-domain (raw, non-STFT) EEG data — the representation consumed by EEGNet/MBEEGNet/vEEGNet/hvEEGNet/ChWi models. This is the final step before a `DataLoader` is created for training.

## Key classes

### `EEG_Dataset(Dataset)`
- Standard EEG dataset for trial-structured data.
- Constructor: `__init__(self, data, labels, ch_list, normalize=-1)`. Requires `data.shape == [Trials, 1, Channels, Time]` (raises `ValueError` otherwise, since the depth-1 dimension is required for `Conv2d`).
- Converts `data`/`labels` to `torch.float`/`torch.long` tensors.
- `normalize` parameter selects a normalization mode applied once at construction time: `1` = min-max over the whole dataset (`minmax_normalize_all_dataset`), `2` = min-max per trial-and-channel (`normalize_trial_by_trial_channel_by_channel`), `3` = min-max per trial across all channels (`normalize_trial_by_trial`). Any other value (including the default `-1`) skips normalization.
- `__getitem__(idx)` returns `(data[idx], labels[idx])`; `__len__` returns number of trials.
- `minmax_normalize_all_dataset(a, b)`: rescales the entire tensor to `[a, b]` using global min/max.
- `normalize_trial_by_trial_channel_by_channel(a, b)`: nested Python loop over trials and channels, rescaling each (trial, channel) time series independently to `[a, b]`.
- `normalize_trial_by_trial(a, b)`: loop over trials, rescaling each full trial (all channels together) to `[a, b]`.

### `EEG_Dataset_ChWi(Dataset)`
- Dataset variant for the "ChWi" (channel-wise) autoencoder, which treats every individual EEG channel as its own independent 1D sample rather than keeping trials intact.
- Constructor: `__init__(self, data, labels=None, ch_list: list=None, norm_type: int=-1)`. Accepts `data` as either a numpy array or an existing torch tensor; expects shape `[Trials, C, T]`.
- Reshapes `data` from `[B, C, T]` to `[(B*C), 1, T]` — i.e. flattens trials and channels together into the batch dimension, with a depth-1 axis for 1D convolution.
- If `labels is None`, synthesizes a dummy all-ones label tensor "only used to keep dataset consistent with training functions" (not meaningful for ChWi's unsupervised training).
- `norm_type` (applied via private `__normalize`): `1` = per-signal min-max to `[0,1]`, `2` = per-signal standardization (zero mean, unit std). Any other positive value raises `ValueError`.
- Has a `return_only_data` flag (default `False`, toggled externally after construction) — when `True`, `__getitem__` returns just the data tensor instead of `(data, label)`, useful when the autoencoder training loop doesn't need labels.

## Dependencies
- Internal: none (self-contained).
- External: `torch`, `torch.utils.data.Dataset`.

## Notes
- The two classes serve genuinely different data layouts: `EEG_Dataset` preserves per-trial multi-channel structure (`Trials x 1 x C x T`, for 2D-conv models like EEGNet-family), while `EEG_Dataset_ChWi` flattens channels into the sample axis (`(Trials*C) x 1 x T`, for 1D-conv per-channel models). Don't assume they're interchangeable.
- Normalization in `EEG_Dataset` is applied once, eagerly, at construction (mutates `self.data` in place via Python loops) rather than lazily per-`__getitem__` — this can be slow for large datasets with per-trial/per-channel normalization since it's not vectorized.
- `EEG_Dataset_ChWi.__normalize` is name-mangled (`__normalize` -> `_EEG_Dataset_ChWi__normalize`) since it's "private" by convention; only called internally from `__init__`.
