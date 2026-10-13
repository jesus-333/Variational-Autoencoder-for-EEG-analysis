# `library/dataset/dataset_stft.py`

## Purpose
Defines the PyTorch `Dataset` class for time-frequency (STFT-transformed) EEG data, the alternative representation to `dataset_time.EEG_Dataset` used when `config['use_stft_representation']` is enabled in `preprocess.get_dataset_d2a`. Also includes basic per-channel/per-class STFT visualization helpers.

## Key classes

### `EEG_Dataset_stft(Dataset)`
- Constructor: `__init__(self, data, labels, info: dict)`. `data` must have shape `[Trials, Channels, Frequency bins, Time bins]`. `info` carries side metadata: `channels_list`, optionally `subjects_list`, `t` (time axis after STFT) and `f` (frequency axis).
- Converts `data`/`labels` to `torch.float`/`torch.long`.
- `self.t = info['t'] + 2` — shifts the STFT time axis by a constant +2 seconds (presumably to align STFT window-center timestamps with absolute trial time; see Notes).
- Hardcodes a 4-class label mapping specific to D2a motor imagery: `{0:'left', 1:'right', 2:'foot', 3:'tongue'}` (and its inverse) — this dataset class is therefore D2a-specific despite the generic name.
- `__getitem__(idx)` returns `(data[idx], labels[idx])`; `__len__` returns number of trials.
- `get_average_trial(ch, label, vmin=None, vmax=None)`: averages the STFT power across all trials of a given class for one channel (selected by name or by numeric label), with optional clipping to `[vmin, vmax]`. Returns `(trial, vmin, vmax)`.
- `visualize_trial(idx_trial, ch, vmin=None, vmax=None)`: plots the STFT of a single trial/channel using `pcolormesh`.
- `visualize_average_trial(ch, label, vmin=None, vmax=None)`: plots the class-averaged STFT for one channel via `get_average_trial`.
- `create_plot_config(vmin, vmax, ch)`: builds a small plotting-config dict (figsize, fontsize, cmap `'Blues_r'`, vmin/vmax).
- `__plot_stft(stft_data, config)` (name-mangled private): draws the actual `pcolormesh` figure with `self.t`/`self.f` axes, optionally saving to disk as both `.png` and `.pdf` if `config['save_fig']` is set.
- `main()` / `if __name__ == "__main__"`: a manual smoke-test script (imports `config_dataset`/`preprocess` by bare module name, which only works if run from inside the `library/dataset/` directory — not a properly packaged entry point) that builds a D2a STFT dataset for subject 3 and visualizes one channel's average trial.

## Dependencies
- Internal: none at import time (the `main()` test function locally imports `config_dataset` and `preprocess`, but by bare name rather than a relative/package import — this only works in an interactive/script context, not via normal package import).
- External: `numpy`, `matplotlib.pyplot`, `torch`, `torch.utils.data.Dataset`.

## Notes
- The hardcoded 4-class label dictionary (`left`/`right`/`foot`/`tongue`) means this class is tied to the BCI Competition IV 2a (D2a) label scheme; reusing it for another dataset's STFT data would mislabel classes in `visualize_average_trial`.
- `get_average_trial`/`visualize_trial` index `self.data[idx_trial, idx_ch]` where `idx_ch` is a boolean mask from `self.ch_list == ch` — this relies on `self.ch_list` being a numpy array (not a plain Python list) for the comparison to broadcast as a boolean mask.
- The `self.t = info['t'] + 2` offset is unexplained in the source (no comment) — worth double-checking against the actual trial timing convention (`trial_start`/`trial_end` in dataset config) before relying on absolute time values from this axis.
- `visualize_average_trial` has a local variable typo `vmnax` (unused, shadows nothing, harmless) in its unpacking of `get_average_trial`'s return value — cosmetic only.
- Leftover `%load_ext autoreload` IPython-magic docstring at the top, consistent with other files in this package — indicates interactive/Spyder-driven development.
