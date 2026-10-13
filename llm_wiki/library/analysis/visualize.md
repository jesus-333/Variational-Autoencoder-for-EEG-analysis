# `library/analysis/visualize.py`

## Purpose
Standalone plotting helpers for raw EEG signals: single-channel time/frequency (PSD) plots, and concatenated multi-trial visualizations (with optional overlay of a reconstructed signal). Function-based (no classes), contrast with `library/analysis/visualizer.py` which wraps similar plotting in a stateful `Visualizer` class tied to a trained model.

## Key functions

### `plot_single_eeg_channel(x_eeg, t, config)`
- Plots a single channel's signal over time, or its Welch PSD if `config['plot_psd']` is True (computed via `scipy.signal.welch` with `config['nperseg']`, default 256, and `config['fs']`, default 1).
- `config` keys: `plot_psd` (optional, default False), `nperseg`, `fs`, `fontsize` (optional), `figsize` (required), `title` (optional).
- Returns `(fig, ax)`.

### `plot_single_eeg_channel_both_time_and_frequency(x_eeg, t, config)`
- Same as above but always plots both time-domain and PSD side by side (2 subplots). `config` keys: `nperseg`, `fs`, `fontsize`, `figsize`, `title`.
- Returns `(fig, ax)` (`ax` is an array of 2 Axes).

### `visualize_set_of_trials(data, config: dict, x_r=None)`
- Concatenates and flattens a contiguous range of trials (`config['idx_start']:config['idx_end']`) for a single channel (`config['idx_ch']`) into one long time series and plots it, optionally overlaying the reconstructed signal `x_r` if provided.
- `config` keys: `idx_start`, `idx_end`, `idx_ch`, `trial_length`, `figsize`, `fontsize`, `add_trial_line` (draws vertical red lines at trial boundaries when True, else just a y-grid).
- Internally uses `extract_and_flat_trial`.

### `extract_and_flat_trial(data, config)`
- Helper: slices `data[idx_start:idx_end]`, selects channel `idx_ch` (assuming shape `N x 1(depth) x C x T`), and flattens trials into one 1D array.

## Dependencies
- External: `numpy`, `torch` (imported but unused directly in visible code), `matplotlib.pyplot`, `scipy.signal`.

## Notes
- All functions take loosely-typed `config` dicts with many optional/required keys accessed via `in` checks — no schema validation; passing an incomplete config raises a `KeyError` deep in the function rather than failing fast.
- Pure visualization/plotting utilities; no dependency on any model or dataset classes (unlike `visualizer.py`).
- Likely overlaps conceptually with plotting code in `library/plot/preprocess_plot.py` — check there before adding new EEG-plotting helpers to avoid further duplication.
