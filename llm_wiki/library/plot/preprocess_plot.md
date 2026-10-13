# `library/plot/preprocess_plot.py`

## Purpose
Matplotlib/MNE-based plotting helpers for inspecting raw and preprocessed EEG data: time-domain trial plots, STFT/event-related-synchronization (ERS) spectrogram grids, band-averaged STFT traces over time, and before/after visualizations of filtering and ERS preprocessing. Companion visualization module to `library/dataset/preprocess.py`.

## Key classes
None.

## Key functions

### `plot_random_trial_time_domain(trials_matrix, ch_list, config: dict) -> None`
- `trials_matrix`: `N x C x T` numpy array (trials x channels x time).
- `ch_list`: array of channel names used to find the index of `config['ch_to_plot']`.
- Randomly samples `config['n_trials_to_plot']` trials (without replacement) for the selected channel and plots them overlaid on one axes, labeled by trial index.
- Config keys used: `ch_to_plot`, `n_trials_to_plot`, `t_start`, `t_end`, `figsize`, `fontsize`, `save_plot`, `show_fig`, `subject`.
- Side effect: saves to `"Plot/eeg_time_domain_subject_{subject}_{ch_to_plot}.png"` if `config['save_plot']` (relative path — assumes a `Plot/` directory exists in CWD; will raise if it doesn't). Calls `fig.show()` if `config['show_fig']`.

### `visualize_single_subject_average_channel_ERS(stft_data, label_list, ch_list, config) -> None`
- `stft_data`: `N x C x F x T` numpy array of STFT/ERS magnitudes.
- Computes per-class, per-channel average spectrograms via `extract_data_to_plot`, then plots a grid (`len(label_to_plot) x len(ch_to_plot)`) of `pcolormesh` spectrograms with a shared/implicit `vmin`/`vmax`/`cmap`.
- Contains a hardcoded `label_legend` dict mapping `{1: 'Left', 2: 'Right', 3: 'Foot'}`, immediately **shadowed** by a second `label_legend = {0: 'Left', 1: 'Right', 2: 'Foot'}` on the next line — the first assignment is dead code (likely leftover from switching label indexing from 1-based to 0-based); only the second (0-indexed) mapping is actually used. This is specific to the 3-class motor-imagery subset (left/right/foot) used in some d2a analyses.
- Config keys: `t`, `f`, `label_to_plot`, `ch_to_plot`, `figsize`, `fontsize`, `cmap` (optional), `vmin`, `vmax`, `y_limit` (optional), `subject`, `save_plot`, `filter_data`, `show_fig`.
- Side effect: saves to `"Plot/ERS_Subject_{subject}_filter.png"` or `"..._NO_filter.png"` depending on `config['filter_data']`.

### `extract_data_to_plot(data, ch_list, label_list, config) -> list[list[np.ndarray]]`
- For each label in `config['label_to_plot']`, averages `data` across all trials with that label, then extracts the requested channels (`config['ch_to_plot']`), squeezing singleton dims.
- Returns a nested list `[label][channel] -> averaged array`, consumed by `visualize_single_subject_average_channel_ERS`.

### `plot_average_band_stft(stft_data, ch_list, freq_array, config) -> None`
- Samples `config['n_trials_to_average']` random trials for one channel (`config['ch_to_plot']`), computes the average power in a frequency band (`config['band_start']`–`config['band_end']`) over time via `compute_average_band_stft`, and plots the resulting 1-D time series.
- Side effect: saves to `"Plot/Average_stft_Subject_{subject}_{ch_to_plot}.png"` if `config['save_plot']`.

### `compute_average_band_stft(stft_data, freq_array, config) -> np.ndarray`
- Averages `stft_data` across trials, then averages across the frequency bins falling within `[band_start, band_end]` (inclusive), returning a 1-D array over time.

### `show_filter_effect_on_trial(trials_matrix, ch_list, filter_config, plot_config) -> None`
- Picks one trial (random, or `plot_config['n_trial_to_plot']` if given), wraps it in an MNE `RawArray`, calls **`filter_RawArray(eeg_trial_RawArray, filter_config)`** to apply filtering, then plots the original vs. filtered signal for one channel overlaid.
- **Bug**: `filter_RawArray` is called but never imported/defined in this file. It is defined in `library/dataset/download.py` (`def filter_RawArray(raw_array_mne, config)`). Calling this function as-is will raise `NameError` unless the caller's namespace happens to have it imported some other way. Fix would be adding `from library.dataset.download import filter_RawArray` (or moving the function to a shared module).
- Side effect: saves to `"Plot/Filter_Effect_Subject_{subject}_{ch_to_plot}_{idx_trial}.png"` if `plot_config['save_plot']`.

### `construct_filter_string(filter_config) -> str`
- Builds a human-readable title string describing the filter (method, bandpass/lowpass/highpass type and cutoff(s), IIR order if applicable) for use in plot titles.
- `filter_config['filter_type']`: `0` = bandpass (uses `fmin`/`fmax`), `1` = lowpass (uses `fmax`), `2` = highpass (uses `fmin`). This 0/1/2 encoding mirrors the convention used in `library/dataset/preprocess.py` / `download.py`.

### `show_after_before_ERS_on_trial(trials_matrix_before_ERS, trials_matrix_after_ERS, ch_list, f, t, config) -> None`
- Side-by-side `pcolormesh` comparison of one trial's spectrogram before vs. after ERS computation, for one channel.
- Always calls `fig.show()` unconditionally (no `save_plot`/`show_fig` config gating here, unlike the other functions in this file).

## Dependencies
- Internal: implicitly depends on `filter_RawArray` from `library.dataset.download` (not imported — see bug note above).
- External: `numpy`, `mne` (for `RawArray`/`create_info` in `show_filter_effect_on_trial`), `matplotlib.pyplot` — imported with a fallback: if the normal `matplotlib.pyplot` import fails, it switches the backend to `'agg'` (headless) and retries, useful for running on servers without a display.

## Notes
- All plotting functions take a `config` dict rather than named keyword arguments, following the repo-wide "everything is a dict" convention; there is no schema validation here (contrast with `library/check_config.py`, which validates dataset/train/model configs but not plot configs).
- All `save_plot` paths are relative (`"Plot/..."`), so the working directory must contain (or be expected to contain) a `Plot/` folder, and the save paths will collide across subjects/channels/trials only by virtue of the filename including those identifiers — no folder-per-run structure.
- `visualize_single_subject_average_channel_ERS`'s dead `label_legend` line (1-indexed, immediately overwritten) is worth knowing about if adapting this code for a different label indexing scheme.
