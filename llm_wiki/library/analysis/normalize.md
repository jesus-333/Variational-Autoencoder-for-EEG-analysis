# `library/analysis/normalize.py`

## Purpose
Collection of z-score normalization variants for EEG data (global, per-trial, per-channel, per-trial-per-channel), used to pre/post-process EEG arrays of shape `B x C x T` before analysis or visualization.

## Key functions

### `z_score(x_eeg, x_mean: float = None, x_std: float = None)`
- Normalizes the whole array with a single global mean/std. If `x_mean`/`x_std` not given, computes them from `x_eeg` itself via `np.mean`/`np.std`.

### `z_score_trial_by_trial(x_eeg)`
- For each trial (first dimension), subtracts that trial's own mean and divides by its own std, computed over the whole `C x T` matrix jointly (all channels pooled). Implemented with an explicit Python loop (marked `# TODO Vectorize`).

### `z_score_trial_by_trial_ch_by_ch(x_eeg)`
- Same as above but normalizes each channel of each trial independently (double loop over trials and channels). Also marked `# TODO Vectorize`.

### `z_score_ch_by_ch(x_eeg, x_mean: float = None, x_std: float = None)`
- Normalizes per-channel using mean/std computed across the batch and time dimensions (`axis=(0,2)`), i.e. one mean/std per channel shared across all trials. **Note**: the `x_mean`/`x_std` parameters are accepted but immediately overwritten inside the function body — passing custom values has no effect (likely a bug/leftover from refactoring).

### `tmp_norm(x_eeg)`
- Identical implementation to `z_score_trial_by_trial_ch_by_ch` (per-trial, per-channel z-score via nested loop). Looks like a leftover duplicate/experiment, name suggests throwaway code.

## Dependencies
- External: `numpy` only.

## Notes
- Four of the five functions ignore vectorization opportunities (explicit Python loops over trials/channels) — fine for small datasets but potentially slow for large ones.
- `z_score_ch_by_ch`'s unused `x_mean`/`x_std` parameters are a latent bug: callers trying to supply precomputed stats are silently ignored.
- `tmp_norm` duplicates `z_score_trial_by_trial_ch_by_ch` — likely safe to remove/consolidate, but check call sites in `analysis_script/` before deleting.
