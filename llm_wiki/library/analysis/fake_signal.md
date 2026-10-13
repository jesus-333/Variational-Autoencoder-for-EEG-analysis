# `library/analysis/fake_signal.py`

## Purpose
Tiny utility module for generating synthetic test signals (sinusoids), presumably used to sanity-check analysis/plotting code (e.g. DTW, spectra) without needing real EEG data.

## Key functions

### `generate_sinusoide(config: dict)`
- Generates a sine wave from a config dict with keys: `f_sampling` (sampling frequency), `t_start`, `t_end` (time window), `f_sine` (sine frequency), and optional `phase` (default 0) and `amplitude` (default 1).
- Returns `(x, t)`: the signal array and its time vector, built with `np.linspace` and `fs * (t_end - t_start)` samples.

### `add_white_noise(x)`
- **Not implemented** — body is just `pass`. Calling it has no effect and returns `None`.

## Dependencies
- External: `numpy`.

## Notes
- `add_white_noise` is a stub; any future agent relying on it must implement it first.
- Very small/standalone file, no internal dependencies — safe to use in isolation for quick signal-processing experiments.
