# `library/analysis/visualizer.py`

## Purpose
A stateful `Visualizer` class that loads the D2a (BCI Competition IV 2a) dataset for one or more subjects, instantiates a vEEGNet-family model, loads trained weights, and plots original-vs-reconstructed EEG signals for a chosen subject/trial/channel. Essentially an interactive/notebook-style exploration tool (note the leftover IPython `%load_ext autoreload` string at module scope, which is inert code — it's just a string literal, not executed).

## Key classes

### `Visualizer`
Holds dataset + model state for repeated interactive reconstruction plotting.

- **Constructor** `__init__(self, config: dict)`:
  - `config['subj_list']`: list of subject IDs to load via `library.config.config_dataset.get_moabb_dataset_config` and `library.dataset.preprocess.get_dataset_d2a`.
  - `config['model_name']`, `config['model_config']`: passed to `library.training.train_generic.get_untrained_model`.
  - `config['device']` (optional, default `'cpu'`).
  - `config['path_weight']` (optional): if present, immediately loads model weights.
  - `config['dataset_config']`: nested dict read for `trial_start`/`trial_end`/`t_max_to_visualize` (time-window settings for plotting).
  - `config['figsize']` (default `(15, 10)`), `config['fontsize']` (default 12), `config['save_fig']` (default False, but not actually used anywhere in visible methods).
- `get_data(self, subj_list)`: loads train/validation/test D2a datasets per subject in `subj_list` into lists, but **returns only the last subject's datasets** (not the lists) — likely a bug, since `train_dataset_list` etc. are built but the function returns the loop's last `train_dataset`/`validation_dataset`/`test_dataset` instead of the accumulated lists.
- `set_model(self, model_name, model_config)`: replaces `self.model` with a freshly constructed untrained model.
- `set_dataset(self, subj_list)`: re-runs `get_data` to reload datasets for a new subject list.
- `update_model_weight(self, path_weight)`: `self.model.load_state_dict(torch.load(path_weight, map_location=self.device))`.
- `plot_reconstructed_signal(self, idx_subj, idx_trial, idx_ch, path_weight=None)`: fetches original/reconstructed signal via `compute_reconstructed_signal`, crops to the visualization time window, and plots both curves on one Axes. **Contains a typo bug**: calls `fig.tigh_layout()` instead of `fig.tight_layout()`, which will raise `AttributeError` at runtime.
- `compute_reconstructed_signal(self, idx_subj, idx_trial, path_weight=None)`: optionally reloads weights, fetches `(x, label)` from `self.test_dataset_list[idx_subj][idx_trial]`, runs the model forward pass to get reconstruction `x_r` (assumes `output[0]` is the reconstruction — matches vEEGNet-style multi-output forward), maps integer label to `{0:'left', 1:'right', 2:'foot', 3:'tongue'}`. **Contains a bug**: calls `label.to('cpu')` on `label`, which by this point is a plain Python string (from `label_dict[...]`), not a tensor — `str` has no `.to()` method, so this will raise `AttributeError`.

## Dependencies
- Internal: `library.dataset.preprocess`, `library.config.config_dataset`, `library.training.train_generic.get_untrained_model`.
- External: `torch`, `numpy`, `matplotlib.pyplot`.

## Notes
- **This file appears broken/unfinished in at least two places**: `fig.tigh_layout()` typo in `plot_reconstructed_signal`, and `label.to('cpu')` called on a string in `compute_reconstructed_signal`. Treat as a prototype/WIP tool rather than production-ready code; verify before using interactively.
- `get_data` silently discards per-subject lists and returns only the last subject loaded, despite `self.train_dataset_list` etc. being set from the constructor using the (correctly built, but misleadingly named inside `get_data`) return values — re-check whether `self.train_dataset_list` actually ends up holding single datasets rather than lists, which would make `self.test_dataset_list[idx_subj][idx_trial]` in `compute_reconstructed_signal` index incorrectly (indexing a single dataset by `idx_subj` instead of a list of datasets).
- There's a `# TODO chiedere a giulia il canale della lingua` (Italian: "ask Giulia about the tongue channel") — an open question about which channel corresponds to the "tongue" class, left unresolved; a commented-out `label_to_ch` dict hints at the intended next step.
- Conceptually overlaps with `library/analysis/visualize.py` (function-based plotting) — this file is the class-based, model-aware counterpart specific to D2a + vEEGNet/hvEEGNet reconstruction.
