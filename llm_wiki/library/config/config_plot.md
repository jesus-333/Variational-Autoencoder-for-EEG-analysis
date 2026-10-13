# `library/config/config_plot.py`

## Purpose
Module docstring: "Old config for various plots in the analysis scripts. Not in use as of today. For a plot of plot the config are directly declared inside the analysis script." Kept for reference/legacy reasons only — new analysis scripts under `analysis_script/` declare their plot configs inline rather than importing from here.

## Key functions

### `get_config_plot_preprocess_random_trial() -> dict`
`n_trials_to_plot=3`, `ch_to_plot='C3'`, `t_start=0`, `t_end=6` (seconds), `figsize=(15,10)`, `fontsize=12`, `show_fig=True`, `save_plot=True`.

### `get_config_plot_preprocess_ERS() -> dict`
`ch_to_plot=['C3','Cz','C4']`, `label_to_plot=[0,1,2]` (an alternative `[1,2,3]` is commented out), `figsize=(18,12)`, `fontsize=12`, `show_fig=True`, `save_plot=True`.

### `get_config_plot_preprocess_average_stft() -> dict`
`ch_to_plot='C3'`, `t_start=0`, `t_end=6`, `band_start=8`, `band_end=13` (Hz band to average, e.g. alpha band), `n_trials_to_average=20`, `figsize=(15,10)`, `fontsize=12`, `show_fig=True`, `save_plot=True`.

### `get_config_latent_space_computation() -> dict`
`sample_from_distribution=False`, `compute_recon_error=True`, `perplexity=30`, `n_iter=1000` (t-SNE parameters).

### `get_config_latent_space_representation() -> dict`
`figsize=(25,10)`, `fontsize=12`, `show_fig=True`, `save_plot=False`, `colormap='Reds'`, `markersize=40`.

### `get_config_set_of_trials() -> dict`
`idx_start=166`, `idx_end=172`, `idx_ch=6`, `trial_length=4`, `figsize=(15,10)`, `fontsize=15`, `add_trial_line=True`.

### `get_config_visualizer() -> dict`
`model_name='vEEGNet'`, `model_config=cm.get_config_vEEGNet()` — **broken as written**, see **Notes**.

### `get_config_plot_reconstruction(idx_trial: int, idx_ch: int, compute_psd: bool) -> dict`
`idx_trial`, `idx_ch`, `compute_psd`, `figsize=(15,10)`, `fontsize=15`.

### `get_style_per_subject_error_plus_std() -> dict`
Hardcoded per-subject plot styling for subjects 1–9 (matching the 9-subject d2a dataset), each a dict of `color`, `marker`, `linestyle` (e.g. `subj_1 = dict(color='blue', marker='o', linestyle='solid')`).

## Dependencies
- Internal: `from . import config_model as cm` (used only by `get_config_visualizer`).
- External: none.

## Notes
- Per the module docstring, this file is **not used** by current analysis scripts — treat it as historical/reference material, not a live dependency. Before reusing any function here, check whether `analysis_script/` already has an inline equivalent.
- `get_config_visualizer()` calls `cm.get_config_vEEGNet()` with **no arguments**, but `get_config_vEEGNet` in `config_model.py` requires `C`, `T`, `hidden_space`, `type_encoder`, `type_decoder` as positional arguments — calling `get_config_visualizer()` as written will raise a `TypeError`. Confirms the "not in use" status of this module.
- `get_style_per_subject_error_plus_std` hardcodes exactly subjects 1–9, implicitly tying it to the 9-subject BCI Competition IV 2a (d2a) dataset; it would need to be extended for datasets with a different subject count.
