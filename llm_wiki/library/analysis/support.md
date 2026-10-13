# `library/analysis/support.py`

## Purpose
Grab-bag of support functions for analysis scripts: dataset+model construction helpers for specific experiments (D2a, Zhou2016), reconstruction-loss computation over a whole dataset, signal cropping/spectra utilities, and aggregation of saved reconstruction-error results across training repetitions. This is one of the most "experiment glue code" heavy files in the library — tightly coupled to the on-disk result layout used by `analysis_script/`.

## Key functions

### `get_dataset_and_model(dataset_config, model_name)`
- Builds the D2a dataset (via `library.dataset.preprocess.get_dataset_d2a`) and an untrained model config for either `'hvEEGNet_shallow'` or `'vEEGNet'` (raises `ValueError` otherwise), using `library.config.config_model`.
- Hardcodes `C = 22` (D2a's channel count) and derives `T` from `dataset_config['trial_start']`/`trial_end'`/sampling rate.
- Marked `# TODO Rewrite the function`.
- Returns `(train_dataset, validation_dataset, test_dataset, model)`.

### `get_dataset_zhou2016_and_model(dataset_config, model_config, model_name)`
- Downloads Zhou2016 train/test data via `library.dataset.download.get_Zhou2016`, adds a depth dimension, trims the last time sample if not resampled (dataset quirk: 1251 raw samples instead of 1250), optionally splits train into train/validation via `library.dataset.support_function.get_idx_to_split_data`, updates `model_config['encoder_config']` with inferred `C`/`T`, wraps arrays in `library.dataset.dataset_time.EEG_Dataset`, and builds the model via `library.training.train_generic.get_untrained_model`.
- Returns `(dataset_train, dataset_validation, dataset_test, model)`.

### `compute_loss_dataset(dataset, model, device, batch_size=32, use_dtw_divergence=False, scale_before_computation=False)`
- Computes a `n_trials x n_channels` matrix of reconstruction loss using `SoftDTW` (from `library.training.soft_dtw_cuda`), channel-by-channel, over the whole dataset via a `DataLoader`.
- `use_dtw_divergence=True` computes the DTW **divergence** (`dtw(x,x_r) - 0.5*(dtw(x,x) + dtw(x_r,x_r))`) instead of plain DTW distance — removes the self-similarity bias of soft-DTW.
- `scale_before_computation=True` min-max scales each channel to `[0,1]` before computing the loss.
- Returns a NumPy array.

### `compute_dtw_between_two_batch(batch_1, batch_2, use_cuda=True)`
- Same per-channel DTW computation pattern as `compute_loss_dataset` but operating directly on two pre-loaded batches (shape `B x 1 x C x T`) instead of a dataset/dataloader.

### `crop_signal(x, idx_ch, t_start, t_end, t_min, t_max)`
- Selects a single channel `idx_ch` from signal `x` and crops the time axis to `[t_min, t_max]` given the signal spans `[t_start, t_end]`. Returns `(x_crop, t_plot)`.

### `compute_latent_space_different_resolution(model, x)`
- Reconstructs `x` three times using a hierarchical VAE (`model.h_vae.reconstruct_ignoring_latent_spaces`), progressively ignoring more of the deeper latent spaces (`[False,False,False]`, `[True,False,False]`, `[True,True,False]`), to visualize how each hierarchy level contributes detail to the reconstruction.

### `compute_spectra_magnitude_and_phase(x, fs, keep_only_positive_frequency=True)`
- FFT-based magnitude/phase/frequency-axis computation via `scipy.fft`, with shifting (`fftshift`) and optional truncation to non-negative frequencies.

### `skip_training_run(subj, repetition)`
- Hardcoded blacklist of `(subject, repetition)` pairs to exclude from aggregated-result computations because "the training run avoided presents some anomalies." Specific hardcoded values: subj 4 reps {17,18,19}; subj 5 rep 19; subj 8 reps {3,14,16}.

### `compute_average_and_std_reconstruction_error(tot_epoch_training, subj_list, epoch_list, repetition_list, method_std_computation=2, skip_run=False, use_test_set=False)`
- Loads `.npy` reconstruction-error files from a hardcoded path pattern: `'Saved Results/repetition_hvEEGNet_{tot_epoch_training}/{train|test}/subj {subj}/recon_error_{epoch}_rep_{repetition}.npy'` for every subject/epoch/repetition combination, and aggregates mean/std across repetitions.
- `method_std_computation`: `1` = std across channels then averaged; `2` = mean across channels then std across repetitions (default); `3` = std of the whole matrix.
- Missing files are caught with a bare `except:` and a printed warning, then simply skipped (not re-raised) — silent data loss if paths are wrong.
- Returns 4 dicts keyed by `subj -> epoch -> value`: `recon_loss_results_mean`, `recon_loss_results_std`, `recon_loss_to_plot_mean`, `recon_loss_to_plot_std`.

### `compute_average_spectra(data, nperseg, fs, idx_ch)`
- Computes Welch PSD (`scipy.signal.welch`) per trial for a single channel `idx_ch`, then returns the across-trial mean/std spectra and frequency axis.

## Dependencies
- Internal: `library.config.config_model`, `library.dataset.preprocess`, `library.dataset.download`, `library.dataset.support_function`, `library.dataset.dataset_time`, `library.training.train_generic`, `library.training.soft_dtw_cuda`.
- External: `torch`, `numpy`, `scipy.fft`, `scipy.signal`.

## Notes
- `compute_average_and_std_reconstruction_error` depends on a **relative** hardcoded path (`'Saved Results/...'`) — only works if the script is run from the repository root; this is a recurring pattern across `analysis_script/`.
- `skip_training_run`'s hardcoded subject/repetition exclusions are specific to one particular `repetition_hvEEGNet_*` experiment; not general-purpose.
- The per-channel DTW loop here duplicates the one in `library/analysis/dtw_analysis.py:compute_recon_error_between_two_tensor` — three near-identical implementations exist across the codebase (`dtw_analysis.py`, here twice as `compute_loss_dataset`/`compute_dtw_between_two_batch`).
- `get_dataset_and_model` is explicitly marked for rewriting by the author.
