# `library/training/loss_function.py`

## Purpose
Defines every loss used to train the classifiers and (hierarchical) VAEs in this repo: plain classification loss, reconstruction loss (MSE or soft-DTW variants), KL-divergence terms (plain and hierarchical/"shifted" per the NVAE paper), and the combined weighted losses for vEEGNet and hvEEGNet, exposed both as free functions and as stateful `*_loss` classes.

## Key classes

### `vEEGNet_loss`
- Stateful loss object built from a `config` dict; used for the plain (single-level) VAE (`vEEGNet`, and reused for `ChWi_autoencoder` in `train_generic.get_loss_function`).
- Constructor reads: `recon_loss_type` (0 = MSE/L2; 1/2/3/4 = SDTW / SDTW-divergence / Block-SDTW / Block-SDTW-Divergence), `gamma_dtw` (default 1, only for DTW types), `device` (to decide `use_cuda` for `SoftDTW`), `edge_samples_ignored` (default 0 — trims that many samples off both ends of the signal before computing reconstruction loss, to avoid padding artifacts), `alpha`/`beta`/`gamma` (default 1 each — weights for recon/KL/classifier terms), and defaults `average_channels`/`average_time_samples` to `False` with a printed warning if missing (relevant only for DTW losses).
- `compute_loss(x, x_r, mu, log_var, predicted_label=None, true_label=None)`: returns `(final_loss, recon_loss, kl_loss)` or, if labels given, `(final_loss, recon_loss, kl_loss, clf_loss)`. `final_loss = alpha*recon + beta*kl [+ gamma*clf]`.
- `compute_recon_loss(x, x_r)`: slices off `edge_samples_ignored` samples from both ends of the time axis, then applies MSE or `compute_dtw_loss_along_channels` depending on `recon_loss_type`.

### `hvEEGNet_loss(vEEGNet_loss)`
- Subclasses `vEEGNet_loss`, overriding `self.kl_loss_function` with `hierarchical_KL` (operates on lists of `mu`/`log_var` per hierarchy level rather than single tensors).
- `compute_loss(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predicted_label=None, true_label=None)`: same weighting scheme as parent, but also returns the per-level `kl_loss_list` alongside the summed `kl_loss`.

Note: there are also two module-level free functions literally named `vEEGNet_loss` and `hvEEGNet_loss` (older-style functional API, defined *before* the classes of the same name in the file) — **the class definitions below them shadow the functions** at module scope, so `loss_function.vEEGNet_loss` and `loss_function.hvEEGNet_loss` always resolve to the classes when imported elsewhere (this is what `train_generic.get_loss_function` uses). The free functions are dead code, unreachable by name from outside the module.

## Key functions
- `classifier_loss(predicted_label, true_label)` — `F.nll_loss`; expects log-probabilities.
- `recon_loss_function(x, x_r)` — plain `F.mse_loss`.
- `recon_loss_frequency_function(x, x_r)` — **stub, `pass`, returns `None`**. Referenced by `vae_loss_function` when `recon_loss_type == 1` but never implemented; calling it will silently propagate `None` into the loss arithmetic and crash downstream.
- `compute_dtw_loss_along_channels(x, x_r, dtw_loss_function, config)` — core DTW reconstruction-loss routine. Inputs shaped `B x 1 x C x T`; loops over EEG channels, computes SoftDTW per channel (type 1 = classic, type 2 = SDTW divergence `dtw(x,y) - 0.5*(dtw(x,x)+dtw(y,y))`, type 3/4 = block variants via `block_sdtw`), accumulates and optionally averages over channels/time per `config['average_channels']`/`config['average_time_samples']`. Returns a tensor of shape `B`.
- `block_sdtw(x, x_r, dtw_loss_function, block_size, soft_DTW_type, normalize_by_block_size=True)` — splits the (already channel-selected) signal into contiguous blocks of `block_size` samples and sums (optionally block-size-normalized) SDTW/SDTW-divergence loss over blocks, to reduce memory/compute versus running DTW over the full sequence length.
- `kl_loss_normal_function(mu, log_var, delta_mu=None, delta_log_var=None)` — standard closed-form KL(N(mu,var) || N(0,1)) when no delta args given; otherwise the NVAE "shifted" KL (Section 3.2) used for non-top hierarchy levels.
- `kl_loss_function(mu_1, log_var_1, mu_2, log_var_2)` — **stub, `pass`**. Intended general two-Gaussian KL; unused/dead.
- `hierarchical_KL(mu_list, log_var_list, delta_mu_list, delta_log_var_list)` — sums `kl_loss_normal_function` across hierarchy levels; level 0 uses the plain KL, deeper levels use the shifted KL with the corresponding `delta_mu`/`delta_log_var` (index `i-1`, since the deepest level has no shift term). Returns `(kl_loss_sum, kl_loss_list)`.
- `vae_loss_function(x, x_r, mu, log_var, config)` — plain VAE loss: `alpha*recon + beta*kl`, dispatching recon type via `config['recon_loss_type']` (0 = MSE, 1 = the unimplemented frequency loss — anything else raises `ValueError`).

## Dependencies
- Internal: `library.training.soft_dtw_cuda.SoftDTW`.
- External: `torch`, `torch.nn.functional`.

## Notes
- `recon_loss_frequency_function` and the module-level `kl_loss_function` are unimplemented stubs (`pass`) — any code path reaching `recon_loss_type == 1` in `vae_loss_function` will fail with a confusing downstream error since `recon_loss_frequency_function` returns `None` rather than raising immediately.
- The free-function/class name collision (`vEEGNet_loss`, `hvEEGNet_loss` both defined as a function and then as a class) is unusual Python but intentional-looking: by definition order the class always wins. Be careful if refactoring — moving the class definition before the function, or vice versa, changes which one is importable.
- `compute_dtw_loss_along_channels` and `block_sdtw` are the main cost driver during DTW-based training (hvEEGNet papers): looping per-channel and optionally per-block means loss computation scales with both channel count and (for block variants) `T / block_size`.
