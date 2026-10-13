# `library/model/hierarchical_VAE.py`

## Purpose
Generic, architecture-agnostic implementation of a deep hierarchical VAE in the style of [NVAE](https://arxiv.org/abs/2007.03898). This is the reusable "engine" that both `hvEEGNet.py` (3 fixed levels built from a torn-apart `vEEGNet`) and `hvEEGNet_V2.py` (N identical residual EEGNet cells) are built on top of. It is NOT EEG-specific — it just takes lists of arbitrary `nn.Module` "cells" for the encoder and decoder and wires them into a multi-level latent-variable hierarchy.

## Key concepts
- A **"level" of hierarchy** = one entry in `encoder_cell_list` / `decoder_cell_list`. The encoder's cells are applied in order (cell 0 = shallowest/earliest, last cell = deepest). The decoder's cells conceptually run in reverse (from the deepest latent space back out to the reconstruction), combining each decoder stage with a sample from a level-specific latent variable.
- Every level except the very deepest one has its own latent variable `z_i` with a `(mu_i, log_var_i)`; the deepest latent variable is produced directly by the encoder's final `sample_layer_z`.
- During training, the decoder's prior-like sampling at each level is corrected using the corresponding encoder output (a learned "delta" `(delta_mu, delta_log_var)` added to the decoder's own `(mu, log_var)` before reparameterizing) — this is the NVAE "relative" latent correction (paper section 3.2). During generation (no encoder outputs available), this correction is skipped and the decoder samples from its own unconditional distribution at each level.

## Key classes

### `hVAE(nn.Module)`
- Top-level module combining an `hvae_encoder` and an `hVAE_decoder`.
- `__init__(encoder_cell_list, decoder_cell_list, config)`: builds the encoder, runs a dummy `(1, 1, C, T)` tensor through `encoder.encode(..., return_shape=True)` to compute `config['encoder_outputs_shape']` (shapes of every encoder cell's output, needed to size the decoder's sampling/combination layers), then builds the decoder with that augmented config. Also stores `hidden_space_shape`/`hidden_space_size_flatten` from the deepest encoder output shape.
  - Requires `config['encoder_config']['C']` and `config['encoder_config']['T']` for the dummy forward pass.
- `forward(x, h=None)`: encodes `x` → `(z, mu, log_var, encoder_cell_outputs)`, decodes with `(z, h, encoder_cell_outputs)` → `(x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)`; prepends the deepest-level `(mu, log_var)` to the lists so index 0 is always the deepest level.
- `generate(z=None)`: if `z` is `None`, samples `z ~ N(0,I)` of shape `hidden_space_shape`; runs only the decoder (`encoder_cell_output=None` → no corrections) to produce a new sample `x`.
- `reconstruct(x, no_grad=True)`: full forward pass, returns only `x_r`.
- `reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore: list[bool])`: encodes `x`, zeroes out selected `encoder_cell_outputs` entries (so the decoder's "correction" step at that level sees all-zero encoder output and — per the `if ... torch.sum(...) != 0` check inside `hVAE_decoder.forward` — skips the correction, falling back to the decoder's own prior at that level), then decodes. Used for ablation studies of individual latent levels' contribution to reconstruction (see `hvEEGNet_shallow.reconstruct_ignoring_latent_spaces`, which requires exactly 3 entries).

### `hvae_encoder(nn.Module)`
- `__init__(encoder_cell_list, config)`: wraps the cells in an `nn.ModuleList`. Runs a dummy forward through `encode(..., return_distribution=False)` to get the final cell's output shape, then builds `sample_layer_z` (a `support_function.sample_layer`) — using `config['hidden_space_dimension_list'][0]` if present (vector/FF latent spaces, one dimension per level) or defaulting to the convolutional/matrix mapping otherwise. Reads `config['convert_logvar_to_var']` — if `True`, the "sigma" returned is `exp(log_var)` (i.e. actually variance, not log-variance) at both the encoder's deepest level and in `forward`.
- `forward(x)`: calls `encode(x, return_distribution=False)`, passes the final cell output through `sample_layer_z` to get `(z, mu, sigma)`, optionally exponentiates `sigma`, and returns `(z, mu, sigma, cell_outputs)` where `cell_outputs` is the list of every intermediate cell's raw output (needed by the decoder for lateral connections).
- `encode(x, return_distribution=True, return_shape=False)`: runs `x` through every cell in order, collecting `cell_outputs` (and optionally their shapes). If `return_distribution`, additionally samples from the deepest latent space and prepends `(z, mu, sigma)` to the return list; otherwise just returns the raw final cell output. Always appends `cell_outputs` (and shapes if requested) to the return list. This is the method used both for the real forward pass and for the various dummy-shape-probing calls elsewhere in the file.

### `hVAE_decoder(nn.Module)`
- `__init__(decoder_cell_list, config)`: the most complex constructor in the file — calls `self.build_decoder(...)` under `torch.no_grad()` to construct, in one pass, 5 parallel `nn.ModuleList`s:
  - `decoder`: the actual decoder cells (as given).
  - `sample_layers_z`: per-level sampling layers used during *generation* (prior).
  - `sample_layers_z_given_x`: per-level sampling layers used during *training* to produce the `(delta_mu, delta_log_var)` correction, conditioned on encoder+decoder features combined.
  - `features_combination_z`: `weighted_sum_tensor` modules that merge a decoder-stage feature map with the corresponding encoder output (for computing the correction).
  - `features_combination_decoder`: `weighted_sum_tensor` modules that merge the running decoder feature map with the previous level's sampled `z` (or, at the deepest level, with `h` if `config['use_h_in_decoder']`, else nothing special).
  - Reads `config['encoder_outputs_shape']` (list of shapes, deepest level last — populated by `hVAE.__init__`), `config['hidden_space_dimension_list']` (optional, per-level vector latent dims), `config['use_h_in_decoder']`, `config['h_shape']` (if using `h`).
- `forward(z, h=None, encoder_cell_output=None)`: iterates over decoder levels `i = 0..N-1` (0 = deepest):
  - Combines running feature map `x` (init to `h` if given, else zeros_like `z`) with `z` via `features_combination_decoder[i]`, then runs it through `decoder[i]`.
  - For every level except the last (shallowest/output level), samples a new `z` from `sample_layers_z[i]` and records `(mu, log_var)`.
  - If `encoder_cell_output is not None` (training mode) and the corresponding encoder output isn't all-zero (ablation support), computes `(delta_mu, delta_log_var)` from `sample_layers_z_given_x[i]` applied to `features_combination_z[i](x, encoder_cell_output[-2-i])`, then re-samples the "corrected" `z` via `reparametrize(mu + delta_mu, log_var + delta_log_var, return_as_tensor=True)`.
  - Returns `(x, mu_list, log_var_list)` during generation, or `(x, mu_list, log_var_list, delta_mu_list, delta_log_var_list)` during training.
  - **Note:** contains several leftover debug `print(...)` statements reporting NaN counts in intermediate tensors (`x`, `encoder_cell_output`, `delta_mu`, `delta_log_var`, `z`) — this is verbose NaN-debugging instrumentation, matching similar prints in `support_function.py`'s `sample_layer`/`reparametrize`. Also contains a `# TODO check after training the use of identity for the i == 0 if h is not passed` comment.
- `build_decoder(decoder_cell_list, config)`: constructs the 5 module lists described above by symbolically running dummy tensors through the would-be forward pass (shape-probing only, under `no_grad`), to correctly size every intermediate `weighted_sum_tensor`/`sample_layer`. Encoder output indexing uses `-1-i` (reverse order, since encoder outputs are stored shallow→deep but the decoder consumes deep→shallow) with an off-by-one adjustment to `-2-i` when actually reading `encoder_cell_output` in `forward` (documented in a long inline comment: index `-1` is the encoder's *input*, so the first usable paired-level output is at `-2`).

## Dependencies
- Internal: `library.model.support_function` (`sample_layer`, `weighted_sum_tensor`).
- External: `torch`, `torch.nn`.

## Notes
- This file is intentionally model-agnostic — it has no knowledge of EEG, EEGNet, or convolution shapes beyond what it infers by probing the cells it's given. Any new "hv-style" model (beyond hvEEGNet/hvEEGNet_V2) can likely reuse it directly by supplying appropriate `encoder_cell_list`/`decoder_cell_list`.
- Numerous NaN-tracking `print()` statements throughout `hVAE_decoder.forward` suggest this hierarchy was prone to NaN propagation during training/debugging; a future cleanup could gate these behind a debug flag or remove them once root-caused.
- The `-2-i` indexing logic in `hVAE_decoder.forward`/`build_decoder` is subtle and easy to get wrong when modifying the number of levels — read the inline comments in `build_decoder` carefully before changing level counts or decoder cell ordering.
