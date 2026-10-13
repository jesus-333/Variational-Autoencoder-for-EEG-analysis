# `library/model/hvEEGNet.py`

## Purpose
"Shallow" hierarchical vEEGNet (hvEEGNet, V1): builds a standard `vEEGNet`, tears it apart into its 3 constituent conv blocks per side, and feeds those 3-level lists into the generic `hierarchical_VAE.hVAE` engine. This is the model behind the hvEEGNet preprint. For the newer, deeper, residual-cell-based variant see `hvEEGNet_V2.py`.

## Key classes

### `hvEEGNet_shallow(nn.Module)`
- `__init__(config)`:
  1. `build_cell_list(config)` constructs a throwaway `vEEGNet.vEEGNet(config)`, then extracts:
     - encoder cells: `[temporal_filter, spatial_filter, separable_convolution]` (the 3 `EEGNet` blocks, shallow→deep).
     - decoder cells: `[separable_convolution_transpose, spatial_convolution_transpose, temporal_convolution_transpose]` (the matching 3 `Decoder_EEGNet` blocks, deep→shallow — i.e. already in the order the generic hierarchy expects for its decoder).
  2. Wraps these in `hierarchical_VAE.hVAE(encoder_cell_list, decoder_cell_list, config)` as `self.h_vae`.
  3. Runs a dummy `(1,1,C,T)` forward through `h_vae` to size the classifier head: `n_neurons = len(mu_list[0].flatten()) * 2` (mean+logvar of the deepest latent concatenated).
  4. If `config['use_classifier']`, builds `clf = Linear(n_neurons, n_classes) + LogSoftmax`.
  - Config keys: everything `vEEGNet` needs (`type_encoder`, `encoder_config`, `type_decoder`, `parameters_map_type`, `hidden_space`, etc. — see `vEEGNet.md`) plus everything `hierarchical_VAE.hVAE` needs (`convert_logvar_to_var`, `use_h_in_decoder`, optionally `hidden_space_dimension_list`), plus `use_classifier`, `n_classes`.
- `forward(x)`: `h_vae(x) → (x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)`; if `use_classifier`, concatenates the **deepest** level's `(mu_list[0], log_var_list[0])`, flattens, classifies, and returns a 6-tuple with `label` appended; else returns the 5-tuple as-is.
- `generate(z=None)`: delegates to `h_vae.generate(z)`.
- `reconstruct(x, no_grad=True)`: delegates to `h_vae.reconstruct(x, no_grad)`.
- `reconstruct_ignoring_latent_spaces(x, latent_space_to_ignore)`: requires `len(latent_space_to_ignore) == 3` (hardcoded — hvEEGNet V1 always has exactly 3 hierarchy levels), delegates to `h_vae.reconstruct_ignoring_latent_spaces`.
- `encode(x, return_distribution=True)`: delegates to `h_vae.encoder.encode`. If `return_distribution`, returns `(z, mu, sigma)` from the deepest level; if not, returns just the raw final-cell output tensor.
- `classify(x, return_as_index=True)`: requires `use_classifier=True` (raises `ValueError` otherwise); forwards, extracts `label = output[5]`, optionally converts to class index via `argmax`.
- `dtw_comparison_2(x, device='cpu')`: computes per-channel Soft-DTW distance between `x` and its reconstruction. Moves model+input to `device`, forwards, then for each EEG channel independently computes `SoftDTW(use_cuda=(device=='cuda'), normalize=False)` between the channel's reconstructed and original signal (each reshaped to `B x T x 1` via `swapaxes(1,2)`), collecting a `B x C` numpy distance matrix. Always moves model and input back to `'cpu'` before returning.

## Dependencies
- Internal: `library.model.vEEGNet`, `library.model.hierarchical_VAE`, `library.training.soft_dtw_cuda.SoftDTW`.
- External: `torch`, `torch.nn`, `numpy`.

## Notes
- The number "3" (temporal/spatial/separable-conv split) is hardcoded as the hierarchy depth throughout this file (`build_cell_list`, the `len(...) != 3` check in `reconstruct_ignoring_latent_spaces`) — this is what "shallow" refers to, as opposed to `hvEEGNet_V2`'s configurable `n_cells`.
- Building a full `vEEGNet` just to cannibalize its submodules means the decoder-selection logic and all config validation in `vEEGNet.__init__`/`check_model_config` still applies (e.g. the MBEEGNet-decoder-not-implemented caveat from `vEEGNet.md` also applies here if `type_encoder == 1`).
- This file imports `library.training.soft_dtw_cuda` — an unusual model→training dependency direction (normally one expects training code to depend on model code, not vice versa); keep this in mind when reorganizing or extracting `library/model` as a standalone package (e.g. into a `jade`-style repo as mentioned by the author) since `dtw_comparison_2` would need `soft_dtw_cuda` to come along.
