# `library/model/vEEGNet.py`

## Purpose
vEEGNet: a (single-level, non-hierarchical) variational autoencoder whose encoder and decoder are EEGNet/MBEEGNet-based. This is the base architecture that `hvEEGNet.py` later decomposes into 3 stages and feeds into the generic hierarchical VAE (`hierarchical_VAE.py`).

## Key classes

### `vEEGNet(nn.Module)`
- `__init__(config)`: validates `config` via `self.check_model_config`, then:
  1. Builds `cnn_encoder`: `EEGNet` if `config['type_encoder'] == 0`, `MBEEGNet` if `== 1`.
  2. Runs `encoder_shape_info` (a dummy forward pass with a `1 x 1 x C x T` input) to determine `n_input_neurons` and `decoder_ouput_shape` (the encoder's output tensor shape, batch dim set to `-1`).
  3. Builds `sample_layer = support_function.sample_layer(decoder_ouput_shape, config, config['hidden_space'])` — the encoder→latent bridge (see `support_function.md`). Stores `parameters_map_type`.
  4. Sets `self.hidden_space` = `n_input_neurons` if `parameters_map_type == 0` (matrix latent space — this overrides whatever `config['hidden_space']` was, since the "hidden space" is really the conv feature-map size), else `config['hidden_space']` for `1`/`2` (vector latent space).
  5. (Optional) builds a classifier head `Linear(hidden_space*2, n_classes) + LogSoftmax` if `config['use_classifier']` — takes the concatenation of `z_mean` and `z_log_var`.
  6. Builds the decoder: copies encoder-config values needed by the decoder into `config['encoder_config']` (`dimension_reshape = decoder_ouput_shape`, `hidden_space`, `parameters_map_type`), then picks `Decoder_EEGNet.EEGNet_Decoder_Upsample` (`type_decoder == 0`) or `EEGNet_Decoder_Transpose` (`type_decoder == 1`) when `type_encoder == 0`; if `type_encoder == 1` it instead builds a bare `MBEEGNet` as a non-functional placeholder decoder (see Notes — the real MBEEGNet decoder was never implemented).
  - Config dict keys: `type_encoder` (0=EEGNet, 1=MBEEGNet), `encoder_config` (nested dict for the chosen encoder, reused/mutated for the decoder too), `type_decoder` (0=Upsample, 1=Transpose), `hidden_space`, `parameters_map_type` (0/1/2, see `support_function.md`), `use_activation_in_sampling`, `sampling_activation`, `use_classifier`, `n_classes`, `type_vae` (present in example configs but not read inside this file — conditional-VAE support is marked "not implemented" in the README), `print_var` (optional, default `True`).
- `forward(x)`: `x → cnn_encoder → sample_layer → (z, z_mean, z_log_var) → decoder(z) → x_r`. If `use_classifier`: concatenates `(z_mean, z_log_var)` (flattened if `parameters_map_type == 0`) and runs the classifier, returning `(x_r, z_mean, z_log_var, predicted_label)`; else returns `(x_r, z_mean, z_log_var)`.
- `reconstruct(x, no_grad=True)`: runs `forward` (optionally under `torch.no_grad()`) and returns only `output[0]` (the reconstruction).
- `generate()`: samples `z ~ N(0, I)` of shape `(1, hidden_space)` and decodes it directly — **note:** this only produces a sensible-shaped output when `parameters_map_type` is `1`/`2` (vector latent space matching `hidden_space`); for `parameters_map_type == 0` the decoder expects a feature-map-shaped `z`, so this path is likely broken/unused in that configuration.
- `encoder_shape_info(C, T)`: dummy-forward helper (see above).
- `check_model_config(config)`: validates `type_encoder` (0 or 1) and `type_decoder` (0 or 1), raising `ValueError` otherwise; optionally prints which was selected.
- `classify(x, return_as_index=True)`: requires `use_classifier=True`; forwards and extracts the 4th output element, converts log-probs to a class index via `argmax` if requested.

## Dependencies
- Internal: `library.model.EEGNet`, `library.model.MBEEGNet`, `library.model.Decoder_EEGNet`, `library.model.support_function`.
- External: `torch`, `torch.nn`.

## Notes
- The MBEEGNet encoder path (`type_encoder == 1`) has no working decoder — see the `# TODO Implement MBEEGNet decoder` / "indefinitely postponed" comment in the code. Treat `vEEGNet` with `type_encoder=1` as encoder-only/classification-only in practice.
- `config['encoder_config']` is mutated in-place during `__init__` to add decoder-specific keys (`dimension_reshape`, `hidden_space`, `parameters_map_type`) — if the same `encoder_config` dict is reused to build multiple models, later keys will leak across instances.
- `generate()` is only reliable for `parameters_map_type` in `{1, 2}`.
- This class is the architecture that `hvEEGNet.hvEEGNet_shallow.build_cell_list` tears apart into its 3 conv blocks (`temporal_filter`, `spatial_filter`, `separable_convolution` from the encoder; the 3 corresponding transpose blocks from the decoder) to feed into the generic hierarchical VAE.
