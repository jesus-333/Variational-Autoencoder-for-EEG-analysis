# `library/model/ChWi.py`

## Purpose
Prototype "Channel-Wise" (ChWi) autoencoder: a 1D-convolutional autoencoder that processes **one EEG channel at a time** (no cross-channel mixing inside the conv stack), optionally applying the exact same encoder weights independently to every channel of a multi-channel signal (`reconstruct_multichannel_EEG`). Also includes a standalone domain-adversarial classifier prototype (`domain_predictor`). Explicitly labeled in its own docstring as "Prototype of a (Ch)annel (Wi)se network."

## Key classes

### `ChWi_module_encoder(nn.Module)`
- One encoder block: `Conv1d` → optional `BatchNorm1d` → optional activation → optional `AvgPool1d`.
- `check_config(module_config)`: in-place-fills defaults (`groups`, `padding`, `use_batch_normalization`, `activation`, `p_kernel`) and raises `ValueError` if `in_channels`/`out_channels`/`c_kernel` are missing.
  - **Bug:** the default-filling lines use `==` instead of `=` for `groups` and `padding` (`module_config['groups'] == 1` / `module_config['padding'] == 0`) — these are no-op comparisons, not assignments, so when `'groups'`/`'padding'` are absent from `module_config`, the subsequent `nn.Conv1d(..., padding=module_config['padding'], groups=module_config['groups'])` call will raise a `KeyError` rather than falling back to the intended defaults. Callers must currently supply `groups` and `padding` explicitly in every module config.
- `forward(x)`: `B x in_channels x T → B x out_channels x T'`.

### `ChWi_encoder_v1(nn.Module)`
- Stacks a list of `ChWi_module_encoder` blocks (one per `config_list` entry) in an `nn.Sequential`.
- `forward(x)`: expects `x` of shape `B x 1 x T` (single channel, **no EEG-channel dimension** — note this is 1D conv over time, depth dim is the conv "channels").
- `reconstruct_multichannel_EEG(x, flatten=False)`: given `x` of shape `B x 1 x C x T` (full multichannel EEG), loops over the `C` EEG channels, calling `forward` on each channel slice `x[:, :, ch_idx, :]` independently (same shared weights for every channel) and stacking results into `B x D x C x T'` (or flattened if requested). This is the method that actually applies the per-channel/shared-weight encoding to a real EEG trial.
- `check_input`: unimplemented stub (`# TODO`, `pass`).

### `ChWi_module_decoder(nn.Module)`
- One decoder block: optional `nn.Upsample` → `Conv1d` (note: a commented-out `ConvTranspose1d` line suggests this was originally meant to use transposed convolution but was switched to a regular `Conv1d`, likely paired with the preceding upsample instead) → optional `BatchNorm1d` → optional activation.
- `forward(x)`: `B x in_channels x T → B x out_channels x T'`.

### `ChWi_decoder_v1(nn.Module)`
- Stacks `ChWi_module_decoder` blocks from `config_list` in an `nn.Sequential`.
- `forward(x)`: expects `x` of shape `B x D x T` (encoded single-channel representation, no EEG-channel dim).

### `ChWi_autoencoder(nn.Module)`
- Full single-channel VAE: `ChWi_encoder_v1` → `support_function.sample_layer` (reused from the hvEEGNet machinery, operating on `Conv2d`-shaped tensors) → `ChWi_decoder_v1`.
- `__init__(config)`: requires `config['encoder_config']` (list for `ChWi_encoder_v1`) and `config['decoder_config']` (list for `ChWi_decoder_v1`), plus `config['sample_layer_config']` (passed to `support_function.sample_layer`; `input_depth` is auto-set from the last encoder module's `out_channels`).
- `forward(x)`: expects `x` of shape `B x 1 x T` (single channel). Encodes → unsqueezes a dummy spatial dim (`x.unsqueeze(2)`, since `sample_layer` is built around `Conv2d`) → samples `(z, mu, log_var)` → squeezes the dummy dim back out → decodes `z` → returns `(x_r, z, mu, log_var)`.

### `domain_predictor(nn.Module)`
- Small MLP intended for domain-adversarial training (predicting, e.g., subject/session identity from latent features, typically paired with a gradient-reversal layer elsewhere — no such layer is present in this file).
- `__init__(config)`: `Linear(input_length, input_length/2) → activation → Linear(.../2, .../4) → activation → Linear(.../4, n_domain) → Softmax()`.
  - **Bugs:** `input_length / 2` and `input_length / 4` use true division (`/`), producing a `float`, which `nn.Linear` will reject as an invalid `in_features`/`out_features` argument (expects `int`) — this class will raise a `TypeError` at construction time unless `config['input_length']` happens to be pre-divided or the caller patches this. Also `nn.Softmax()` is called with no `dim` argument, matching the `# TODO add dim` comment already in the code — PyTorch will emit a deprecation warning and infer the softmax dimension by legacy/implicit behavior rather than the explicit `dim=-1`/`dim=1` that should be set.
- `forward(x)`: `self.predictor(x)`.

## Dependencies
- Internal: `library.model.support_function` (`get_activation`, `sample_layer`).
- External: `torch`, `torch.nn`.

## Notes
- This whole file reads as an early-stage prototype (per its own docstring) with several live bugs that would surface immediately on use: the `==`/`=` typo in `ChWi_module_encoder.check_config`, and the float-division + missing-`dim` issues in `domain_predictor`. Treat as not production-ready; verify against `analysis_script/ChWi/test_autoencoder_v1.py` and `training_scripts/train_ChWi_v1.py` / `library/training/train_ChWi.py` for what configuration actually gets exercised in practice (those configs apparently supply `groups`/`padding` explicitly, sidestepping the encoder bug).
- Reuses `support_function.sample_layer`, which is `Conv2d`-oriented — `ChWi_autoencoder` works around this by inserting/removing a dummy spatial dimension, rather than there being a dedicated 1D sampling layer.
