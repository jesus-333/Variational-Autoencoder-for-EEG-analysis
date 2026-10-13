# `library/config/config_model.py`

## Purpose
Factory functions returning example/default config `dict`s for every model family in `library/model/` (EEGNet, MBEEGNet, vEEGNet, hierarchical vEEGNet/hvEEGNet, ChWi autoencoder, classifier-on-top-of-hvEEGNet, aEEGNet/attention). The module docstring explicitly warns: "the specific parameters present are intended as examples only" — these are starting points to copy and edit, not fixed production configs.

## Key functions

### `get_config_EEGNet(C: int, T: int) -> dict`
Base EEGNet config, also reused (nested) by MBEEGNet/vEEGNet/hvEEGNet configs.
- `c_kernel_1` `(1,128)`, `c_kernel_2` `(C,1)`, `c_kernel_3` `(1,32)` — temporal/spatial/separable-conv kernels.
- `filter_1=8`, `filter_2=16` — depth-map counts for temporal and separable convolutions.
- `p_kernel_1=(1,4)`, `p_kernel_2=(1,8)` — pooling kernels, must have form `(1, x)`.
- `C`, `T` — number of EEG channels / time samples.
- `D=2` — depth multiplier.
- `activation='elu'` — one of `relu`, `elu`, `selu`, `gelu`.
- `use_bias=False`, `prob_dropout=0.5`, `use_dropout_2d=True` (Dropout2d vs Dropout), `flatten_output=True` (flattens tensor after separable conv).

### `get_config_EEGNet_classifier(C, T, n_classes) -> dict`
`get_config_EEGNet` + `n_classes`, forces `flatten_output=True`.

### `get_config_EEGNet_stft_classifier(C, T, n_channels) -> dict`
`get_config_EEGNet` variant for STFT input: sets `depth_first_layer=n_channels`, forces all three kernels to `(7,7)`, `n_classes=4` (hardcoded), `flatten_output=True`.

### `get_config_MBEEGNet(C, T) -> dict`
Three parallel EEGNet branches with different temporal kernels: `temporal_kernel_1/2/3 = (1,64)/(1,16)/(1,4)`, `dropout_1/2/3=0.5` each, plus `C`, `T`, and a nested `eegnet_config` (shared settings for layers after the first, from `get_config_EEGNet`).

### `get_config_MBEEGNet_classifier(C, T, n_classes) -> dict`
`get_config_MBEEGNet` + `n_classes`, forces `eegnet_config['flatten_output']=True`.

### `get_config_vEEGNet(C, T, hidden_space, type_encoder, type_decoder) -> dict`
- `type_encoder`: 0 = EEGNet, 1 = MBEEGNet (not implemented per README — see **Notes**, there is a real bug here).
- `hidden_space` (int, ignored unless `parameters_map_type == 0`), `encoder_config` (nested, reused for decoder), `type_decoder` (upsample vs transposed conv — see `Decoder_EEGNet.py`), `type_vae=0` (0=normal VAE, 1=conditional VAE, not implemented), `n_classes=4`, `use_classifier=False`, `parameters_map_type` (0=convolution, 1=feed-forward, 3=feed-forward with LoRA — maps encoder output to mean/logvar), `rank=8` (LoRA rank, only used if `parameters_map_type==3`), `use_spectral_norm=False` (applies to feed-forward layers if `parameters_map_type` is 1 or 2), `use_activation_in_sampling=True`, `sampling_activation='elu'`.

### `get_config_hierarchical_vEEGNet(C, T, type_decoder=0, parameters_map_type=0) -> dict`
Config for hvEEGNet. Builds an `encoder_config` from `get_config_EEGNet` with `flatten_output=True` (legacy, unused for hvEEGNet), `p_kernel_1=None` (no pooling), `p_kernel_2=(1,10)`. Top-level keys: `hidden_space=1` (unused placeholder, required only for the full standard-vEEGNet construction step before hvEEGNet repurposes its modules), `type_encoder=0` (docstring says "Ignore", keep 0), `encoder_config`, `type_decoder` (0=upsample, 1=transposed conv — "Keep the value to 0" per docstring), `type_vae=0`, `n_classes=4` (ignored unless classifier used), `use_h_in_decoder=False` (NVAE Fig. 2 style extra input to deepest latent level — ignore), `use_activation_in_sampling=True`, `sampling_activation='elu'`, `convert_logvar_to_var=False` (if True, model returns variance instead of log-variance), `hidden_space_dimension_list=[32,128,512]` (only relevant if `parameters_map_type==1`), `parameters_map_type` (passed through — "Keep the value to 0" per docstring, i.e. convolution), `rank=16` (LoRA rank, used only if `parameters_map_type==2`), `use_spectral_norm=False`, `use_classifier=False`. Also forces `encoder_config['print_var']=False`.

### `get_config_ChWiAutoencoder(n_modules: int) -> dict`
`encoder_config` (via `get_config_ChWi_encoder`), `decoder_config` (via `get_config_ChWi_decoder`), `sample_layer_config` (`parameters_map_type=0`, `use_activation_in_sampling=True`, `sampling_activation='elu'`).

### `get_config_ChWi_encoder(n_modules) -> dict` / `get_config_ChWi_decoder(n_modules) -> dict`
Each returns `{'modules': [...]}`, a list of `n_modules` per-module configs. **Bug**: both call `get_config_ChWi_encoder()` (no-arg) inside the loop instead of `get_config_ChWi_module_encoder()`/`get_config_ChWi_module_decoder()` — see **Notes**.

### `get_config_ChWi_module_encoder() -> dict`
`in_channels=1`, `out_channels=1`, `c_kernel=1`, `padding='same'`, `groups=1`, `use_batch_normalization=True`, `activation='elu'`, `p_kernel=1`.

### `get_config_ChWi_module_decoder() -> dict`
Same as module-encoder but `scale_factor=1` instead of `p_kernel` (upsampling factor instead of pooling kernel).

### `get_config_classifier_v1(use_only_mu_for_classification=True) -> dict`
`use_only_mu_for_classification` (if False, concatenates decoder's mu and logvar as classifier input), `neurons_list=[256, 4]`, `activation='elu'`, `prob_dropout=0.2`, `use_bias=False`.

### `get_config_classifier_model_v1(C, T, type_decoder, parameters_map_type, path_weights_hvEEGNet, use_only_mu_for_classification=True, freeze_encoder=True) -> dict`
Wraps a pretrained-hvEEGNet encoder + classifier head: `config_clf` (from `get_config_classifier_v1`), `config_hvEEGNet` (from `get_config_hierarchical_vEEGNet`), `path_weights` (path to pretrained hvEEGNet weights to load), `freeze_encoder` (if True, encoder weights are frozen during classifier training).

### `get_config_aEEGNet(C, T, n_external_query=0) -> dict`
Attention-augmented EEGNet. `eegnet_config` (from `get_config_EEGNet`), `use_channels_self_attention=True`, `channels_self_attention_config` (from `get_config_attention_module`), `channels_n_external_query=n_external_query`, `channels_external_attention_config_list=[]` (one config per external-attention module — see **Notes** for a bug), `use_depthwise_attention=False` (marked "STILL TO IMPLEMENT"). Prints a reminder that `qk_head_output_length`, `v_head_output_length`, `external_query_input_length` must be filled in manually after this call.

### `get_config_attention_module() -> dict`
`use_head_for_q/k/v=True`, `qk_head_output_length=-1` (must be set by caller), `v_head_output_length=-1` (must be set by caller), `normalize_qk=True`, `external_query_input_length=-1` (only for external-attention modules).

## Dependencies
- Internal: none (self-contained; other modules such as `config_training.py` and `library/model/*` import *from* this file).
- External: none.

## Notes
- **Likely dead/unreachable branch in `get_config_vEEGNet`**: the intended `elif type_encoder == 1:` for MBEEGNet got merged into the trailing comment of the previous line (`encoder_config['flatten_output'] = False # ... elif type_encoder == 1: #MBEEGnet`), so there is no real `elif` — only `if type_encoder == 0: ... else: raise ValueError(...)`. Consequently, inside the `if` block, `encoder_config` is first set to the EEGNet config and then *unconditionally overwritten* by `encoder_config = get_config_MBEEGNet(C, T)` (lines that look like they belong to the missing elif are actually still inside the `if type_encoder == 0` body due to indentation). Net effect: calling with `type_encoder=0` silently returns an MBEEGNet-based encoder config, not EEGNet; calling with `type_encoder=1` always raises `ValueError`. This contradicts the docstring/README (which describe `type_encoder=1` as "not implemented" but `type_encoder=0` as EEGNet). Treat `get_config_vEEGNet`'s encoder selection as broken/misleading and verify directly against `library/model/vEEGNet.py` before relying on it.
- **Bug in `get_config_ChWi_encoder`/`get_config_ChWi_decoder`**: both loops append `get_config_ChWi_encoder()` (recursing into themselves with no arguments, which itself recurses — actually `get_config_ChWi_encoder()` called with no args would raise a `TypeError` since `n_modules` is required) rather than `get_config_ChWi_module_encoder()` / `get_config_ChWi_module_decoder()`. As written, calling `get_config_ChWi_encoder(n_modules>0)` or `get_config_ChWi_decoder(n_modules>0)` will raise `TypeError: get_config_ChWi_encoder() missing 1 required positional argument`. This code path appears broken/untested.
- **Bug/typo in `get_config_aEEGNet`**: the function builds and returns a dict with key `channels_external_attention_config_list`, but the loop that's supposed to populate it appends to `config['external_attention_config_list']` (missing the `channels_` prefix), which is not a key that exists in `config` — this will raise a `KeyError` when `n_external_query > 0`.
- `print(...)` calls inside `get_config_aEEGNet` are the only side-effecting (non-dict-building) statements in this file; expect console output as a side effect of calling it.
- Several models documented in `library/README/README_model.md` (EEGNet, MBEEGNet, vEEGNet, hvEEGNet) match this file reasonably well, but ChWi, classifier, and aEEGNet configs are not covered by that README at all — this wiki page is the more complete reference for those.
