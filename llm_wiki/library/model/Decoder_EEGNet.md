# `library/model/Decoder_EEGNet.py`

## Purpose
Mirror-image decoders for `EEGNet`, used by `vEEGNet` to reconstruct an EEG signal from a latent sample `z`. Provides two architectural variants for undoing the pooling/downsampling done by `EEGNet`'s encoder: upsample-based and transposed-convolution-based.

## Key classes

### `EEGNet_Decoder_Upsample(nn.Module)`
- Inverse of `EEGNet` that uses `nn.Upsample` to undo `AvgPool2d`, and ordinary (non-transposed) `Conv2d`/depthwise `Conv2d` with `padding='same'` for the conv steps (relying on upsampling, not transposed conv, to restore spatial size).
- `__init__(config)`: if `config['parameters_map_type'] == 1` (vector latent space), builds a feed-forward `ff_decoder = Linear(hidden_space, prod(dimension_reshape)) + activation` to project `z` back up to a conv feature-map shape (`config['dimension_reshape']`) before the conv stack; for `parameters_map_type == 0` (matrix latent space), `z` is already feature-map shaped and is used directly.
  - Builds 3 blocks mirroring `EEGNet` in reverse: `separable_convolution_transpose` (dropout → upsample by `p_kernel_2` → activation → `BatchNorm2d` → pointwise conv `filter_1*D → filter_2`... — **note the conv dims here look suspicious, see Notes** → depthwise conv `c_kernel_3`), `spatial_convolution_transpose` (dropout → upsample by `p_kernel_1` → activation → `BatchNorm2d` → `ConvTranspose2d` depthwise `c_kernel_2`), `temporal_convolution_transpose` (`BatchNorm2d` → `Conv2d(filter_1, 1, c_kernel_1, padding='same')` — projects back down to depth 1, i.e. the reconstructed signal).
  - Config keys: `use_bias`, `D`, `activation`, `prob_dropout`, `use_dropout_2d`, `hidden_space`, `parameters_map_type`, `dimension_reshape` (target conv shape, e.g. `[-1, filter_1*D, 1, T']`), `filter_1`, `filter_2`, `c_kernel_1/2/3`, `p_kernel_1/2`, `print_var`.
- `forward(z)`: optional FF projection + reshape → `separable_convolution_transpose` → `spatial_convolution_transpose` → `temporal_convolution_transpose` → reconstructed `x_r` of shape `B x 1 x C x T`.
- `debug_shape()`: diagnostic — feeds a random tensor through each sub-layer and prints shapes.

### `EEGNet_Decoder_Transpose(nn.Module)`
- Alternative decoder that always starts from a vector `z` (no `parameters_map_type` branch — always does the FF projection + reshape) and uses `ConvTranspose2d` with `stride=(1,2)` in the separable and temporal stages instead of upsampling, to grow the time dimension.
- Same 3-block structure (`separable_convolution_transpose`, `spatial_convolution_transpose`, `temporal_convolution_transpose`) as the Upsample variant but with transposed convolutions (strided) replacing the commented-out `nn.Upsample` calls.
- Config keys: same as above minus `parameters_map_type` (always acts as if vector/FF mode).
- `forward(z)`: FF projection → reshape → 3 transpose-conv blocks → `x_r`.

## Dependencies
- Internal: `library.model.support_function` (activation/dropout, parameter counting).
- External: `torch`, `torch.nn`, `numpy` (for `np.prod`/`np.abs` on `dimension_reshape`).

## Notes
- Selected in `vEEGNet.py` via `config['type_decoder']`: `0` → `EEGNet_Decoder_Upsample`, `1` → `EEGNet_Decoder_Transpose`.
- In `EEGNet_Decoder_Upsample.separable_convolution_transpose`, the pointwise `Conv2d(filter_1*D, filter_2, ...)` is listed **before** the depthwise `Conv2d(filter_1*D, filter_1*D, c_kernel_3, ...)` in the `nn.Sequential`, i.e. channels go `filter_1*D → filter_2` and then the next conv still expects `filter_1*D` input channels — this only works because `BatchNorm2d(filter_2)` is applied *before* the pointwise conv that changes the channel count to `filter_2`, meaning the batchnorm's channel count (`filter_2`) does not match the tensor it's actually applied to at that point in the `Sequential` (which still has `filter_1*D` channels) unless `filter_1*D == filter_2`. Flag this as a likely bug/fragile assumption if `filter_1*D != filter_2` is ever used — read `EEGNet_Decoder_Upsample.__init__` directly before changing filter-size configs.
- `EEGNet_Decoder_Transpose` hardcodes `stride=(1, 2)` for the time-expanding transpose convs, meaning it assumes pooling factors of 2 along time were used in the corresponding `EEGNet` encoder (not parameterized via `p_kernel_1/2` the way the Upsample variant is).
- Both classes' `debug_shape()` methods index directly into the internal `nn.Sequential` by integer position (e.g. `self.separable_convolution_transpose[1]`) — fragile if the block composition changes.
