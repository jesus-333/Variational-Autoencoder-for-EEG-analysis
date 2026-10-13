# `library/model/EEGNet.py`

## Purpose
PyTorch implementation of EEGNet ([arXiv](https://arxiv.org/abs/1611.08024), [journal](https://iopscience.iop.org/article/10.1088/1741-2552/aace8c)), the feature-extraction backbone reused throughout the repo (as a classifier, as the vEEGNet/hvEEGNet encoder, and — via `hvEEGNet_V2.EEGNet_cell` — as a residual building block for the deep hierarchical model).

## Key classes

### `EEGNet(torch.nn.Module)`
- Reimplements EEGNet's convolutional feature extractor only; the original paper's final classification head is **not** included here (see `EEGNet_Classifier` for that).
- Constructor validates `config` via `check_config.check_model_config_EEGNet(config)`, then builds 3 blocks:
  - `temporal_filter`: `Conv2d(depth_first_layer, filter_1, kernel_size=c_kernel_1, padding='same', bias=use_bias)` + `BatchNorm2d`. `depth_first_layer` defaults to `1` but can be overridden via `config['depth_first_layer']` for time-frequency (depth > 1) inputs.
  - `spatial_filter`: depthwise `Conv2d(filter_1, filter_1*D, kernel_size=c_kernel_2, groups=filter_1)` + `BatchNorm2d` + activation + optional `AvgPool2d(p_kernel_1)` + dropout.
  - `separable_convolution`: depthwise `Conv2d(filter_1*D, filter_1*D, kernel_size=c_kernel_3, groups=filter_1*D, padding='same')` + pointwise `Conv2d(filter_1*D, filter_2, kernel_size=1)` + `BatchNorm2d` + activation + optional `AvgPool2d(p_kernel_2)` + dropout.
  - Config dict keys: `use_bias`, `D` (depth multiplier), `activation` (`'relu'|'elu'|'selu'|'gelu'`), `prob_dropout`, `use_dropout_2d`, `C` (EEG channels, stored but unused inside this class), `T` (time samples, stored but unused), `filter_1`, `filter_2`, `c_kernel_1`, `c_kernel_2`, `c_kernel_3`, `p_kernel_1`, `p_kernel_2` (either can be `None` to skip pooling), `flatten_output` (optional, default `False`), `print_var` (optional, default `False`), `input_size` (optional — if present together with `print_var=True`, triggers `debug_shape`), and optionally `depth_first_layer`.
- `forward(x)`: input `B x depth_first_layer x C x T` (typically `B x 1 x C x T`) → temporal_filter → spatial_filter → separable_convolution. If `flatten_output` is True, returns `x.flatten(1)` (shape `B x N`), else returns the raw conv tensor `B x filter_2 x 1 x T'`.
- `debug_shape(input_size)`: feeds a random tensor of `input_size` through each sub-layer individually and prints the shape after every step — purely diagnostic.

### `EEGNet_Classifier(torch.nn.Module)`
- Wraps `EEGNet` with a `Linear(input_neurons, n_classes) + LogSoftmax(dim=1)` head.
- `compute_number_of_neurons(input_size)`: runs a dummy forward pass to determine the flattened feature size needed for the classifier's input layer (so the caller doesn't need to hand-compute conv output shapes).
- `classify(x, return_as_index=True)`: convenience wrapper around `forward` that either returns raw log-probabilities or (`return_as_index=True`) the `argmax` class index per sample via `torch.exp` + `torch.argmax`.

## Dependencies
- Internal: `library.model.support_function` (activation/dropout getters, parameter counting), `library.check_config` (`check_model_config_EEGNet`).
- External: `torch`.

## Notes
- **The "depth ≠ EEG channels" gotcha** (also called out in `README_model.md`): the input tensor's dimension-1 axis is the convolutional "depth"/feature-map channel count (almost always `1`), not the number of EEG electrodes — EEG channel count `C` lives in dimension 2. Mixing these up is a common source of shape errors across the codebase.
- `self.C` and `self.T` are stored on the instance but never read elsewhere in this file — effectively dead attributes kept for potential introspection/debugging.
- `p_kernel_1`/`p_kernel_2` set to `None` disables pooling at that stage (replaced by `nn.Identity()`) — used by `hvEEGNet_V2.EEGNet_cell`, which needs the output time dimension to exactly match the input (`T' == T`) for its residual connection to work.
- `EEGNet` is reused as a residual building block in `hvEEGNet_V2.py`'s `EEGNet_cell`, as the encoder/decoder backbone in `vEEGNet.py`, and inside `MBEEGNet.py` (three parallel instances with different `c_kernel_1`).
