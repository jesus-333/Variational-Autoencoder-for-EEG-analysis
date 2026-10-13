# `library/model/aEEGNet.py`

## Purpose
Experimental/prototype EEGNet + attention variant ("aEEGNet") that replaces EEGNet's depthwise spatial filter with a self-attention and/or external-query cross-attention mechanism computed across EEG channels. Appears to be unfinished/prototype code (see Notes).

## Key classes

### `aEEGNet_v1(nn.Module)`
- `__init__(config)`: builds a throwaway `EEGNet.EEGNet(config['eegnet_config'])` (forcing `flatten_output=True`) purely to reuse its `temporal_filter` submodule. Then either reuses the temp EEGNet's `spatial_filter` as-is, or replaces it with a `channels_attention(config)` module if `config['use_channels_self_attention']` or `config['channels_n_external_query'] > 0`.
- `forward(x)`: **returns `x` unchanged** — the forward pass is not implemented; `temporal_filter`/`spatial_filter` are built but never applied.

### `channels_attention(nn.Module)`
- `__init__(config)`: conditionally builds `self_attention_channels = self_attention_module(...)` (if `config['use_channels_self_attention']`) and a list of `external_attention_module` instances, one per `config['channels_n_external_query']`, each configured from `config['channels_external_attention_config_list'][i]`.
- `forward(x)`: **returns `x` unchanged** — not implemented; none of the constructed attention submodules are invoked.

### `self_attention_module(nn.Module)`
- Computes self-attention **across EEG channels** (not across time), independently per feature-map depth slice.
- `__init__(config)`: builds query/key/value head projections via `support_function.create_head(config)`; stores `normalize_qk`, `v_head_output_length`.
- `forward(x)`: expects `x` of shape `B x D x C x T` (D = depth/feature maps, C = EEG channels, T = time). For each depth slice `x[:, i]` (shape `B x C x T`), computes `q/k/v` via the heads and `support_function.compute_attention_values`, writing results into a `B x D x C x v_head_output_length` output tensor.
  - **Bug:** `if len(x.shape != 4):` — this is a boolean-vs-length-comparison bug; `x.shape != 4` compares a `torch.Size` tuple to an int (always `True` elementwise is not how `!=` works here — actually `x.shape != 4` evaluates the tuple-vs-int inequality as a single `bool`, always `True`, so `len(True)` raises `TypeError: object of type 'bool' has no len()`). This guard clause will always raise a `TypeError` if ever reached, rather than performing the intended shape check (`len(x.shape) != 4`). Treat this method as currently non-functional / untested.

### `external_attention_module(nn.Module)`
- Cross-attention between the EEG-channel feature map and an external query vector (e.g. a condition/context vector).
- `__init__(config)`: builds key/value heads via `support_function.create_head(config)` (discarding the q head it returns), plus its own `q_head` built from `config['external_query_input_length']`/`config['qk_output_length']` if `config['use_head_for_q']`.
- `forward(x, external_query)`: expects `x` shape `B x D x C x T` and `external_query` shape `B x 1 x N`. Per depth slice, computes attention between the (shared) external query and that slice's key/value, writing into a `B x D x 1 x v_head_output_length` output.
  - **Same `len(x.shape != 4)` / `len(external_query.shape != 3)` bug** as `self_attention_module.forward` — these guard clauses will raise `TypeError` rather than validating shapes.

## Dependencies
- Internal: `library.model.EEGNet`, `library.model.support_function` (`create_head`, `compute_attention_values`).
- External: `torch`, `torch.nn`.

## Notes
- This file is clearly a prototype/work-in-progress: `aEEGNet_v1.forward` and `channels_attention.forward` are stubs that just return their input unchanged, and both attention modules contain a shape-check bug (`len(x.shape != 4)` instead of `len(x.shape) != 4`) that would raise `TypeError` if the guard were ever triggered.
- Not referenced by any training script or config file found elsewhere in the repo at the time of writing — treat as dormant/exploratory code, not a supported model path. Confirm with `grep -r aEEGNet` before building on it.
