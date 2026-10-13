# `library/model/support_function.py`

## Purpose
Grab-bag of small building blocks reused by almost every model in `library/model/`: activation/dropout factory functions, parameter counting, a generic "weighted sum" feature-fusion module, a LoRA linear layer, the modules that map encoder features to VAE distribution parameters (mean/logvar), the reparameterization trick, and the attention primitives used by `aEEGNet.py`. This is the lowest-level, most widely-imported file in the model subpackage.

## Key functions

### `get_activation(activation_name: str) -> nn.Module`
- Maps a string (`'relu'`, `'elu'`, `'selu'`, `'gelu'`, case-insensitive) to the corresponding `torch.nn` activation instance. Raises `ValueError` for anything else.

### `get_dropout(prob_dropout: float, use_droput_2d: bool) -> nn.Module`
- Returns `nn.Dropout2d(prob_dropout)` if `use_droput_2d` else `nn.Dropout(prob_dropout)`.

### `count_trainable_parameters(layer) -> int`
- Sums `p.numel()` over `layer.parameters()` where `p.requires_grad`. Used for the informational "Number of trainable parameters" print statements in `EEGNet`, `Decoder_EEGNet`, etc.

### `reparametrize(mu: torch.tensor, log_var: torch.tensor) -> torch.tensor`
- Standard VAE reparameterization trick: `z = mu + exp(0.5 * log_var) * eps`, `eps ~ N(0, I)` sampled with `torch.randn_like` and `.type_as(mu)` (so it follows the model to GPU).
- **Note:** contains leftover debug `print(...)` statements that check `torch.isnan(...).any()` on `mu`, `sigma`, `eps` on every call — noisy in production, should probably be removed or gated behind a verbose flag.

### `create_head(config: dict) -> (q_head, k_head, v_head)`
- Used by `aEEGNet.py`'s attention modules. Builds `nn.Linear` "head" projections for query/key/value only if the corresponding `config['use_head_for_q'/'k'/'v']` flag is set, else `nn.Identity()`. Expects `config['input_length']`, `config['qk_head_output_length']`, `config['v_head_output_length']`.

### `compute_attention_values(q, k, v, normalize_qk: bool) -> torch.Tensor`
- Plain scaled dot-product attention via `torch.bmm`: transposes `k`, computes `q @ k^T`, optionally divides by `sqrt(Dq)`, softmaxes over the last dim, then multiplies by `v`. Shapes: `q` is `B x C x Dq`, `k` is `B x Dk x C` (note: pre-transpose convention, unusual), `v` is `B x C x Dv`, output `B x C x Dv`.

## Key classes

### `weighted_sum_tensor(nn.Module)`
- `__init__(depth_x1, depth_x2, depth_output)`: a `1x1` `Conv2d` from `depth_x1 + depth_x2` to `depth_output` channels.
- `forward(x1, x2)`: concatenates `x1` and `x2` along the channel dim, then applies the `1x1` conv — effectively a learned weighted sum of two feature maps along the depth axis.
- Used in `hierarchical_VAE.py` to merge encoder and decoder feature maps (NVAE-style lateral connections) and to merge `z` samples with decoder activations.

### `LoRALinear(nn.Module)`
- `__init__(in_dim, out_dim, rank, alpha=-1, use_spectral_norm=False)`: two low-rank `nn.Linear` layers (`lora_a`: `in_dim -> rank`, `lora_b`: `rank -> out_dim`), no bias. If `alpha <= 0`, defaults `alpha = rank`. Optionally wraps both linears with `nn.utils.parametrizations.spectral_norm`.
- `forward(x)`: returns `(alpha / rank) * lora_b(lora_a(x))` — this is the LoRA *delta* only, not added to a frozen base weight; callers are expected to add it to a base linear's output themselves (see `map_to_distribution_parameters_with_vector_LoRA`, which actually uses it as a standalone replacement layer rather than an additive adapter — so despite the name, it is used here as a cheap low-rank linear layer, not as a LoRA adapter on top of a pretrained weight).

### `map_to_distribution_parameters_with_convolution(nn.Module)`
- `__init__(depth, use_activation=False, activation='elu')`: optional activation + `Conv2d(depth, depth*2, kernel_size=1)`.
- `forward(x)`: input `B x D x H x W` → doubles channel depth → `chunk(2, dim=1)` → returns `(mean, log_var)`, each `B x D x H x W`. Used when `parameters_map_type == 0` (matrix/convolutional latent space).

### `map_to_distribution_parameters_with_vector(nn.Module)`
- `__init__(hidden_space_dimension, input_shape, use_activation=False, activation='elu')`: two parallel `nn.Linear(flatten(input_shape[1:]), hidden_space_dimension)` branches (`map_mu`, `map_sigma`), each with optional leading activation.
- `forward(x)`: returns `(mu, log_var)` vectors of size `hidden_space_dimension`. Used when `parameters_map_type == 1` (vector latent space, no LoRA).

### `map_to_distribution_parameters_with_vector_LoRA(nn.Module)`
- Same role as above but the two linear branches are `LoRALinear` instances instead of full `nn.Linear`, to reduce parameter count. Used when `parameters_map_type == 2`.

### `sample_layer(nn.Module)`
- The central "bridge" module between a conv feature map and a VAE latent variable; used pervasively by `vEEGNet`, `hierarchical_VAE`, `hvEEGNet`, `ChWi.py`.
- `__init__(input_shape, config, hidden_space_dimension=-1)`:
  - `config['parameters_map_type']` selects the strategy:
    - `0`: convolutional/matrix latent space — wraps `map_to_distribution_parameters_with_convolution`. `depth` comes from `config['input_depth']` if present, else `input_shape[1]`.
    - `1`: vector latent space via plain feed-forward — wraps `map_to_distribution_parameters_with_vector`, plus a `map_from_hidden_space_sample = nn.Linear(hidden_space_dimension, n_neurons_output)` to project the sampled `z` back to the conv feature-map shape for the decoder.
    - `2`: vector latent space via LoRA feed-forward — same as `1` but with `map_to_distribution_parameters_with_vector_LoRA` and a `LoRALinear` for the back-projection. Needs `config['rank']` and `config['use_spectral_norm']`.
  - Also reads `config['use_activation_in_sampling']` and `config['sampling_activation']`.
  - Raises `ValueError` for any other `parameters_map_type`.
- `forward(x)`: for types `1`/`2`, flattens `x` first. Computes `(mean, log_var) = self.parameters_map(x)`, samples `z = reparametrize(mean, log_var)`. For type `0`, returns `z` directly (still shaped as a feature map). For types `1`/`2`, passes `z` through the feed-forward back-projection and reshapes to the original conv feature-map shape (`self.input_shape`, with batch dim set to `-1`). Returns `(x, mean, log_var)` where `x` is always in "feature-map" shape ready for the next conv decoder stage.
  - **Note:** contains a leftover debug `print(f"\t x : {torch.isnan(x).any()}, ...")` on every forward call.
- `reparametrize(mu, log_var, return_as_tensor=False)`: thin wrapper around the module-level `reparametrize` function; if `return_as_tensor=True` and `parameters_map_type` is `1`/`2`, additionally projects `z` through `self.ff_layer` and reshapes — used by `hierarchical_VAE.hVAE_decoder` to get the "corrected" `z` (NVAE-style delta mu/logvar) directly in feature-map shape.
  - **Note:** also littered with debug prints (`"z before/after ff_layer"`, and a verbose dump of `z`/weights/bias if any NaN is detected) — looks like leftover NaN-debugging instrumentation from training instability investigation, not something to rely on or keep when refactoring.

## Dependencies
- Internal: none (this is a leaf utility module within `library.model`); imported by `EEGNet.py`, `Decoder_EEGNet.py`, `vEEGNet.py`, `hierarchical_VAE.py`, `ChWi.py`, `aEEGNet.py`, `classifier.py`.
- External: `torch`, `torch.nn`.

## Notes
- `parameters_map_type` is the single most important config flag across the whole model subpackage: `0` = convolutional/matrix latent space, `1` = feed-forward/vector latent space, `2` = feed-forward/vector latent space with LoRA. It must be set consistently between an encoder's `sample_layer` config and wherever downstream code (e.g. `vEEGNet.forward`) branches on the same value.
- The many `torch.isnan(...)` debug prints throughout `reparametrize`/`sample_layer` are almost certainly leftover instrumentation from chasing a NaN-producing training bug (see `hierarchical_VAE.py` for similar instrumentation) — expect console spam during training/inference and consider stripping them in any cleanup pass.
- `LoRALinear` as implemented does **not** wrap a frozen pretrained weight (the classic LoRA pattern); it is used directly as a parameter-efficient replacement for `nn.Linear`.
