# `library/model/classifier.py`

## Purpose
Simple feed-forward classifiers, including one that sits on top of a **pretrained, frozen hvEEGNet encoder** — i.e. the standard "train a VAE for reconstruction, then train a lightweight classifier on its latent representations" transfer-learning pattern used in this repo.

## Key classes

### `classifier_v1(nn.Module)`
- Plain MLP classifier: `Linear(input_size, neurons_list[0])` → repeated `[dropout, activation, Linear(neurons_list[i], neurons_list[i+1])]` blocks for each consecutive pair in `config['neurons_list']` → final `LogSoftmax(dim=1)`.
  - Config keys: `input_size`, `neurons_list` (list of layer widths, first element is the first hidden layer's output size), `activation`, `prob_dropout`, `use_bias` (optional, default `True`).
- `forward(x)`: `self.clf(x)`.

### `classifier_model_v1(nn.Module)`
- Composes a **frozen (optionally) hvEEGNet encoder** with a `classifier_v1` head, for transfer learning from a pretrained reconstruction model.
- `__init__(config)`:
  1. Builds `hvEEGNet.hvEEGNet_shallow(config['config_hvEEGNet'])` and loads its weights from `config['path_weights']` via `torch.load(..., map_location='cpu')`.
  2. Keeps only `self.encoder = tmp_hvEEGNet.h_vae.encoder` (discards the decoder/classifier of the loaded hvEEGNet).
  3. If `config['freeze_encoder']`, iterates `self.encoder.parameters()` setting `param.require_grad = False` — **note the typo**: PyTorch's actual attribute is `requires_grad`, not `require_grad`; this line silently does nothing (sets a new, unused attribute) and the encoder is **not actually frozen** despite the flag/intent. Flag this as a live bug if encoder-freezing is relied upon.
  4. Computes classifier input size via a dummy `(1,1,C,T)` forward through the encoder: either `len(mu.flatten())` (if `config_clf['use_only_mu_for_classification']`) or `len(mu.flatten()) + len(log_var.flatten())`, writes it into `config_clf['input_size']`.
  5. Builds `self.clf = classifier_v1(config_clf)`.
  - Config keys: `config_hvEEGNet` (full hvEEGNet config, see `hvEEGNet.md`), `config_clf` (`classifier_v1` config plus `use_only_mu_for_classification`), `path_weights` (path to a saved `state_dict`), `freeze_encoder`.
- `forward(x)`: encodes `x` via the (deepest-level) encoder output `(_, mu, log_var, _)`, flattens `mu` alone or `cat([mu, log_var])` depending on `use_only_mu_for_classification`, classifies → log-probabilities `y`.
- `classify(x, return_as_index=True)`: forwards and optionally converts log-probs to a class index via `argmax`.

## Dependencies
- Internal: `library.model.support_function`, `library.model.hvEEGNet`.
- External: `torch`, `torch.nn`.

## Notes
- **Known bug:** `freeze_encoder` does not actually freeze the encoder (`param.require_grad = False` is a no-op typo for `param.requires_grad = False`). Any training run that assumes the encoder is frozen is actually fine-tuning it end-to-end — worth confirming with the user/experiment logs whether this was intentional or a genuine bug before "fixing" it, since it may silently change results of past experiments that relied on (or accidentally benefited from) this behavior.
- This file only wraps `hvEEGNet` (V1), not `hvEEGNet_V2` — if V2 becomes the primary model, a V2-compatible version of `classifier_model_v1` would need to be written (the encoder-extraction path `tmp_hvEEGNet.h_vae.encoder` would need adjusting since V2's encoder cell structure differs).
