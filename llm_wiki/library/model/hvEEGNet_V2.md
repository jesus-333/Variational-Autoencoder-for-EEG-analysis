# `library/model/hvEEGNet_V2.py`

## Purpose
Second-generation hvEEGNet, developed per the recent "Work on a new hvEEGNet" commit on this branch. Unlike V1 (`hvEEGNet.py`), which derives its 3 hierarchy levels from a torn-apart `vEEGNet`, V2 builds the hierarchy from `n_cells` identical, shape-preserving residual `EEGNet_cell` blocks, making the hierarchy depth a free hyperparameter rather than fixed at 3.

## Key classes

### `hvEEGNet_V2(torch.nn.Module)`
- `__init__(config)`: reads `n_cells = config['n_cells']`, builds `encoder_cell_list` and `decoder_cell_list` as **two separate lists of `n_cells` independently-initialized `EEGNet_cell(config['encoder_config'])` instances** (not shared/tied weights between encoder and decoder, and not shared between cells), then wraps them in `hierarchical_VAE.hVAE(encoder_cell_list, decoder_cell_list, config)` as `self.h_vae`.
  - `use_classifier` is accepted in config but **raises `NotImplementedError`** if `True` ("Please set `use_classifier` to False").
  - Config keys: `n_cells`, `encoder_config` (passed as-is to every `EEGNet_cell`, including `print_var`/`input_shape` debug options — see below), plus whatever `hierarchical_VAE.hVAE` needs, plus `use_classifier` (must be `False`).
- `forward(x)`: `h_vae(x) → (x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list)`; classifier branch raises `NotImplementedError` (dead code path, since construction already forbids `use_classifier=True`).

### `EEGNet_cell(torch.nn.Module)`
- A single residual hierarchy level: wraps one `EEGNet` whose output is reshaped and added back to its own input, so the cell is shape-preserving (`B x 1 x C x T → B x 1 x C x T`) and can be stacked arbitrarily deep.
- `__init__(config)`: builds `self.eegnet = EEGNet.EEGNet(config)` and forces `self.eegnet.flatten_output = False`. If `config['input_shape']` is provided, runs a debug forward pass and **prints a warning** (does not raise) if the output shape doesn't match the input shape.
  - Critical configuration constraint (documented in the class docstring): to keep the residual connection shape-valid, `p_kernel_1` and `p_kernel_2` should be set to `-1`/`None` (no pooling, so `T' == T`), and `filter_2` should be set equal to `C` (the number of EEG/input channels) so the EEGNet output depth matches the cell's channel dimension after the permute below.
- `forward(x)`: `output = eegnet(x)` → `B x filter_2 x 1 x T` → `permute(0, 2, 1, 3)` → `B x 1 x filter_2 x T` → add to `x` (residual). Requires `filter_2 == C` and `T' == T` for the addition to broadcast correctly (see constraint above); otherwise this will raise a shape-mismatch error or silently broadcast incorrectly.

## Dependencies
- Internal: `library.model.EEGNet`, `library.model.hierarchical_VAE`.
- External: `torch`.

## Notes
- `use_classifier` must currently be `False` — classifier support for V2 is explicitly unimplemented (two separate `NotImplementedError` raises, in `__init__` and `forward`). Treat this as a known gap, not a bug to silently work around.
- Because `encoder_cell_list` and `decoder_cell_list` are built as two independent loops of `EEGNet_cell(config['encoder_config'])`, **every cell — encoder and decoder alike — shares the exact same hyperparameters** (there's a single `config['encoder_config']`, no separate `decoder_config`); only the learned weights differ between cells, not the architecture.
- This is the model referenced by `training_scripts/d2a_hvEEGNet_V2.py` and the modified `library/model/hvEEGNet_V2.py`/`library/training/train_generic.py`/`library/training/wandb_training.py` seen in the current `jesus-experiment` branch diff — i.e. this is actively in-progress work, not a finished/stable API. Double check `library/check_config.py` and `library/config/config_model.py` for the latest expected config shape before relying on documented defaults here.
- The residual-cell design means `hvEEGNet_V2`'s hierarchy depth (`n_cells`) is now a free hyperparameter, unlike V1's fixed depth-3 hierarchy derived from EEGNet's 3 fixed conv blocks.
