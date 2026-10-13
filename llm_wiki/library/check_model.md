# `library/check_model.py`

## Purpose
Small interactive/manual "smoke test" script for model creation and forward passes. Not a pytest test file (no `assert`s) — meant to be run line-by-line in a Python/IPython console (see the `%load_ext autoreload` docstring hint) to catch obvious breakage when editing `vEEGNet` or `hvEEGNet`.

## Key classes
None.

## Key functions

### `check_vEEGNet() -> None`
- Builds a `vEEGNet` model config via `config_model.get_config_vEEGNet(C, T, hidden_space, type_encoder, type_decoder)` with hardcoded `C=22`, `T=512`, `hidden_space=16`, `type_encoder=0`, `type_decoder=0`.
- Instantiates `vEEGNet.vEEGNet(model_config)`, runs a forward pass on random input `torch.rand(5, 1, C, T)`, and prints input/output shapes plus `z_mean`, `z_log_var`, `predict_labels`.
- No return value — purely for visual inspection of printed output.

### `check_hVAE_shallow() -> None`
- Builds a shallow hierarchical vEEGNet config via `config_model.get_config_hierarchical_vEEGNet(C, T, type_decoder, parameters_map_type)` with `C=22`, `T=1250`, `type_decoder=0`, `parameters_map_type=1`.
- Instantiates `hvEEGNet.hvEEGNet_shallow(model_config)`, times model creation, runs a CPU forward pass (timed), and — if `torch.cuda.is_available()` — moves model+input to `'cuda'` and times a GPU forward pass too.
- Prints timing info; no return value.

## Dependencies
- Internal: `library.config.config_model` (aliased `cm`), `library.model.vEEGNet`, `library.model.hvEEGNet`.
- External: `torch`, `time`.

## Notes
- Imports use relative package syntax (`from .config import config_model as cm`, `from .model import vEEGNet, hvEEGNet`), so this file must be run as part of the `library` package (e.g. `import library.check_model as cm` as its own docstring suggests), not executed standalone.
- `hvEEGNet_shallow` here refers to the shallow 2-level model in `library/model/hvEEGNet.py` — compare with the deeper model in `library/model/hvEEGNet_V2.py`, which this file does not test.
- Useful as a quick regression check after modifying `vEEGNet`/`hvEEGNet`/`hierarchical_VAE` code, but it is not wired into any CI/test runner.
