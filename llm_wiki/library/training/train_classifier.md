# `library/training/train_classifier.py`

## Purpose
Per-model `train_epoch`/`validation_epoch` pair for `classifier_model_v1` (`library/model/classifier.py`), the standalone classifier meant to be trained on top of a frozen/pretrained hvEEGNet latent space. Functionally near-identical to `train_EEGNet.py` but adds `log_dict` support and an (empty) `main()`.

## Key functions
### `train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict=None) -> train_loss`
- Standard supervised classification loop (forward → NLL loss → backward → step), weighted-averaged over the epoch. If `log_dict is not None`, stores `log_dict['train_loss']` and pretty-prints it.

### `validation_epoch(model, loss_function, validation_loader, train_config, log_dict=None) -> validation_loss`
- Same under `model.eval()` + `torch.no_grad()`; logs `log_dict['validation_loss']` if provided.

### `main()`
- Empty stub (`pass`); placeholder for a standalone entry point, unlike `train_EEGNet.py`'s `main_stft()`.

## Dependencies
- External: `torch`, `pprint`.

## Notes
- No internal `library.*` imports (the commented-out import block at the top shows this file used to import `wandb_support`, `metrics`, `dataset`, `vEEGNet`, config modules, `loss_function`, and `train_generic`, but none of that is active now) — it only relies on its two arguments' external interfaces (`model`, `loss_function`, `optimizer`, dataloaders), so it's decoupled from the rest of the training package at import time.
- `main()` is unimplemented; this file is intended to be used purely as a plugin into `train_generic`'s dispatch rather than run standalone.
