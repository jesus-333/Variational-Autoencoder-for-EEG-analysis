# `library/training/train_ChWi.py`

## Purpose
Per-model `train_epoch`/`validation_epoch` pair for the `ChWi_autoencoder` model (Channel-Wise network, `library/model/ChWi.py`), dispatched to by `train_generic.get_train_and_validation_function`.

## Key functions
### `train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict=None) -> train_loss`
- Standard VAE-style epoch: forward pass returns `(x_r, z, mu, log_var)`; unlike the other VAE models, ChWi's tensors are `B x D x T` (no explicit EEG-channel axis), so this function does `x.unsqueeze(2)` (and the same for `x_r`, `mu`, `log_var`) to add a dummy channel dimension before calling `loss_function.compute_loss`, because the shared loss code expects 4D `B x D x C x T` tensors.
- Accumulates `train_loss`, `recon_loss`, `kl_loss`, and (if `train_config['use_classifier']`) `clf_loss`, normalized by dataset size at the end. Logs to `log_dict` and prints via `pprint` if `log_dict is not None`.

### `validation_epoch(model, loss_function, validation_loader, train_config, log_dict=None) -> validation_loss`
- Same structure under `torch.no_grad()`. Note: calls `model.train()` (not `model.eval()`) at the top — likely a copy-paste bug inherited from the train function, meaning dropout/batchnorm stay in training mode during validation for this model.

## Dependencies
- External: `torch`, `pprint`.

## Notes
- **Bug/inconsistency**: `validation_epoch` sets `model.train()` instead of `model.eval()`, unlike `train_EEGNet.py`, `train_classifier.py`, `train_hvEEGNet.py`, and `train_vEEGNet.py`, which all correctly use `model.eval()` for validation. This means validation metrics for ChWi may be affected by training-mode-only layers (dropout, batch norm running stats not frozen).
- `clf_loss` is accumulated from `batch_train_loss[4]`/`batch_validation_loss[4]`, which implies `loss_function.compute_loss` must return a 5-tuple `(final_loss, recon_loss, kl_loss, kl_loss_list, clf_loss)` when classifier mode is on — i.e. this file assumes a `hvEEGNet_loss`-style loss object (with per-level `kl_loss_list`) even though it's training a (non-hierarchical) ChWi autoencoder; check that the loss object configured in `train_generic.get_loss_function('ChWi_autoencoder', ...)` (which returns `vEEGNet_loss`, not `hvEEGNet_loss`) actually returns index 4 as `clf_loss` — `vEEGNet_loss.compute_loss` returns `(final_loss, recon_loss, kl_loss, clf_loss)` (4-tuple, no `kl_loss_list`), so indexing `[4]` here would actually be out of range / wrong element when `use_classifier` is True. Verify before relying on classifier training for ChWi.
