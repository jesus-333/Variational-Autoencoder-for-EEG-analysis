# `library/training/train_vEEGNet.py`

## Purpose
Per-model `train_epoch`/`validation_epoch` pair for the (non-hierarchical) `vEEGNet` model, dispatched to by `train_generic.get_train_and_validation_function`.

## Key functions
### `train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict=None) -> train_loss`
- If `train_config['use_classifier']`: unpacks `x_r, mu, log_var, predict_label = model(x)`; else unpacks `x_r, mu, log_var = model(x)` and sets `true_label = predict_label = None`.
- Loss via `loss_function.compute_loss(x, x_r, mu, log_var, predict_label, true_label)` — matches `vEEGNet_loss.compute_loss`'s signature (single-level, not hierarchical).
- Accumulates `train_loss`, `recon_loss`, `kl_loss`, and (if classifier) `clf_loss` from tuple index `[3]` (note: index 3, not 4 — because `vEEGNet_loss.compute_loss` returns a 4-tuple `(final_loss, recon_loss, kl_loss, clf_loss)` unlike the hierarchical 5-tuple). Logs to `log_dict` (`train_loss`, `train_loss_recon`, `train_kl_loss`, `train_loss_clf`) and pretty-prints if provided.

### `validation_epoch(model, loss_function, validation_loader, train_config, log_dict=None) -> validation_loss`
- Mirrors `train_epoch` under `model.eval()` + `torch.no_grad()`, with matching `log_dict` keys prefixed `validation_`.

## Dependencies
- External: `torch`, `pprint`.

## Notes
- No internal `library.*` imports are active (commented-out block at top shows previous imports of `wandb_support`, `metrics`, `dataset`, `vEEGNet`, config modules, `loss_function`, `train_generic` — all currently unused); the file only depends on the interfaces of its function arguments.
- Correctly uses tuple index `[3]` for `clf_loss` (vs. `train_hvEEGNet.py` and `train_ChWi.py`, which use index `[4]` because the hierarchical loss returns an extra `kl_loss_list` element) — a useful cross-check when debugging loss unpacking issues across the `train_*` files.
