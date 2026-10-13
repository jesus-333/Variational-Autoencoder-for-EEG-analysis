# `library/training/train_hvEEGNet.py`

## Purpose
Per-model `train_epoch`/`validation_epoch` pair for both `hvEEGNet_shallow` and `hvEEGNet_V2` (both dispatch here from `train_generic.get_train_and_validation_function`). Handles the hierarchical VAE's multi-level outputs (`mu_list`, `log_var_list`, `delta_mu_list`, `delta_log_var_list`) and the optional attached classifier head.

## Key functions
### `train_epoch(model, loss_function, optimizer, train_loader, train_config, log_dict=None) -> train_loss`
- If `train_config['use_classifier']` is true: forward pass unpacks `x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predict_label = model(x)`; otherwise unpacks the 5-tuple without `predict_label` and sets `true_label = predict_label = None`.
- Loss via `loss_function.compute_loss(x, x_r, mu_list, log_var_list, delta_mu_list, delta_log_var_list, predict_label, true_label)` — matches `hvEEGNet_loss.compute_loss`'s signature.
- Accumulates `train_loss`, `recon_loss`, `kl_loss` always, and `clf_loss` (from tuple index `[4]`) only if `use_classifier`, each weighted by batch size and normalized by dataset length at the end.
- Populates `log_dict` with `train_loss`, `train_loss_recon`, `train_kl_loss` (and `train_loss_clf` if classifier is on) and pretty-prints it, if `log_dict` is provided.

### `validation_epoch(model, loss_function, validation_loader, train_config, log_dict=None) -> validation_loss`
- Mirrors `train_epoch` under `model.eval()` + `torch.no_grad()`. Populates `log_dict` with `validation_loss`, `validation_loss_recon`, `validation_kl_loss` (+ `validation_loss_clf`).

## Dependencies
- External: `torch`, `pprint`.

## Notes
- Both `hvEEGNet_shallow` and `hvEEGNet_V2` share this exact epoch logic, so their `forward()` signatures must stay compatible (same tuple shape/order) for this file to keep working for both — check `library/model/hvEEGNet.py` and `library/model/hvEEGNet_V2.py` together if modifying either model's `forward`.
- There's a commented-out per-level KL logging block (`# for i in range(len(validation_loss[3])): ...`) in `validation_epoch` — dead code showing an abandoned attempt to log each hierarchy level's KL term individually (the live code only logs the summed `kl_loss`, not the `kl_loss_list` returned by `hvEEGNet_loss.compute_loss`).
