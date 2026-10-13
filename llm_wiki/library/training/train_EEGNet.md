# `library/training/train_EEGNet.py`

## Purpose
Per-model `train_epoch`/`validation_epoch` pair for the plain classifier models (`EEGNet_Classifier`, and reused for `MBEEGNet_Classifier` since both just produce label predictions). Also contains a `main_stft()` standalone entry point for training an STFT-input EEGNet classifier directly from this file.

## Key functions
### `train_epoch(model, loss_function, optimizer, train_loader, train_config) -> train_loss`
- Classic supervised classification loop: forward pass → `predict_label = model(x)`, `loss_function(predict_label, true_label)` (an `NLLLoss`, per `train_generic.get_loss_function`), backward, optimizer step. Returns the batch-size-weighted average loss over the epoch. No `log_dict` parameter here (unlike most other `train_*.py` files).

### `validation_epoch(model, loss_function, validation_loader, train_config) -> validation_loss`
- Same forward/loss computation under `model.eval()` + `torch.no_grad()`, no backward pass.

### `main_stft()`
- Standalone script entry (`if __name__ == "__main__"`): builds a MOABB dataset config with STFT parameters (`cd.get_moabb_dataset_config`, `cd.get_config_stft`), a classifier train config (`ct.get_config_classifier`), and an EEGNet STFT classifier model config (`cm.get_config_EEGNet_stft_classifier`), then calls `train_generic.train_and_test_model('EEGNet', ...)`.

## Dependencies
- Internal: `library.config.{config_model, config_dataset, config_training}` (aliased `cm`, `cd`, `ct`), `library.training.train_generic`.
- External: `torch`.

## Notes
- Signature mismatch vs. sibling files: this is the only `train_epoch`/`validation_epoch` pair without an optional `log_dict` parameter, so it cannot report per-epoch metrics back into the shared `train_generic.train` loop's `log_dict` the way `train_vEEGNet`/`train_hvEEGNet`/`train_ChWi` do (though `train_generic.train` still handles its classification metrics separately via `library.metrics`, so this isn't broken, just inconsistent).
- `main_stft()` is meant to be run as a script directly (`python train_EEGNet.py`), separate from the library's `train_and_test_model` call sites elsewhere.
