# `library/training/federated/client.py`

## Purpose
Federated-learning client side, built on the [Flower](https://flower.ai/) (`flwr`) framework. Wraps a local PyTorch model + dataloaders + train/validation epoch functions into a `flwr.client.NumPyClient` that the Flower simulation/server can drive. Specifically wired for `hvEEGNet_shallow` but generic enough to take any model/epoch-function pair as constructor arguments.

## Key functions
- `get_weights(model: torch.nn.Module) -> list` — extracts `model.state_dict()` values as a list of numpy arrays (Flower's wire format for parameters).
- `get_random_weights_hvEEGNet(model_config: dict) -> list` — convenience: builds an untrained `hvEEGNet.hvEEGNet_shallow(model_config)` and returns its weights via `get_weights`. Useful for initializing the Flower server with correctly-shaped random weights before any client has trained.

## Key classes
### `Client_V1(flwr.client.NumPyClient)`
- `__init__(model, train_loader, validation_loader, train_epoch_function, validation_epoch_function, loss_function, optimizer, lr_scheduler, train_config: dict)` — stores everything needed to run local epochs; moves `model` to `train_config['device']`; reads `train_config['client_id']` to tag its own results.
- `get_parameters(config) -> weights_list` — same as module-level `get_weights` but bound to `self.model`.
- `get_properties(config) -> {}` — required by the Flower API; returns nothing.
- `set_weights(parameters_list: list)` — rebuilds an `OrderedDict` state dict from `(key, numpy_array)` pairs zipped against `self.model.state_dict().keys()` (relies on dict key order matching the order Flower sends parameters in, which in turn relies on `get_weights`/`get_parameters` having produced them in that same order) and loads it with `strict=True`.
- `fit(parameters: list, config: dict) -> (new_parameters, num_train_examples, metrics_dict)` — the core local-training round: `set_weights(parameters)`, then runs `train_config['epochs']` local epochs using the injected `train_epoch_function`, stepping `lr_scheduler` if present. Because Flower's metrics dict can't hold lists, per-epoch `train_loss_kl`/`train_loss_recon`/`train_loss_total` are flattened into individually-keyed entries like `train_loss_kl_{epoch}` rather than stored as arrays. Returns updated weights, the number of training examples (for FedAvg weighting), and the flattened metrics dict (also includes `epochs` and `client_id`).
- `evaluate(parameters: list, config: dict) -> (loss, num_validation_examples, {})` — `set_weights(parameters)`, runs one validation epoch via the injected `validation_epoch_function`, returns the validation loss (cast to `float`) and dataset size; the returned metrics dict is always empty (client id is only captured into a local `log_dict` that is never returned/used).

## Dependencies
- Internal: `library.model.hvEEGNet`.
- External: `torch`, `collections.OrderedDict`, `flwr` (hard dependency — importing this module raises `ImportError` with an install hint if `flwr` isn't installed).

## Notes
- `set_weights` assumes parameter order from the server matches `self.model.state_dict().keys()` order exactly — if the model architecture differs between client/server instantiation (e.g. different `model_config`), this will silently misassign weights to the wrong layers before `strict=True` catches a shape mismatch (or worse, matches shapes coincidentally).
- `evaluate`'s `log_dict['client_id']` assignment at the end is effectively dead: the dict is built but never included in the returned tuple (`{}` is returned instead), so per-client identification is lost for evaluation rounds (unlike `fit`, which does return its metrics dict).
- Despite the generic constructor, the only consumer (`federated/simulation.py`) always builds `Client_V1` specifically for `hvEEGNet_shallow`.
