# `library/training/federated/simulation.py`

## Purpose
Ties `client.py` and the rest of the training stack together to run a federated-learning simulation without needing physically separate devices — using Flower's `VirtualClientEngine` (clients are spawned on demand by client-id string inside a single process/machine). Specifically wired for `hvEEGNet_shallow`.

## Key functions
### `generate_client_function_hvEEGNet_training(model_config_list, train_config_list, train_dataset_list, validation_dataset_list) -> client_function_hvEEGNet_training`
- A function factory: validates that `train_dataset_list` and `validation_dataset_list` have equal length (raises `ValueError` otherwise), then returns a closure `client_function_hvEEGNet_training(cid: str)` that Flower's `VirtualClientEngine` calls to materialize client `cid` on demand.
- Inside the returned closure: converts `cid` to `int`, indexes into all four input lists at that index to get this client's `model_config`/`train_config`/datasets, syncs `train_config['measure_metrics_during_training']`/`['use_classifier']` from `model_config['use_classifier']`, builds a fresh `hvEEGNet.hvEEGNet_shallow(model_config)`, looks up the right epoch functions via `train_generic.get_train_and_validation_function(model)` and loss function via `train_generic.get_loss_function(model_name='hvEEGNet_shallow', config=train_config)`, builds an `AdamW` optimizer (+ optional `ExponentialLR` scheduler per `train_config['use_scheduler']`), builds train/validation `DataLoader`s from this client's dataset slice, and finally constructs a `client.Client_V1(...)` and returns `tmp_client.to_client()` (Flower's required conversion from `NumPyClient` to its internal `Client` type).
- Prints progress markers (`"Get configs - OK"`, `"Model creation - OK"`, etc.) at each step — informal debug logging rather than structured logging.

## Dependencies
- Internal: `library.training.federated.client`, `library.model.hvEEGNet`, `library.training.train_generic`.
- External: `torch`, `flwr` (hard dependency, same `ImportError`-with-hint pattern as `client.py`/`server.py`).

## Notes
- Assumes **per-client** configuration: each of the four input lists must have one entry per simulated client, in matching order (`i`-th element of every list belongs to client `i`). This is the main contract a caller must respect when setting up a simulation (e.g. with `flwr.simulation.start_simulation(client_fn=generate_client_function_hvEEGNet_training(...), ...)`, not shown in this file but implied by the Flower API it's built for).
- Hardcoded to `hvEEGNet_shallow` (both the model class and the `model_name='hvEEGNet_shallow'` string passed to `train_generic.get_loss_function`) — to simulate federated training of a different model, this function would need to be duplicated/parameterized.
- No test/debug entry point (no `if __name__ == '__main__':` block) — this module is purely a library of client-factory functions, invoked from `training_scripts/federated/simulation_script.py`.
