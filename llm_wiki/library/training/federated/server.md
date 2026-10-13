# `library/training/federated/server.py`

## Purpose
Federated-learning server/strategy side (Flower framework): a `FedAvg` strategy subclass that aggregates client weight updates each round, saves the aggregated model to disk, and logs per-client and server-wide loss curves to Weights & Biases, including matplotlib plots of each client's training loss history.

## Key functions
- `get_weights(model)` / `get_random_weights_hvEEGNet(model_config)` — identical helpers to the ones in `federated/client.py` (duplicated rather than imported from a shared location).

## Key classes
### `FedAvg_with_wandb(flwr.server.strategy.FedAvg)`
- `__init__(server_config: dict)` — calls `check_config.check_server_config(server_config)`, builds a template `hvEEGNet.hvEEGNet_shallow(server_config['model_config'])` (used later purely as a vessel to `load_state_dict` into before saving), opens a `wandb.init(...)` run using `server_config['wandb_config']` (`project_name`, `notes`, `name_training_run`), creates a `wandb.Artifact('hvEEGNet_federated', ...)`, and tracks `count_rounds`/`tot_rounds` (`server_config['num_rounds']`).
- `aggregate_fit(server_round, results, failures)` — overrides Flower's base method: calls `super().aggregate_fit(...)` to get the FedAvg-aggregated parameters, then (if aggregation succeeded) converts them to numpy arrays, saves them to disk via `save_model_weights()`, uploads the checkpoint to the W&B artifact, and for each client's returned `results[i][1].metrics` dict: extracts per-epoch `train_loss_recon`/`train_loss_kl`/`train_loss_total` (flattened by `extract_metric_from_log_dict`, undoing the per-epoch key flattening done client-side in `Client_V1.fit`), accumulates them into server-wide averages, and creates per-client loss plots per `server_config['log_loss_type']` (1 = separate plots per metric, 2 = one combined plot, 3 = both). Logs averaged `server_recon_loss`/`server_kl_loss`/`server_total_loss` to W&B. On the final round (`count_rounds == tot_rounds`), calls `self.wandb_run.finish()`.
- `save_model_weights()` — converts the aggregated `List[np.ndarray]` (`self.model_weights`, set inside `aggregate_fit`) into a state dict keyed by `self.model.state_dict().keys()` (same ordering assumption as the client side), loads it into `self.model`, and saves to `{server_config['path_to_save_model']}/model_round_{count_rounds}.pth`.
- `extract_metric_from_log_dict(log_dict, metric_name) -> np.ndarray` — reconstructs a per-epoch array from Flower's flattened `{metric_name}_{epoch}` keys (inverse of the flattening done in `Client_V1.fit`), using `log_dict['epochs']` to know how many entries to expect.
- `create_and_log_plot(...)` / `create_metric_plot(...)` — build a matplotlib figure (one or more loss curves vs. epoch) per client per round and log it to W&B under key `{client_id}/{metric_name}_round_{count_rounds}` (with `commit=False`, meaning it's batched with the next `wandb.log` call rather than committed immediately).

## Dependencies
- Internal: `library.training.wandb_support`, `library.check_config`, `library.model.hvEEGNet`.
- External: `os` (imported, unused in visible code), `torch`, `numpy`, `matplotlib.pyplot`, `collections.OrderedDict`, `flwr`, `wandb` (both hard dependencies — both raise `ImportError` with install hints if missing).

## Notes
- `save_model_weights`'s docstring/comment says it "creates a folder specified in the path" but the actual code (`self.server_config['path_to_save_model']` as a bare expression) does **not** call `os.makedirs` — it's a no-op statement (likely intended to be `os.makedirs(self.server_config['path_to_save_model'], exist_ok=True)` but the call was dropped). If `path_to_save_model` doesn't already exist, `torch.save(...)` below it will raise `FileNotFoundError`.
- Multiple `# TODO` markers: "Remove in future. Used only to log something for the server at the end of each round" (around the per-round average-loss accumulation) and "add a central dataset and computation at the end of each round" — both flag this aggregation/logging logic as provisional/incomplete by the author's own admission.
- Duplicates `get_weights`/`get_random_weights_hvEEGNet` from `client.py` verbatim rather than importing from a shared module — if one copy is bugfixed, check the other.
- Hardcoded to `hvEEGNet_shallow` specifically (both for the template model and the W&B artifact name `'hvEEGNet_federated'`), unlike `client.py`'s somewhat more generic constructor.
