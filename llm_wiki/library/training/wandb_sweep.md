# `library/training/wandb_sweep.py`

## Purpose
Driver script for running a W&B hyperparameter sweep over the EEGNet STFT classifier, plus a post-hoc function to download the best sweep runs' weights and recompute test metrics for them. Meant to be run as a standalone script (`python wandb_sweep.py`), not imported as a library module from elsewhere in the package (its imports are flat, non-relative).

## Key functions
### `train_sweep(config=None)`
- One training run inside a W&B sweep agent. Builds `dataset_config`/`train_config`/`model_config` from the config helpers (hardcoded `C=22`, `T=512`, classifier config, EEGNet STFT classifier), opens `wandb.init(...)`, pulls the current sweep's suggested hyperparameters via `wandb.config`, overlays them onto the three config dicts with `update_config_with_sweep_parameters`, derives `model_config['filter_2'] = filter_1 * D`, handles the STFT `window` parameter specially (sweep configs serialize tuples as lists, so it's converted back to a tuple), logs the merged configs back to `wandb.config`, creates a model artifact, trains via `train_generic.train_and_test_model('EEGNet', ...)`, and logs the artifact.

### `update_config_with_sweep_parameters(sweep_config, other_config)`
- In-place: for every key already present in `other_config`, overwrite it with the sweep-suggested value from `sweep_config` if that key exists there too. Used to patch sweep-selected hyperparameters into the dataset/train/model config dicts (and the nested STFT params dict).

### `compute_metrics_from_sweep_EEGNet_classifier(sweep_id: str, download_path=None) -> metrics_info`
- Queries the W&B API for all runs of `sweep_id`, filters to completed runs with `accuracy_validation >= 0.6` (via `filter_list_of_runs`), then for each surviving run: rebuilds the dataset/model from the run's logged config, downloads both the end-of-training and best-validation-loss weight files (path convention from `support_function.get_sweep_path(sweep_id)`), reloads each into a freshly constructed model, and recomputes test-set metrics for both checkpoints via `metrics.compute_metrics`. Returns a dict keyed by subject id, each value a list of `{'END': acc, 'BEST': acc}` dicts (one per matching run for that subject).

### `filter_list_of_runs(list_of_runs, metric, min_value)`
- Keeps only runs with `state == 'finished'` and `run.summary[metric] >= min_value`.

### `main_sweep()`
- Entry point: builds a sweep config via `ct.get_config_sweep('accuracy_validation', 'maximize')`, then calls `wandb.agent(sweep_id=..., function=train_sweep)` against a **hardcoded, already-created** sweep id (`'jesus_333/ICT4AWE_Extension/wjim0nwt'`); the code to actually create a new sweep (`wandb.sweep(...)`) is present but commented out.

## Dependencies
- Internal (flat imports, **not** relative — `import config_dataset as cd`, `import train_generic`, `import preprocess as pp`, `import support_function`, `import metrics` — this module assumes it is run with the `library/training/` directory itself on `sys.path`, unlike every other file in the package which uses relative imports like `from . import ...` / `from .. import ...`).
- External: `wandb`, `torch`, `pprint`, `os`, `shutil` (`shutil` is imported but not used in the visible code).

## Notes
- **Import style outlier**: this is the only file in `library/training/` using flat (non-package-relative) imports (`import config_dataset as cd` instead of `from ..config import config_dataset as cd`). Running it via `python -m library.training.wandb_sweep` or importing it as `library.training.wandb_sweep` from elsewhere will raise `ModuleNotFoundError` unless the caller's `sys.path` happens to include `library/training/` and `library/` directly. Treat this as a standalone script meant to be run from inside its own directory, not a properly integrated package module.
- Tightly coupled to a specific historical sweep (`wjim0nwt`) and project (`ICT4AWE_Extension`) and to `EEGNet` STFT classification specifically — not a generic sweep utility.
- `shutil` import appears unused.
