# `library/training/support_function.py`

## Purpose
Small grab-bag of helpers used around training: dataset-splitting index generation and a lookup table mapping a specific W&B sweep ID to the local path convention used for its saved weights.

## Key functions
- `get_idx_to_split_data(n_elements: int, percentage_split: float, seed=-1) -> (idx_1, idx_2)` — seeds `np.random` and returns a random permutation of `range(n_elements)` split into two index arrays at `n_elements * percentage_split`. Used for reproducible train/validation(/test) splits.
- `split_dataset(full_dataset, percentage_split: float)` — **marked `# NOT USED`** in a comment above it. Uses `torch.utils.data.random_split` to split a dataset object into two subsets by percentage.
- `get_sweep_path(sweep_id)` — hardcoded dict mapping one specific sweep id (`'jesus_333/ICT4AWE_Extension/wjim0nwt'`) to a local weight-path prefix (`'TMP_Folder/model_'`). Raises a `KeyError` for any other sweep id (no `.get` default). Used by `wandb_sweep.compute_metrics_from_sweep_EEGNet_classifier`.

## Dependencies
- External: `torch`, `numpy`.

## Notes
- `get_sweep_path` is a one-off, single-entry lookup table tied to a specific historical W&B run — not generalizable; treat as experiment-specific scaffolding rather than a reusable utility.
- `split_dataset` is explicitly dead/unused per its own comment; `get_idx_to_split_data` is the one actually used elsewhere (e.g. in dataset preprocessing) for splitting.
