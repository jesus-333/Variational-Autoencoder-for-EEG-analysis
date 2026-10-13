# `library/support_function.py`

## Purpose
Small grab-bag of generic helper functions used across the repo: splitting data into train/test index arrays, and a lookup table mapping a wandb sweep ID to the local path where that sweep's model weights were saved.

## Key classes
None.

## Key functions

### `get_idx_to_split_data(n_elements: int, percentage_split: float, seed: int = None) -> (idx_1: np.ndarray, idx_2: np.ndarray)`
- Validates `n_elements > 1` and `0 < percentage_split < 1`, raising `ValueError` otherwise.
- Optionally seeds `np.random.seed(seed)` for reproducibility.
- Returns a random permutation of `range(n_elements)` split into two index arrays: the first of size `int(n_elements * percentage_split)`, the second with the remainder.
- Used to build train/validation (or train/test) splits by index rather than by directly splitting data arrays.

### `split_dataset(full_dataset, percentage_split: float) -> (dataset_1, dataset_2)`
- Marked `# NOT USED` in a comment directly above it.
- Thin wrapper around `torch.utils.data.random_split` splitting a PyTorch `Dataset` into two subsets by the given percentage.

### `get_sweep_path(sweep_id) -> str`
- Looks up a hardcoded dict mapping a single specific wandb sweep ID (`'jesus_333/ICT4AWE_Extension/wjim0nwt'`) to a local save path (`'TMP_Folder/model_'`).
- Will raise `KeyError` for any other `sweep_id` — this is a one-off/throwaway lookup table from a specific past experiment, not a general-purpose utility, and the dict would need extending (or the function replacing) for new sweeps.

## Dependencies
- Internal: none.
- External: `torch`, `numpy`.

## Notes
- `get_sweep_path`'s hardcoded single-entry dict is a strong signal of research-script residue — don't rely on it for any sweep other than the one baked in; extend the dict or write a new helper for other sweeps/experiments.
- `split_dataset` is dead code per its own comment; prefer `get_idx_to_split_data` + manual `Subset`/indexing, which is the pattern used elsewhere in the dataset code.
