# `library/dataset/support_function.py`

## Purpose
Small collection of standalone helper functions used across the dataset pipeline: splitting data into index subsets, and looking up where a given wandb sweep's model weights were saved.

## Key functions

### `get_idx_to_split_data(n_elements: int, percentage_split: float, seed=-1)`
- Returns two index arrays `(idx_1, idx_2)` that partition `range(n_elements)` into a random permutation split at `int(n_elements * percentage_split)`.
- Seeds `np.random.seed(seed)` before permuting, for reproducibility — called throughout `preprocess.py` for train/test and train/validation splits.

### `split_dataset(full_dataset, percentage_split: float)` — marked `# NOT USED`
- Splits a PyTorch `Dataset` object directly via `torch.utils.data.random_split` into two sub-datasets sized by `percentage_split`. Explicitly flagged as unused dead code in a comment.

### `get_sweep_path(sweep_id)`
- Looks up a hardcoded dictionary mapping one specific wandb sweep ID (`'jesus_333/ICT4AWE_Extension/wjim0nwt'`) to a local path prefix (`'TMP_Folder/model_'`) where that sweep's model weight files were saved.
- Will raise `KeyError` for any other `sweep_id` — this is a personal lookup table for one past experiment, not a general-purpose function.

## Dependencies
- Internal: none.
- External: `torch` (only used by the unused `split_dataset`), `numpy`.

## Notes
- `get_idx_to_split_data` is the one function here actually used by the rest of the pipeline (`preprocess.py`); the other two are effectively dead/personal-use code that a future agent can likely ignore or remove.
- `get_sweep_path`'s hardcoded single-entry dictionary is a strong signal that this function was written for one specific experiment/paper run and never generalized — don't extend it without understanding it's tied to a specific wandb project/sweep from past research.
