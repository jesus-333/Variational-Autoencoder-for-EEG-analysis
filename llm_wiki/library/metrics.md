# `library/metrics.py`

## Purpose
Classification-evaluation utilities: computing accuracy/Cohen's kappa/sensitivity/specificity/F1/confusion-matrix for a trained classifier model, either from a single DataLoader or by sweeping over a folder of saved `.pth` checkpoint files. Used by classifier training/evaluation scripts (e.g. `library/training/train_classifier.py`, `library/model/classifier.py`-based models) rather than by the (h)vEEGNet reconstruction pipeline.

## Key classes
None.

## Key functions

### `compute_metrics(model, loader, device) -> dict`
- Calls `compute_label` then `compute_metrics_from_labels`. Returns the metrics dict described below.

### `compute_label(model, loader, device) -> (true_label: Tensor, predict_label: Tensor)`
- Sets `model.eval()`, moves it to `device`.
- Iterates the DataLoader, calling `model.classify(x)` on each batch (so the model passed in **must** implement a `.classify()` method — e.g. `library/model/classifier.py`, `library/model/EEGNet.py` classifier variants, `MBEEGNet_Classifier`).
- Concatenates all batches' true/predicted labels and moves them to CPU.

### `compute_metrics_from_labels(true_label, predict_label) -> dict`
- Returns a dict with keys: `accuracy`, `cohen_kappa`, `sensitivity` (weighted recall), `specificity` (custom multiclass average, see below), `f1` (weighted), `confusion_matrix` (custom 4x4 normalized matrix, see below).
- Uses `sklearn.metrics`: `accuracy_score`, `cohen_kappa_score`, `recall_score`, `f1_score`.

### `compute_specificity_multiclass(true_label, predict_label, weight_sum=True) -> float`
- For each unique label value, binarizes true/predicted labels (one-vs-rest) and computes binary specificity via `compute_specificity_binary`.
- Averages across classes, weighted by class support (`weight_sum=True`) or unweighted (`weight_sum=False`).

### `compute_specificity_binary(true_label, predict_label) -> float`
- Builds a 2x2 `sklearn.metrics.confusion_matrix` and computes `TN / (TN + FP)` using **`cm[1, 1]` as TN and `cm[1, 0]` as FP** — this indexing assumes a specific row/column label ordering from `confusion_matrix`; it is non-standard (the conventional binary confusion matrix layout has `TN` at `[0,0]` and `FP` at `[0,1]`), so this should be treated with caution/verified against actual label encoding before trusting the output.

### `compute_multiclass_confusion_matrix(true_label, predict_label) -> np.ndarray (4x4)`
- Manually builds a confusion matrix by iterating sample-by-sample and incrementing `confusion_matrix[true, predict]`.
- **Hardcoded to a 4-class problem** (`np.zeros((4, 4))`) — will silently produce wrong results (or index errors) if used with a different number of classes. Assumes labels are already integer-encoded `0..3`.
- Normalizes by dividing by the total number of samples (not per-row), so this is a joint-probability-style matrix, not the typical per-class-normalized confusion matrix.

### `compute_metrics_given_path(model, loader_list, path, device='cpu') -> dict`
- Lists all files in `path` (assumed to be `.pth` checkpoints), and for each one: loads `state_dict` into `model` (CPU-mapped), then for each loader in `loader_list` computes metrics and appends to accumulator lists.
- **Likely bug**: unpacks `compute_metrics(...)` (which returns a `dict`) into six variables via `accuracy, cohen_kappa, sensitivity, specificity, f1, confusion_matrix = compute_metrics(...)` — unpacking a `dict` iterates over its **keys** (strings), not its values, so this would raise a `ValueError` (wrong number of values, since dict key order matches but types are wrong) or silently assign key-name strings instead of the actual metric values. This function is likely broken/unused in its current form.
- Returns `metrics_per_file`, a dict of lists keyed by metric name, one entry appended per (file, loader) pair.

## Dependencies
- Internal: none (standalone module); expects to be called with a model that implements `.classify(x)`.
- External: `numpy`, `torch`, `sklearn.metrics` (`cohen_kappa_score`, `accuracy_score`, `recall_score`, `f1_score`, `confusion_matrix`), `os`.

## Notes
- `compute_metrics_given_path` has a likely dict-unpacking bug (see above) — treat it as unreliable/untested before reusing.
- `compute_multiclass_confusion_matrix` hardcodes 4 classes; this matches the 4-class BCI Competition IV 2a (left/right/foot/tongue motor imagery) use case elsewhere in the repo, but will break for other datasets with a different number of classes.
- `compute_specificity_binary`'s TN/FP indexing should be double-checked against `sklearn.metrics.confusion_matrix`'s actual label-to-index mapping for the data in use; it may give specificity/sensitivity transposed.
