# `library/analysis/dtw_analysis.py`

## Purpose
Provides a single helper to compute a channel-by-channel reconstruction error between two EEG tensors using the soft-DTW implementation from `library.training.soft_dtw_cuda`. Used in analysis pipelines to score how well a model's reconstruction matches the original signal.

## Key functions

### `compute_recon_error_between_two_tensor(x_1, x_2, device='cpu', average_channels=False, average_time_samples=False) -> torch.tensor`
- Computes the soft-DTW distance between two tensors of identical shape `B x 1 x C x T` (batch, depth=1, EEG channels, time samples).
- Iterates over the channel dimension `C`, computing soft-DTW independently per channel (soft-DTW is defined for 1D sequences), and stacks results into a `B x C` tensor.
- `average_time_samples=True` divides the per-channel error by `T` (normalizes by sequence length).
- `average_channels=True` additionally averages over the channel dimension, collapsing output to shape `B`.
- Raises `ValueError` if `x_1.shape != x_2.shape`.
- Moves inputs to `device` and runs `SoftDTW(use_cuda=(device=='cuda'), normalize=False)` per channel; result is moved back to CPU.

## Dependencies
- Internal: `library.training.soft_dtw_cuda.SoftDTW` (the differentiable CUDA/CPU soft-DTW loss).
- External: `torch`.

## Notes
- This is a thin wrapper reused across several analysis scripts (e.g. `library/analysis/support.py:compute_loss_dataset` and `compute_dtw_between_two_batch` duplicate very similar per-channel DTW loops inline rather than calling this function — worth consolidating if touching this code).
- No gradient tracking concerns here since it's only used for evaluation/analysis, not training (contrast with `soft_dtw_cuda.py`, which is used as a differentiable training loss).
