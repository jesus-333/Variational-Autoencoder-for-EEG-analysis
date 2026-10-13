# `library/training/soft_dtw_cuda.py`

## Purpose
Third-party soft-DTW (differentiable Dynamic Time Warping) implementation vendored into the repo from [Maghoumi/pytorch-softdtw-cuda](https://github.com/Maghoumi/pytorch-softdtw-cuda) (MIT licensed, copyright notice preserved at the top of the file), with both a CUDA (Numba `@cuda.jit`) and a CPU (Numba `@jit(nopython=True, parallel=True)`) backend behind a single `torch.nn.Module` interface. This is the DTW engine used by every reconstruction loss in `loss_function.py` (`compute_dtw_loss_along_channels`) and by several `analysis_script/*` and `examples/*` scripts for computing DTW-based reconstruction error.

## Key classes
### `SoftDTW(torch.nn.Module)`
- `__init__(use_cuda, gamma=1.0, normalize=False, bandwidth=None, dist_func=None)`: `gamma` is the soft-min smoothing parameter; `bandwidth` is an optional Sakoe-Chiba band for pruning (0/`None` disables pruning); `dist_func` defaults to a batched squared-Euclidean distance (`_euclidean_dist_func`).
- `forward(X, Y)`: `X`/`Y` shaped `batch x seq_len x dims`. Picks the CUDA or CPU autograd `Function` via `_get_func_dtw` (dimension/shape assertions on batch size and feature dim). If `normalize=True`, computes the symmetrized/normalized SDTW `dtw(X,Y) - 0.5*(dtw(X,X)+dtw(Y,Y))` in one batched call (stacking `[X,X,Y]` vs `[Y,X,Y]`) rather than three separate calls.
- `_euclidean_dist_func(x, y)` (static): pairwise squared-Euclidean distance matrix between all timesteps of `x` and `y`.

### `_SoftDTWCUDA(torch.autograd.Function)`
- CUDA forward/backward using the diagonal/anti-diagonal parallel DP algorithm (one CUDA block per batch element, one thread per sequence position), implemented via the Numba `@cuda.jit` kernels `compute_softdtw_cuda` and `compute_softdtw_backward_cuda`. Uses `cuda.to_device(...)` to move tensors to CUDA explicitly before the kernel launch.

### `_SoftDTW(torch.autograd.Function)`
- CPU implementation (credited to Kanru Hua's `pytorch-softdtw`), using Numba-JIT'd, parallelized (`prange`) nested loops for forward (`compute_softdtw`) and backward (`compute_softdtw_backward`) passes. Converts to/from numpy arrays at the CPU/GPU boundary.

## Key functions
- `timed_run(a, b, sdtw)` / `profile(batch_size, seq_len_a, seq_len_b, dims, tol_backward)` — benchmarking helpers (not used by training code), run only under `if __name__ == "__main__"` to compare CPU vs CUDA timing/correctness.

## Dependencies
- External: `numpy`, `torch`, `numba` (`jit`, `prange`, `cuda`), `math`, `os`.

## Notes
- **Hardcoded, Windows-specific, machine-specific paths**: lines 35-36 set `os.environ['NUMBAPRO_NVVM']` and `['NUMBAPRO_LIBDEVICE']` to a literal `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\...` path. On any non-Windows machine, or a Windows machine without that exact CUDA version/path, this is at best a no-op (if Numba doesn't need it) and at worst actively wrong; it executes unconditionally at import time for every consumer of this module, including CPU-only usage. If running on Linux/Mac or a different CUDA version and hitting Numba/CUDA errors, check/remove these two lines first.
- This is the **active** implementation — see Notes in `soft_dtw_cuda_backup.md` for the (functionally near-identical, currently unused) backup file and the one concrete code difference between them (`cuda.to_device` here vs. `cuda.as_cuda_array` in the backup, inside `_SoftDTWCUDA.forward`/`backward`).
- Everything imported elsewhere in the repo (`library/analysis/support.py`, `library/analysis/dtw_analysis.py`, `library/model/hvEEGNet.py`, `library/training/loss_function.py`, several `analysis_script/*` and `examples/*` files) imports from `library.training.soft_dtw_cuda`, never from `soft_dtw_cuda_backup` — confirmed via repo-wide grep.
