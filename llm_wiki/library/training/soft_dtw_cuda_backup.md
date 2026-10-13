# `library/training/soft_dtw_cuda_backup.py`

## Purpose
A backup/snapshot copy of `soft_dtw_cuda.py` (same vendored third-party soft-DTW implementation, same MIT license header). **Not imported anywhere in the codebase** — confirmed via `grep -rn "soft_dtw_cuda" --include=*.py .` across the whole repo: every import site (`library/analysis/support.py`, `library/analysis/dtw_analysis.py`, `library/model/hvEEGNet.py`, `library/training/loss_function.py`, several `analysis_script/*`/`examples/*` scripts) references `soft_dtw_cuda` (no `_backup` suffix). This file appears to exist purely as a point-in-time backup kept alongside the active file, per the user's own description of the repo as having "less defined structure" than their newer projects.

## Key classes / functions
Identical in structure to `soft_dtw_cuda.py`: `SoftDTW(torch.nn.Module)`, `_SoftDTWCUDA(Function)`, `_SoftDTW(Function)`, plus the same `timed_run`/`profile` benchmarking helpers. See `soft_dtw_cuda.md` for full documentation of the shared logic — not repeated here since it is the same code.

## Dependencies
Same as `soft_dtw_cuda.py`: `numpy`, `torch`, `numba`, `math`, `os`.

## Notes
- **Only functional difference found** between this file and `soft_dtw_cuda.py`: inside `_SoftDTWCUDA.forward` and `_SoftDTWCUDA.backward`, this backup uses `cuda.as_cuda_array(...)` to wrap tensors before passing them into the Numba CUDA kernels, whereas the active `soft_dtw_cuda.py` uses `cuda.to_device(...)`. Everything else (CPU implementation, `SoftDTW` module, docstrings, hardcoded `NUMBAPRO_NVVM`/`NUMBAPRO_LIBDEVICE` Windows paths) is byte-for-byte the same.
- Because it's dead code with no import sites, treat this file as historical reference only; if it's ever needed again (e.g. `cuda.as_cuda_array` behaves differently than `cuda.to_device` for some Numba/CUDA version combination), compare directly against `soft_dtw_cuda.py` rather than assuming this is current. A future cleanup pass could likely delete this file outright, pending the user's decision on repo reorganization.
