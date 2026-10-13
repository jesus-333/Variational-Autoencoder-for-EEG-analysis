# `library/analysis/sdtw_py.py`

## Purpose
A pure-Python/NumPy (Numba-accelerated) implementation of the soft-DTW distance between two 1D sequences, intended purely as an **analysis metric** — it does not track gradients and is not meant to be used as a training loss (contrast with `library/training/soft_dtw_cuda.py`, which is the differentiable/trainable version).

## Key functions

### `soft_dtw_python(x: np.array, y: np.array, gamma: float = 1, p_distance_grade: int = 2)`
- Computes the full soft-DTW distance matrix via `compute_distance_matrix` and returns the bottom-right element (the final alignment cost).

### `compute_distance_matrix(x, y, gamma=1, p_distance_grade=2) -> np.array` (Numba `@jit(nopython=True)`)
- Builds the DP distance matrix for soft-DTW by iterating **anti-diagonals** (from the upper-left corner down to the lower-right), rather than row-by-row — an optimization/parallelization-friendly traversal order.
- Requires `len(x) == len(y)` (raises `ValueError` otherwise) and `gamma > 0`.
- Uses `np.finfo(np.float16).max` as a sentinel for "out of bounds" neighbor values at the matrix edges.
- Each cell = `lp_distance(x[i], y[j]) + softmin(left, upper_left, upper)`.

### `compute_softmin(values_list, gamma=1) -> float` (Numba `@jit`)
- Computes `-gamma * log(sum(exp(-v/gamma)))` (the smooth min used by soft-DTW). Falls back to `np.min(values_list)` if the exponential sum underflows to exactly 0 (numerical-precision guard).

### `lp_distance(a: float, b: float, p_distance_grade: int = 2) -> float` (Numba `@jit`)
- Computes the Lp distance `(|a-b|^p)^(1/p)` between two scalars (used as the local cost in the DTW matrix).

## Dependencies
- External: `numpy`, `numba` (`@jit(nopython=True, parallel=False)` on all three helper functions).

## Notes
- Requires `x` and `y` to have equal length (unlike the CUDA soft-DTW in `library/training/soft_dtw_cuda.py`, which is often used with differing sequence lengths between the two inputs' time dimension via batched tensors).
- This is a reference/debugging implementation; `library/training/soft_dtw_cuda.py` is the one actually used in training loops and most analysis code (see `library/analysis/dtw_analysis.py`, `library/analysis/support.py`).
- Numba JIT compilation means first call per process incurs compile overhead.
