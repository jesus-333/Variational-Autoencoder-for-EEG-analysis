# `library/analysis/latent_space.py`

## Purpose
Computes and visualizes the latent-space embeddings of a (h)vEEGNet-style VAE model over a dataset, optionally reducing dimensionality (t-SNE/PCA) for plotting, and includes a generic clustering-evaluation helper.

## Key functions

### `compute_latent_space_dataset(model, dataset, config: dict, device='cpu')`
- Wraps `dataset` in a `DataLoader` (batch size from `config['batch_size']`, default 32) and runs the model (no grad) over all samples.
- Calls `model.encode(x, return_distribution=True)` to get `(z, mu, log_var)`; expects the model to expose `model.h_vae.hidden_space_size_flatten` (i.e. a hierarchical VAE wrapper, matching `library/model/hvEEGNet*.py`).
- `config['sample_from_distribution']` (bool, required): if True stores the reparameterized sample `z`, otherwise stores the mean `mu`, as the embedding for each item.
- `config['compute_recon_error']` (bool, required): if True also reconstructs each batch (`model.reconstruct`) and computes DTW reconstruction error per item via `dtw_analysis.compute_dtw_softDTWCuda` (note: this function name does not currently exist in `dtw_analysis.py`, which only defines `compute_recon_error_between_two_tensor` — likely stale/broken call, see Notes).
- `config['reduce_dimension']` (bool, optional, default False): if True, reduces the embedding to 2D via `reduce_dimension_lanten_space`.
- Returns `(z_reduced, recon_error)` if `compute_recon_error` else just `z_reduced`.
- **Bug-prone**: indexing `hidden_space_embedding[i*batch_size:(i+1)*batch_size]` assumes every batch is full size; the last (possibly smaller) batch will silently write into the wrong slice range if `len(dataset) % batch_size != 0`.

### `reduce_dimension_lanten_space(z: torch.Tensor, config: dict)`
- Dimensionality reduction dispatcher based on `config['reduction_methods']`: `'tsne'` (sklearn `TSNE`, `n_components=2`, configurable `perplexity`/`n_iter`), `'pca'` (sklearn `PCA(n_components=2)`), or `'umap'` (**not implemented** — just prints a message and returns `None` implicitly, which will break the caller).
- Raises `ValueError` for any other value.

### `plot_latent_space(z_reduced_list: list, plot_config: dict, color=None)`
- Plots one scatter subplot per entry in `z_reduced_list` (e.g. comparing t-SNE vs PCA side by side).
- `plot_config` keys used: `figsize`, `fontsize`, `markersize`, `colormap`, `alpha`, `reduction_method_name_list` (titles per subplot), `show_fig`.

### `clustering_evaluation(X, centers, labels)`
- Generic clustering-quality helper (not vEEGNet-specific): computes a pairwise proximity matrix (`scipy.spatial.distance.pdist`/`squareform`, Euclidean), an intra-cluster distance array `d`, and an inter-cluster distance matrix `D` between `centers`.
- Returns `(PM, d, D)`.
- Contains an acknowledged bug marker in comments (`[NEW CORRECTED CODE!!]`) and the intra-cluster distance normalization lines (`d[k] = d[k]/2` etc.) are **outside** the `for k in range(K)` loop due to indentation, so only the last cluster's `d[k]` gets normalized — the others remain raw sums. Treat `d` with caution.

## Dependencies
- Internal: `library.analysis.dtw_analysis` (for a `compute_dtw_softDTWCuda` call that appears to no longer exist in that module).
- External: `torch`, `numpy`, `matplotlib.pyplot`, `scikit-learn` (`TSNE`, `PCA`), `scipy.spatial.distance`.

## Notes
- `compute_latent_space_dataset` assumes a hierarchical VAE model shape (`model.h_vae.hidden_space_size_flatten`, `model.encode(..., return_distribution=True)`, `model.reconstruct`) — matches `library/model/hvEEGNet_V2.py` / `hierarchical_VAE.py` conventions, not plain `vEEGNet`.
- The `dtw_analysis.compute_dtw_softDTWCuda` reference is very likely dead/broken code from a prior refactor of `dtw_analysis.py` (that module now only exposes `compute_recon_error_between_two_tensor`). Verify before relying on the `compute_recon_error=True` path.
- `umap` branch is a no-op stub.
- `clustering_evaluation` looks like generic/borrowed boilerplate (variable names `K`, `PM`, `D` suggest a course/assignment origin) with a known indentation bug in the intra-cluster distance computation.
