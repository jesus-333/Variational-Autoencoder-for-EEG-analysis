# Quickstart — Variational-Autoencoder-for-EEG-analysis

This file is the entry point for an LLM/agent picking up work in this repository. Read this first, then use [0_index.md](0_index.md) to jump to the per-file docs under `library/`.

## What this repo is

A PhD-era research codebase (Alberto Zancanaro) for deep-learning analysis of EEG signals, centered on **variational autoencoders (VAEs)** built on top of **EEGNet**-style convolutional architectures. It backs several publications:

- vEEGNet (v1/v2): a VAE whose encoder/decoder are EEGNet-shaped, learning latent representations to classify and reconstruct raw EEG.
- **hvEEGNet**: a *hierarchical* VAE (NVAE-inspired) built by splitting a vEEGNet into its temporal/spatial/separable-convolution stages and treating each as a level of a deep latent hierarchy. This is the model the current `jesus-experiment` branch is actively iterating on (`hvEEGNet_V2.py`, see `training_scripts/d2a_hvEEGNet_V2.py`).

The repo also supports EEG classification (EEGNet, MBEEGNet, ChWi), federated-learning experiments, soft-DTW-based reconstruction losses, and extensive post-hoc analysis (reconstruction error, latent-space clustering, variability/Wasserstein studies) across many public EEG datasets (BCI Competition IV 2a "d2a", Zhou2016, Weibo2014, Cho2017, Ofner2017, GrosseWentrup2009, Lee2019_MI, Schirrmeister2017, BI2014a, SEED, TUAR, ...).

## Structure of the repository

```
library/            <-- the actual "package" (see below — not a real installable package)
  model/             model definitions (EEGNet, MBEEGNet, vEEGNet, hvEEGNet, hvEEGNet_V2, ChWi, aEEGNet, classifier...)
  training/          training loops, loss functions (incl. soft-DTW), wandb integration, federated learning
  dataset/           dataset download (moabb-based) + preprocessing + PyTorch Dataset wrappers
  config/            factory functions returning config *dictionaries* for model/training/dataset/plot
  analysis/          DTW analysis, latent-space analysis, normalization, visualization helpers
  plot/              preprocessing/STFT plotting helpers
  README/            pre-existing human-written READMEs (model/training/dataset/code_info) — partially outdated, cross-checked against code while writing this wiki
training_scripts/    per-dataset, per-model executable training scripts (e.g. d2a_hvEEGNet_V2.py)
analysis_script/     large collection of ad-hoc, often one-off post-training analysis/plotting scripts, organized loosely by dataset or topic
examples/            small runnable examples (reconstruction, model creation, training) meant as entry points for new users
json_config/         JSON config files for some experiments (e.g. ChWi)
Saved Model/, Saved Results/, model_weights/, Other/   large trees of trained weights, result tables, figures — data, not code
```

**Important: `library/` is organized like a Python package (has `__init__.py` files throughout) but was never actually packaged** — there's no `setup.py`/`pyproject.toml`. Imports are path-based (`from library.model import vEEGNet`, run from the repo root). Do not assume `pip install -e .` works or that `library` is importable from arbitrary working directories.

The repo owner has noted that, since this is an older/less-structured repo compared to more recent ones, parts of it (especially the hvEEGNet-related model/training code) may eventually be migrated into a newer repo (`jade`) — no concrete decision has been made yet. Don't assume this repo is the permanent home of this code.

## How the pieces typically fit together

1. Get a `dataset_config` dict from `library/config/config_dataset.py`, or write your own. Download/load data via `library/dataset/download.py` (mostly wraps `moabb`, with some dataset-specific local-file paths e.g. TUAR, SEED).
2. Preprocess with `library/dataset/preprocess.py` (filtering, baseline removal, STFT/ERS computation), wrap tensors in a PyTorch `Dataset` from `library/dataset/dataset_time.py` or `dataset_stft.py`.
3. Get a `model_config` dict from `library/config/config_model.py` and instantiate a model from `library/model/` — **models take one config dict, not a long argument list**; the config dict's exact keys are documented per-file in `library/model/*.md`.
4. Get a `train_config` dict from `library/config/config_training.py`, then train via `library/training/train_generic.py` (the generic dispatcher) or a model-specific `train_<model>.py`. Each model-specific training file conventionally exposes `train_and_test_model(...)`, `train(...)`, `test(...)`, `train_epoch(...)`, `validation_epoch(...)`, `check_train_config(...)` (see `library/README/README_training.md`).
5. Optionally track runs with Weights & Biases via `library/training/wandb_training.py` (`train_wandb_<model>(...)` wrappers) — same call signature as the non-wandb training functions.
6. Post-hoc analysis (reconstruction error, DTW, latent-space clustering, spectra, variability studies) lives in `library/analysis/` (reusable functions) and `analysis_script/` (one-off scripts that call into `library/analysis/`, often per-dataset).

See `library/README/README_model.md` and `README_training.md` (pre-existing human docs, cross-referenced while building this wiki) for example config dicts for EEGNet/MBEEGNet/vEEGNet/hvEEGNet.

## Conventions worth knowing

- **Config-dict-driven**: almost everything (models, training, dataset, plots) is configured via plain Python `dict`s produced by factory functions in `library/config/`, not dataclasses or CLI args. When adding a new parameter, follow this pattern rather than introducing a new config mechanism.
- **Conv-channel dimension gotcha**: models expect EEG input shaped `(batch, 1, eeg_channels, time_samples)` — the `1` is the Conv2d "channel" dimension and has nothing to do with EEG channels. Easy to get the two kinds of "channel" confused.
- **hvEEGNet's `hierarchical_VAE.py`** is a generic, architecture-agnostic hierarchical-VAE engine (any `encoder_cell_list`/`decoder_cell_list` of `nn.Module`s); `hvEEGNet.py` and `hvEEGNet_V2.py` are both thin EEG-specific wrappers around it. If you need a new hierarchical model, consider reusing `hierarchical_VAE.py` directly rather than writing a new hierarchy engine.
- **soft-DTW**: `library/training/soft_dtw_cuda.py` is a vendored third-party CUDA+CPU soft-DTW implementation and is the one actually used everywhere (`loss_function.py`, `library/analysis/*`, scripts). `soft_dtw_cuda_backup.py` is an unused near-duplicate — don't edit it expecting it to affect anything, and don't assume both need to be kept in sync.
- **wandb is optional but still imported in places** — training can run without a wandb account, but the `wandb` package itself is still a hard dependency for some files/imports per the existing README.
- Many large data/weights directories (`Saved Model/`, `Saved Results/`, `model_weights/`, `wandb/`) are present in the working tree; they are experiment artifacts, not source code — don't read them for understanding the pipeline.

## Known issues found while documenting this codebase

These were spotted incidentally while writing the per-file docs (see the "Notes" section of the relevant `.md` for full detail). They have **not** been independently fixed or exhaustively verified beyond a code read — treat as a to-do/investigate list, not a confirmed bug report:

- `library/config/config_model.py`: a dead `elif` branch in `get_config_vEEGNet`, and a self-recursive/missing-key issue in the ChWi and aEEGNet config builders.
- `library/config/config_plot.py`: appears to be dead code per its own docstring (not actually used elsewhere).
- `library/analysis/latent_space.py`: references a `dtw_analysis.compute_dtw_softDTWCuda` that looks dead/unreachable; `clustering_evaluation` has an indentation bug in its intra-cluster distance loop.
- `library/analysis/visualizer.py`: a `fig.tigh_layout()` typo (no-op instead of `tight_layout()`), and a `.to('cpu')` call on what is actually a plain string label.
- `library/analysis/dtw_analysis.py` / `support.py`: duplicated per-channel DTW loop logic across the two files.
- `library/dataset/preprocess.py`: `compute_stft` computes a frequency filter but then discards it unconditionally (dead computation / likely missing `return` or assignment).
- `library/dataset/download.py` (SEED): `get_dataset_SEED_split_in_train_test_validation` appears to use `idx_test` where `idx_validation` was probably intended (copy-paste bug).
- `library/dataset/download.py`: inconsistent µV scaling conventions between the "automatic" and "handmade" moabb download code paths.
- `library/dataset/download.py` (TUAR): dataset labels are random placeholders, not real labels — don't train a classifier on TUAR labels expecting them to be meaningful.
- `library/model/support_function.py`: `reparametrize`/`sample_layer` contain leftover NaN-debugging `print()` spam (also present in `hierarchical_VAE.py`'s `hVAE_decoder.forward`).
- `library/model/classifier.py`: `freeze_encoder` uses `param.require_grad` (no such attribute — should be `requires_grad`), so it's currently a no-op and the encoder is never actually frozen.
- `library/model/ChWi.py`: a `==` vs `=` typo in `check_config`, plus float-division/missing-`dim` issues in `domain_predictor`.
- `library/model/aEEGNet.py`: some `forward()` methods look like unfinished stubs; a `len(x.shape != 4)` bug (almost certainly meant `len(x.shape) != 4`) appears in two attention modules.
- `library/model/Decoder_EEGNet.py`: `EEGNet_Decoder_Upsample` has a suspicious batchnorm/channel-count ordering issue worth double-checking before relying on it.

## Where to go next

- Full per-file reference: [0_index.md](0_index.md).
- Model architecture questions: start with `library/model/hierarchical_VAE.md`, `hvEEGNet.md`, `hvEEGNet_V2.md`.
- Training pipeline questions: start with `library/training/train_generic.md`.
- Dataset/preprocessing questions: start with `library/dataset/download.md` and `preprocess.md`.
- Config schema questions (what keys does a model/training config dict need): the relevant file under `library/config/`.
