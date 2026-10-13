# `library/model/MBEEGNet.py`

## Purpose
PyTorch implementation of MBEEGNet ([mdpi](https://www.mdpi.com/2079-6374/12/1/22), [pubmed](https://pubmed.ncbi.nlm.nih.gov/35049650/)): three parallel `EEGNet` branches with different temporal kernel sizes, whose flattened feature vectors are concatenated. Used as an alternative encoder for `vEEGNet` (`type_encoder == 1`), though the matching decoder path is not implemented (see Notes).

## Key classes

### `MBEEGNet(nn.Module)`
- `__init__(config)`: builds 3 `EEGNet` instances (`eegnet_1/2/3`) sharing a base `config['eegnet_config']` dict, but each run overwrites `eegnet_config['c_kernel_1']` with `config['temporal_kernel_1'|'2'|'3']` and `eegnet_config['dropout']` with `config['dropout_1'|'2'|'3']` before constructing the branch, and forces `eegnet_config['flatten_output'] = True`.
  - **Note:** because the three branches mutate and reuse the *same* `eegnet_config` dict object sequentially, only the last-set `c_kernel_1`/`dropout` values are visible if you inspect `config['eegnet_config']` after construction — the per-branch values are baked into each `EEGNet` instance's layers at construction time, so this works, but it's a shared-mutable-dict pattern to be careful with if `eegnet_config` is reused elsewhere.
  - Config dict keys: `eegnet_config` (base dict, see `EEGNet.py`), `temporal_kernel_1/2/3`, `dropout_1/2/3`, plus (needed later by `MBEEGNet_Classifier`) `C`, `T`, `n_classes`.
- `forward(x)`: runs `x` through all three EEGNet branches independently and concatenates their flattened outputs along dim 1 → `B x (N1+N2+N3)`.

### `MBEEGNet_Classifier(nn.Module)`
- Wraps `MBEEGNet` with `Linear(input_neurons, n_classes) + LogSoftmax(dim=1)`, mirroring `EEGNet_Classifier`.
- `compute_number_of_neurons(C, T)`: dummy forward pass with a `1 x 1 x C x T` input to determine classifier input size.
- `classify(x, return_as_index=True)`: same convenience pattern as `EEGNet_Classifier.classify`.

## Dependencies
- Internal: `library.model.EEGNet`.
- External: `torch`, `torch.nn`.

## Notes
- The original MBEEGNet paper's final layer is a classifier; `MBEEGNet` itself only returns the concatenated feature vector (matching the EEGNet convention in this repo).
- `vEEGNet.py` references `type_encoder == 1` (MBEEGNet) for the encoder, but the corresponding decoder branch (`type_encoder == 1` in the decoder-creation code) actually instantiates a bare `MBEEGNet` (not a dedicated decoder) with a `# TODO Implement MBEEGNet decoder` comment noting it was "indefinitely postponed" because hvEEGNet (EEGNet-based) performed better — so the MBEEGNet encoder path of `vEEGNet` is effectively non-functional for reconstruction.
