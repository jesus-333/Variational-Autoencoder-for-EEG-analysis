# `library/config/__init__.py`

## Purpose
Empty file. Its only role is to mark `library/config` as a Python package so that modules inside it (`config_dataset.py`, `config_model.py`, `config_plot.py`, `config_training.py`) can be imported as `library.config.<module>`.

## Key functions
None — the file has no content.

## Dependencies
None.

## Notes
No re-exports are defined here; callers must import the specific submodule they need (e.g. `from library.config import config_model as cm`).
