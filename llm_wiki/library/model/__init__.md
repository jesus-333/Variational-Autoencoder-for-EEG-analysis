# `library/model/__init__.py`

## Purpose
Empty file. Its only role is to mark `library/model/` as a Python package so that `from library.model import ...` imports work.

## Notes
No exports, no side effects. All model files inside `library/model/` are imported with explicit submodule paths (e.g. `from . import EEGNet`, `from library.model import hvEEGNet_V2`), not re-exported from here.
