# `library/__init__.py`

## Purpose
Empty file. Its only role is to mark `library/` as a Python package so that submodules (`library.model`, `library.training`, `library.dataset`, `library.analysis`, `library.config`, `library.plot`) can be imported with the `from library.subpackage import ...` syntax described in the root `README.md`.

## Key classes
None.

## Key functions
None.

## Dependencies
None.

## Notes
- No package-level exports are defined here (no `__all__`, no re-exports). Every import must reach into the specific submodule file (e.g. `from library.model import hvEEGNet`), not just `library`.
