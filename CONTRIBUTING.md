# Contributing

1. Create a virtual environment and install with `pip install -e ".[ml,dev]"` (see `docs/INSTALLATION.md`).
2. Work on a branch; keep changes small and focused.
3. Before opening a pull request:
   ```bash
   ruff check src tests scripts app.py
   pytest
   ```
4. Tests must not download data or train for more than a few seconds — use the synthetic fixtures in `tests/conftest.py`.
5. Any number you add to docs or the dashboard must come from a file produced by `sar-flood evaluate` (or be clearly marked as historical) and state the split and data it was measured on.
6. Do not commit datasets, checkpoints under `outputs/`, or secrets; `.gitignore` covers the defaults.
