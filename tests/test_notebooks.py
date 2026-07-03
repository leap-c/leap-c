"""Headless smoke tests for the marimo notebooks in ``notebooks/``.

Every top-level ``.py`` in ``notebooks/`` that is a marimo notebook is discovered
dynamically and executed end-to-end via ``marimo export html``, which runs all cells
top-to-bottom with no display and exits non-zero on any cell error. Adding or editing a
notebook is therefore covered automatically without touching this file.

The module self-skips where the ``notebooks`` extra (marimo) is not installed.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("marimo")  # skip cleanly where the notebooks extra isn't installed

# tests/test_notebooks.py -> repo root is parents[1]
REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
NOTEBOOKS = sorted(p for p in NOTEBOOKS_DIR.glob("*.py") if "marimo.App(" in p.read_text())


def test_notebooks_discovered():
    """Guard against a broken glob silently covering zero notebooks."""
    assert NOTEBOOKS, f"No marimo notebooks found in {NOTEBOOKS_DIR}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_runs_headless(notebook: Path, tmp_path: Path):
    """Each notebook runs to completion headless (all cells, no errors)."""
    out = tmp_path / f"{notebook.stem}.html"
    result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", notebook.name, "-o", str(out)],
        cwd=notebook.parent,  # so `from nb_utils... import ...` resolves
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=600,  # first run compiles the acados solver
    )
    assert result.returncode == 0, (
        f"{notebook.name} failed to run headless:\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
