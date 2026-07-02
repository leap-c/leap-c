# Notebooks

These are [marimo](https://marimo.io) notebooks (plain `.py` files, not `.ipynb`).

- `intro.py` — high-level intro using `create_planner`.
- `diff_mpc_tutorial.py` — differentiable-MPC tutorial built around `AcadosDiffMpcTorch`.

## Running

marimo is behind the `notebook` extra, so pass `--extra notebook` (or run
`uv sync --extra notebook` once and drop the flag afterwards):

```bash
# Interactive editor (edit + run cells) — usual choice
uv run --extra notebook marimo edit notebooks/diff_mpc_tutorial.py

# Read-only app view (runs it as a deployed app, cells hidden)
uv run --extra notebook marimo run notebooks/diff_mpc_tutorial.py
```

marimo prints a `http://localhost:2718` URL and normally opens it in your
browser automatically.
