# Building the documentation

leap-c's documentation is built with [Sphinx](https://www.sphinx-doc.org/). It combines:

- **Hand-written guides** in Markdown (via [MyST](https://myst-parser.readthedocs.io/))
  under `source/`.
- **An auto-generated API reference** produced by
  [sphinx-autoapi](https://sphinx-autoapi.readthedocs.io/) directly from the docstrings
  and type hints in `leap_c/`.

The rendered site is published to GitHub Pages at <https://leap-c.github.io/leap-c/>.

## Prerequisites

Install the documentation dependencies (the `docs` extra) into your environment.
Recommended (uv):

```bash
uv pip install -e ".[docs]"
```

Alternatively (pip):

```bash
pip install -e ".[docs]"
```

This pulls in Sphinx, the Furo theme, MyST-Parser, sphinx-autoapi, and sphinx-autobuild.

> **You do not need a working acados build to build the docs.** sphinx-autoapi parses the
> source tree statically (without importing `leap_c`), so the API reference builds from
> source alone.

## Build once

From this `docs/` folder:

```bash
make html
```

The site is written to `build/html/`. Open `build/html/index.html` in your browser. To
rebuild from scratch, run `make clean html`.

## Live preview (recommended while writing)

For a live-reloading server that rebuilds and refreshes the browser on every save:

```bash
make livehtml
```

Then open the printed URL (default <http://127.0.0.1:8000>). Edits to any page — or to a
docstring in `leap_c/` — refresh automatically.

## Layout

```
docs/source/
├── conf.py                 # Sphinx + autoapi configuration
├── index.md                # landing page + top-level toctree
├── installation.md
├── getting_started/
├── notebooks.md
├── troubleshooting.md
├── tutorials/              # hand-written tutorials
└── api/
    ├── index.md            # curated "Core API" page (main public classes)
    └── generated/          # produced by autoapi on every build (git-ignored; do not edit)
```

## Editing the API reference

The API reference is generated from the code — **there are no API pages to edit by hand.**
To improve it, edit the **docstrings and type hints in `leap_c/`**. Docstrings use the
[Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
(enforced by ruff). autoapi hides private (`_`-prefixed) members; anything documented is
picked up automatically on the next build.

To feature different entry points on the curated landing page, edit `source/api/index.md`.

## Deployment

Docs are built and deployed by `.github/workflows/documentation.yml`:

- **Pull requests** touching `docs/**` or any `*.py` trigger a build as a check (no deploy).
- **Pushes to `main`** build and publish to the `gh-pages` branch →
  <https://leap-c.github.io/leap-c/>.

The workflow installs the docs dependencies transitively via the `dev` extra, so any new
docs dependency added to `pyproject.toml` is picked up automatically.

## Troubleshooting

- **`sphinx-build: command not found` or import errors** — the `docs` extra is not installed
  in the active environment. Run `uv pip install -e ".[docs]"` (or `pip install -e ".[docs]"`).
- **Broken `python` / `pyenv` shim** — build with this project's package manager,
  [uv](https://docs.astral.sh/uv/), from the repository root; it needs no pre-existing
  environment:

  ```bash
  uv run --extra docs sphinx-build docs/source docs/build/html
  ```

  Alternatively, build in a dedicated virtual environment:

  ```bash
  python3 -m venv .venv-docs && . .venv-docs/bin/activate
  pip install -e ".[docs]"
  make html
  ```
