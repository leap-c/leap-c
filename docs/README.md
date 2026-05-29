# Building the documentation

The docs are built with Sphinx + MyST-NB. Tutorials such as
`source/tutorials/race_car.md` are MyST-NB notebooks (jupytext-compatible);
the build renders them *statically* — cells are not executed.

## One-shot setup

Install the `docs` extra once (Sphinx, the theme, `myst-parser`, `myst-nb`):

```bash
pip install -e .[docs]
```

## Build HTML

From this `docs/` directory:

```bash
make html
```

Then open `build/html/index.html` in your favorite browser. For a clean
rebuild, use `make clean && make html`.

## Running a tutorial as a Jupyter notebook

The `.md` source is the single source of truth. To execute cells interactively,
convert it with jupytext:

```bash
pip install jupytext jupyter
jupytext --to notebook source/tutorials/race_car.md
jupyter lab source/tutorials/race_car.ipynb
```

The generated `.ipynb` is not checked into git.
