# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))
start_year = 2025
current_year = datetime.now().year

project = "leap-c"
copyright = f"{start_year}-{current_year}, Dirk, Jasper, Leonard"
author = "Dirk, Jasper, Leonard"
release = "2025"


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # For math equations
    "sphinx.ext.intersphinx",  # For linking to other documentation
    "myst_nb",  # Markdown + MyST-NB notebook support (supersedes myst_parser)
]

# MyST-NB registers parsers for .md (MyST markdown) and .ipynb (notebook).
# Keep .rst supported by default docutils.
source_suffix = [".rst", ".md", ".ipynb"]

# MyST-NB: render notebooks statically — do not execute cells at build time.
# Tutorials are intended to be *run* by users locally; the docs build just
# renders the source (prose + code + any cell outputs already stored).
nb_execution_mode = "off"

# MyST parser extensions used in tutorials (math, code-cell directive).
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "colon_fence",
]

html_theme = "furo"
