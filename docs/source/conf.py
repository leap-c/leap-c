# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))
start_year = 2025
current_year = datetime.now().year

project = "leap-c"
copyright = f"{start_year}-{current_year}, Dirk, Jasper, Leonard"
author = "Dirk, Jasper, Leonard"
release = "2025"


extensions = [
    "autoapi.extension",  # Generate API reference from source (no import required)
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",  # For math equations
    "sphinx.ext.intersphinx",  # For linking to other documentation
    "myst_parser",  # Add this for Markdown support
]

# Add this to enable Markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- AutoAPI -----------------------------------------------------------------
# Parses the source tree directly (without importing leap_c), so the docs build
# even when acados is not compiled. Private (_-prefixed), dunder, and imported
# members are hidden by omitting "private-members"/"special-members"/
# "imported-members" from autoapi_options. "undoc-members" is required: without
# it AutoAPI prunes the (empty) top-level leap_c package and emits no pages, so
# undocumented members are shown with their typed signatures (docstrings TBD).
autoapi_type = "python"
autoapi_dirs = ["../../leap_c"]
autoapi_root = "api/generated"
autoapi_add_toctree_entry = False  # placed manually via index.md
autoapi_member_order = "groupwise"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
]

# Render "Attributes:" docstring sections as :ivar: fields rather than separate
# attribute directives, so they do not collide with AutoAPI's own attribute
# entries (avoids "duplicate object description" warnings for dataclass fields).
napoleon_use_ivar = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

html_theme = "furo"
