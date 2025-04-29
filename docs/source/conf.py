# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "leap-c"
copyright = "2025, Dirk, Jasper, Leonard"
author = "Dirk, Jasper, Leonard"
release = "2025"


extensions = [
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

html_theme = "sphinx_rtd_theme"
