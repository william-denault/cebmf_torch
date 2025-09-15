# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cebmf_torch"
copyright = "2025, william-denault"
author = "william-denault"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',
    "sphinx.ext.mathjax",  # Required for LaTeX math rendering
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.imgconverter',
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',           # for Google/NumPy style docstrings
    'sphinx_autodoc_typehints',      # optional, for type hints
    'sphinx.ext.viewcode',           # adds links to source code
]

myst_enable_extensions = [
    "dollarmath",   # enables $$...$$ block math
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

nb_execution_mode = "off"  # never execute notebooks

# Ensure the package can be imported by Sphinx
sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.path.abspath("../"))
sys.path.insert(1, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../src/"))

# autosummary removed to avoid importing modules at builder init

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"

# Mock heavy or optional imports that aren't needed to build docs
autodoc_mock_imports = []

# Autosummary: generate stub pages
autosummary_generate = True

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
# source_suffix = '.md'
master_doc = "index"

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
