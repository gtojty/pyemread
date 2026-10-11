"""Sphinx configuration for the pyemread documentation."""
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import pyemread  # noqa: E402

project = "pyemread"
author = "Tao Gong"
copyright = "2017-2026, Tao Gong"
version = release = pyemread.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autosummary_generate = True
autodoc_default_options = {"members": True, "undoc-members": False, "show-inheritance": False}
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_ivar = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}
html_theme = "pydata_sphinx_theme"
html_title = f"pyemread {release}"
html_theme_options = {"github_url": "https://github.com/gtojty/pyemread"}
exclude_patterns = ["_build"]
