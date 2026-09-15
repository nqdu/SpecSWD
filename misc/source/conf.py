"""Sphinx configuration for the SpecSWD user and API documentation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

project = "SpecSWD"
author = "Nanqiao Du"
copyright = "Nanqiao Du"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

myst_enable_extensions = ["amsmath", "dollarmath"]
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
# API pages can still be generated before the compiled extensions are built.
autodoc_mock_imports = ["specd.lib.libswd", "specd.lib.cps330"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["static"]

latex_engine = "pdflatex"
latex_elements = {
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
""",
}
