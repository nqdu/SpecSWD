# conf.py

project = 'SpecSWD'
author = 'Nanqiao Du'

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../../'))

latex_engine = 'pdflatex'  
# Set up LaTeX formatting elements
latex_elements = {
    'preamble': r'''
    \usepackage{amsmath}
    \usepackage{amssymb}
    '''
}

# Add myst-parser to the extensions list
extensions = [
    'myst_parser',  # Markdown parser
    'sphinx.ext.mathjax',  # for math rendering in HTML
    'sphinx.ext.autodoc',  # For generating Python API docs
    'sphinx.ext.napoleon',  # For Google and NumPy docstring styles (optional)
    'sphinx.ext.viewcode',  # To link source code in the API docs (optional)
]

# You might have:
autodoc_default_options = {
    'members': True,
    'undoc-members': True,  # <--- include this to document all functions
    'private-members': False,
    'special-members': False,
    'show-inheritance': True,
}

# Add .md as a valid source suffix
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',  # Enable Markdown support
}

# Optional: Set up the master document
master_doc = 'index'  # Default is 'index.rst', which you can change as needed
html_theme = 'sphinx_rtd_theme'