# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

import sphinx_rtd_theme
import sys
import os


# At top on conf.py (with other import statements)
import recommonmark
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser

# -- Project information -----------------------------------------------------

project = 'py_neuromodulation'
copyright = '2021, Timon Merk'
author = 'Timon Merk'

# The full version, including alpha/beta/rc tags
print("CURRENT WORKING DIRECTORY")
print(os.getcwd())
#sys.path.insert(0, os.path.abspath('pyneuromodulation'))
#sys.path.insert(0, os.path.abspath('examples'))
print('adding path')
#print(os.path.join(os.pardir, 'pyneuromodulation'))
#sys.path.append(os.path.join(os.pardir, 'pyneuromodulation')) THIS DOESN'T WORK FOR SOME REASON..
#sys.path.append(os.path.join(os.pardir, 'examples'))

sys.path.insert(0, r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
#sys.path.append(r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
print(sys.path)
# -- General configuration ---------------------------------------------------
# sys.path.insert(0, os.path.abspath('../pyneuromodulation'))
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}
source_suffix = ['.rst', '.md']
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'numpydoc',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'recommonmark'
]

#source_suffix = ['.rst', '.md']



autosummary_generate = True
autodoc_default_options = {'inherited-members': None}

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = 'sphinx_rtd_theme'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
