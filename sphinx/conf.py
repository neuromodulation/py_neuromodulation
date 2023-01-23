# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sphinx_rtd_theme
import sys
import os
#import pdb

#pdb.set_trace()
print("CURRENT WORKING DIRECTORY")
print(os.getcwd())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "sphinx":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "py_neuromodulation")
print(f"Script Directory to add: {SCRIPT_DIR}")
sys.path.append(SCRIPT_DIR)

#print("CURRENT WORKING DIRECTORY")
#print(os.getcwd())
#print('adding path')
#sys.path.insert(0, r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
print(sys.path)

# At top on conf.py (with other import statements)
import recommonmark
from recommonmark.transform import AutoStructify
from recommonmark.parser import CommonMarkParser

# -- Project information -----------------------------------------------------
project = 'py_neuromodulation'
copyright = '2021, Timon Merk'
author = 'Timon Merk'

source_parsers = {
    '.md': 'recommonmark.parser.CommonMarkParser',
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'numpydoc',
    'sphinx_rtd_theme',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'nbsphinx',
    'recommonmark'
]

# do not include .ipynb in source_suffix:
# https://stackoverflow.com/a/70474616/5060208

source_suffix = ['.rst', '.md', ] # '.ipynb'
autosummary_generate = True
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
