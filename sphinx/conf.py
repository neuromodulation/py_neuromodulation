# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sphinx_rtd_theme
import sys
import os

print("CURRENT WORKING DIRECTORY")
print(os.getcwd())
print('adding path')
sys.path.insert(0, r'C:\Users\ICN_admin\Documents\py_neuromodulation\pyneuromodulation')
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
    'recommonmark',
    "linkify"
]

autosummary_generate = True
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# At the bottom of conf.py
def setup(app):
    app.add_config_value('recommonmark_config', {
            'url_resolver': lambda url: github_doc_root + url,
            'auto_toc_tree_section': 'Contents',
            }, True)
    app.add_transform(AutoStructify)
