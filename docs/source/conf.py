# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
import os
import json
import re

import py_neuromodulation

print("CURRENT WORKING DIRECTORY")
print(os.getcwd())
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "source":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "py_neuromodulation")
print(f"Script Directory to add: {SCRIPT_DIR}")
sys.path.append(SCRIPT_DIR)

print(sys.path)


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
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx_gallery.gen_gallery',
    'nbsphinx',
    'recommonmark',
    'sphinx_togglebutton',
]

source_suffix = ['.rst', '.md', ]

autosummary_generate = True


sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    'plot_gallery': True,  # Avoid annoying Unicode/bool default warning
    'thumbnail_size': (160, 112),
    'image_scrapers': ("matplotlib", ),
    'show_memory': True,
    'notebook_images': 'https://py-neuromodulation.readthedocs.io/en/latest/',
}

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']

html_theme_options = {
    "show_nav_level": 4,
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/neuromodulation/py_neuromodulation",
            icon="fa-brands fa-square-github",
        )
    ],
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numba": ("https://numba.readthedocs.io/en/latest", None),
    "mne": ("https://mne.tools/stable", None),
    "pandas" : ("https://pandas.pydata.org/docs", None),
}

