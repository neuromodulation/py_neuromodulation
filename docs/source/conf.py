# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from pathlib import Path
import os
from sphinx_gallery.sorting import FileNameSortKey

print("CURRENT WORKING DIRECTORY")
print(Path.cwd())

SCRIPT_DIR = Path(__file__).absolute().parent

if SCRIPT_DIR.name == "source":
    # this check is necessary, so we can also run the script from the root directory
    SCRIPT_DIR = SCRIPT_DIR.parent.parent / "py_neuromodulation"
print(f"Script Directory to add: {SCRIPT_DIR}")
sys.path.append(str(SCRIPT_DIR))

print(sys.path)

os.environ["MNE_LSL_LIB"] = str(SCRIPT_DIR.parent / "liblsl/noble_amd64/liblsl.1.16.2.so")

exclude_patterns = ["_build", "_templates"]


# -- Project information -----------------------------------------------------
project = "py_neuromodulation"
copyright = "2021, Timon Merk"
author = "Timon Merk"

source_parsers = {
    ".md": "recommonmark.parser.CommonMarkParser",
}

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "sphinx_gallery.gen_gallery",
    "sphinx_togglebutton",
    # "nbsphinx",
]

source_suffix = [
    ".rst",
    ".md",
]

autosummary_generate = True

PYDEVD_DISABLE_FILE_VALIDATION = 1

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
    "within_subsection_order": FileNameSortKey,
}

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]

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
    "pandas": ("https://pandas.pydata.org/docs", None),
}
