# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
import os
import json
import panel
import re

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
    'nbsite.pyodide'
]

source_suffix = ['.rst', '.md', ]
autosummary_generate = True

sphinx_gallery_conf = {
    "examples_dirs": "../../examples",
    "gallery_dirs": "auto_examples",
}

templates_path = ["_templates"]
exclude_patterns = []


html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']


from panel.io.convert import BOKEH_VERSION, MINIMUM_VERSIONS, PY_VERSION
from panel.io.convert import BOKEH_VERSION, MINIMUM_VERSIONS, PY_VERSION
from panel.io.resources import CDN_DIST

def base_version(version: str) -> str:
    """Extract the final release and if available pre-release (alpha, beta,
    release candidate) segments of a PEP440 version, defined with three
    components (major.minor.micro).

    Useful to avoid nbsite/sphinx to display the documentation HTML title
    with a not so informative and rather ugly long version (e.g.
    ``0.13.0a19.post4+g0695e214``). Use it in ``conf.py``::

        version = release = base_version(package.__version__)

    Return the version passed as input if no match is found with the pattern.
    """
    # look at the start for e.g. 0.13.0, 0.13.0rc1, 0.13.0a19, 0.13.0b10
    pattern = r"([\d]+\.[\d]+\.[\d]+(?:a|rc|b)?[\d]*)"
    match = re.match(pattern, version)
    if match:
        return match.group()
    else:
        return version

version = release = base_version(panel.__version__)

if panel.__version__ != version and (PANEL_ROOT / 'dist' / 'wheels').is_dir():
    py_version = panel.__version__.replace("-dirty", "")
    panel_req = f'./wheels/panel-{py_version}-py3-none-any.whl'
    bokeh_req = f'./wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl'
else:
    panel_req = f'{CDN_DIST}wheels/panel-{PY_VERSION}-py3-none-any.whl'
    bokeh_req = f'{CDN_DIST}wheels/bokeh-{BOKEH_VERSION}-py3-none-any.whl'

def get_requirements():
    with open('pyodide_dependencies.json') as deps:
        dependencies = json.load(deps)
    requirements = {}
    for src, deps in dependencies.items():
        if deps is None:
            continue
        src = src.replace('.ipynb', '').replace('.md', '')
        for name, min_version in MINIMUM_VERSIONS.items():
            if any(name in req for req in deps):
                deps = [f'{name}>={min_version}' if name in req else req for req in deps]
        requirements[src] = deps
    return requirements


nbsite_pyodide_conf = {
    'PYODIDE_URL': 'https://cdn.jsdelivr.net/pyodide/v0.23.1/full/pyodide.js',
    'requirements': [bokeh_req, panel_req, 'pyodide-http'],
    'requires': get_requirements()
}