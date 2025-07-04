"""
Cebra Decoding with no training Example
======================================

The following example shows how to use the Cebra package for decoding without patient-individual training.
Note that the plot_8_cebra_example.py file just hosts the example_cebra_decoding.html file due to computational limitations.
When re-running this file, please refer to the example_cebra_decoding.ipynb in the examples directory.

"""

import os

# load example_cebra_decoding.html
with open(os.path.join("..", "examples", "example_cebra_decoding.html"), "rt") as fh:
    html_data = fh.read()

tmp_dir = os.path.join("..", "docs", "source", "auto_examples")
if os.path.exists(tmp_dir):
    # building the docs with sphinx-gallery
    with open(os.path.join(tmp_dir, "out.html"), "wt") as fh:
        fh.write(html_data)
# set example path for thumbnail
# sphinx_gallery_thumbnail_path = '_static/CEBRA_embedding.png'


# %%
# CEBRA example
# -------------
#
# .. raw:: html
#     :file: out.html
