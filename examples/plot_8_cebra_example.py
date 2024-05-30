"""
Cebra Decoding with no training Example
======================================

The following example show how to use the Cebra decoding without training.

"""

# %%
# sphinx_gallery_thumbnail_path = '_static/CEBRA_embedding.png'


import os

# load example_cebra_decoding.html
with open(os.path.join("..", "examples", "example_cebra_decoding.html"), "rt") as fh:
    html_data = fh.read()

tmp_dir = os.path.join("..", "docs", "source", "auto_examples")
if os.path.exists(tmp_dir):
    # building the docs with sphinx-gallery
    with open(os.path.join(tmp_dir, "out.html"), "wt") as fh:
        fh.write(html_data)

# %%
# CEBRA example
# -------------
# Show example
#
# .. raw:: html
#     :file: out.html
