# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import hipct_reg

project = "hipct-reg"
project_copyright = "2024, David Stansby"
author = "David Stansby"
version = hipct_reg.__version__
release = hipct_reg.__version__
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

default_role = "any"
nitpicky = True
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
sphinx_gallery_conf = {
    "examples_dirs": "tutorial",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
    "filename_pattern": "/",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "theme.css",
]
html_theme_options = {
    "logo": {
        "text": "hipct-reg",
    },
}

html_use_index = False
html_show_sourcelink = False
html_show_copyright = False
html_sidebars: dict[str, list[str]] = {"**": []}
