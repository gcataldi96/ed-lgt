# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Exact Diagonalization for Lattice Gauge Theories"
copyright = "2023, Giovanni Cataldi"
author = "Giovanni Cataldi"

# The full version, including alpha/beta/rc tags
release = "31/07/2023"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.mathjax",
    # "sphinx_gallery.gen_gallery", # removed, problems on baltig gitlab CI/CD
]

# Resolve references to external libraries used in type annotations/docstrings.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# TO print init function of class
autoclass_content = "both"

# Keep annotations from cluttering signatures/pages and rely on explicit docstrings.
autodoc_typehints = "none"

# Better handling of NumPy-style docstring type fields.
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "np.ndarray": "numpy.ndarray",
    "np.array": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
    "numpy.ndarray": "numpy.ndarray",
    "np.uint16": "numpy.uint16",
    "np.float64": "numpy.float64",
    "np.int64": "numpy.int64",
    "csr_matrix": "scipy.sparse.csr_matrix",
    "csc_matrix": "scipy.sparse.csc_matrix",
    "spmatrix": "scipy.sparse.spmatrix",
    "scipy.sparse.matrix": "scipy.sparse.spmatrix",
    "sequence": "typing.Sequence",
    "sequence[int]": "``sequence[int]``",
    "list[str]": "``list[str]``",
    "list[int]": "``list[int]``",
    "tuple[int, int]": "``tuple[int, int]``",
    "tuple[float, float]": "``tuple[float, float]``",
    "tuple[numpy.ndarray, numpy.ndarray]": "``tuple[numpy.ndarray, numpy.ndarray]``",
    "tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]": "``tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]``",
    "numpy.int64": "``numpy.int64``",
    "numpy.float64": "``numpy.float64``",
    "numpy.uint16": "``numpy.uint16``",
    "QMBTerm": "``QMBTerm``",
    "QMB_state": "``QMB_state``",
    "callable": "collections.abc.Callable",
    "array-like": "``array-like``",
    "scalar": "``scalar``",
    "optional": "``optional``",
    "ints": "``ints``",
    "floats": "``floats``",
    "lists": "``lists``",
    "instance": "``instance``",
}

# Ignore a few legacy pseudo-types/inline-enum tokens still present in old docs.
nitpick_ignore_regex = [
    (r"py:class", r"^(list|tuple|dict|set|sequence)\[.*\]$"),
    (r"py:class", r"^(QMBTerm|QMB_state)$"),
    (r"py:class", r"^edlgt\.modeling\.qmb_term\.QMBTerm$"),
    (r"py:class", r"^edlgt\.models\.quantum_model\.QuantumModel$"),
    (r"py:class", r"^(ndarray|row_list|col_list|value_list|support_indices|support_coeffs|support_configs|support_keys|discarded_weight)$"),
    (r"py:class", r"^numpy\.(int64|float64|uint16)$"),
    (r"py:class", r'^\{"[^"]*"$'),
    (r"py:class", r'^".*"\}$'),
    (r"py:func", r"^single_structure_factor$"),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []

# -- sphinx galler configutations --
# sphinx_gallery_conf = {
#    "examples_dirs": "../examples",  # path to your example scripts
#    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
#    "filename_pattern": "/",  # pattern for files to be executed. This means all files
# }
