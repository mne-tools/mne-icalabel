"""Configure details for documentation with sphinx."""

import os
import subprocess
import sys
import warnings
from datetime import date

import mne
import sphinx_gallery  # noqa: F401
from mne.fixes import _compare_version
from sphinx_gallery.sorting import ExampleTitleSortKey

sys.path.insert(0, os.path.abspath(".."))
import mne_icalabel  # noqa: E402

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
curdir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(curdir, "..")))
sys.path.append(os.path.abspath(os.path.join(curdir, "..", "mne_icalabel")))
sys.path.append(os.path.abspath(os.path.join(curdir, "sphinxext")))

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
needs_sphinx = "4.0"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "numpydoc",
    "sphinx_copybutton",
    "gh_substitutions",  # custom extension, see sphinxext/gh_substitutions.py
]

# configure sphinx-copybutton
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# generate autosummary even if no references
# -- sphinx.ext.autosummary
autosummary_generate = True

autodoc_default_options = {"inherited-members": None}
autodoc_typehints = "none"

# prevent jupyter notebooks from being run even if empty cell
# nbsphinx_execute = 'never'
# nbsphinx_allow_errors = True

error_ignores = {  # These we do not live by:
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
}

# -- numpydoc
# Below is needed to prevent errors
numpydoc_xref_param_type = True
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
numpydoc_xref_ignore = {
    "of",
    "shape",
    "n_components",
    "n_pixels",
    "n_classes",
}
numpydoc_xref_aliases = {
    # Python
    "Path": "pathlib.Path",
    # MNE
    "Epochs": "mne.Epochs",
    "ICA": "mne.preprocessing.ICA",
    "Info": "mne.Info",
    "Raw": "mne.io.Raw",
}
numpydoc_validate = True
numpydoc_validation_checks = {"all"} | set(error_ignores)
numpydoc_validation_exclude = {  # set of regex
    # we currently don't document these properly (probably okay)
    r"\.__getitem__",
    r"\.__contains__",
    r"\.__hash__",
    r"\.__mul__",
    r"\.__sub__",
    r"\.__add__",
    r"\.__iter__",
    r"\.__div__",
    r"\.__neg__",
}


default_role = "py:obj"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "MNE-ICALabel"
td = date.today()
copyright = "2021-%s, MNE Developers. Last updated on %s" % (td.year, td.isoformat())

author = "Adam Li"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = mne_icalabel.__version__
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False
html_copy_source = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url="https://github.com/mne-tools/MNE-ICALabel",
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["version-switcher", "navbar-icon-links"],
}
# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html"],
}

html_context = {
    "versions_dropdown": {
        "dev": "v0.3 (devel)",
        "stable": "v0.2",
        "v0.1": "v0.1",
    },
}

# html_sidebars = {'**': ['localtoc.html']}

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "joblib": ("https://joblib.readthedocs.io/en/latest", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "mne": ("https://mne.tools/dev", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "pooch": ("https://www.fatiando.org/pooch/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}
intersphinx_timeout = 5

# Resolve binder filepath_prefix. From the docs:
# "A prefix to append to the filepath in the Binder links. You should use this
# if you will store your built documentation in a sub-folder of a repository,
# instead of in the root."
# we will store dev docs in a `dev` subdirectory and all other docs in a
# directory "v" + version_str. E.g., "v0.3"
if "dev" in version:
    filepath_prefix = "dev"
else:
    filepath_prefix = "v{}".format(version)

os.environ["_MNE_BUILDING_DOC"] = "true"
scrapers = ("matplotlib",)
try:
    import mne_qt_browser

    _min_ver = _compare_version(mne_qt_browser.__version__, ">=", "0.2")
    if mne.viz.get_browser_backend() == "qt" and _min_ver:
        scrapers += (mne.viz._scraper._MNEQtBrowserScraper(),)
except ImportError:
    pass

compress_images = ("images", "thumbnails")
# let's make things easier on Windows users
# (on Linux and macOS it's easy enough to require this)
if sys.platform.startswith("win"):
    try:
        subprocess.check_call(["optipng", "--version"])
    except Exception:
        compress_images = ()

sphinx_gallery_conf = {
    "doc_module": ("mne_icalabel",),
    "reference_url": {
        "mne_icalabel": None,
    },
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
    "backreferences_dir": "generated",
    "plot_gallery": "True",  # Avoid annoying Unicode/bool default warning
    "thumbnail_size": (160, 112),
    "remove_config_comments": True,
    "min_reported_time": 1.0,
    "abort_on_example_error": False,
    # 'reset_modules_order': 'both',
    "image_scrapers": scrapers,
    "show_memory": not sys.platform.startswith(("win", "darwin")),
    "line_numbers": False,  # messes with style
    "within_subsection_order": ExampleTitleSortKey,
    "capture_repr": ("_repr_html_",),
    "junit": os.path.join("..", "test-results", "sphinx-gallery", "junit.xml"),
    "matplotlib_animations": True,
    "compress_images": compress_images,
    "filename_pattern": "^((?!sgskip).)*$",
}

# sphinxcontrib-bibtex
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""


# Enable nitpicky mode - which ensures that all references in the docs resolve.
nitpicky = True
nitpick_ignore = []
