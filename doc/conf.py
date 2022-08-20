"""Configure details for documentation with sphinx."""

import inspect
import os
import subprocess
import sys
from datetime import date
from importlib import import_module
from typing import Dict, Optional

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

# -- project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# General information about the project.
project = "MNE-ICALabel"
author = "Adam Li"
td = date.today()
copyright = f"2021-{td.year}, MNE Developers. Last updated on {td.isoformat()}"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = mne_icalabel.__version__
# The full version, including alpha/beta/rc tags.
release = version

gh_url = "https://github.com/mne-tools/mne-icalabel"

# -- general configuration ------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "4.0"

# The document name of the “root” document, that is, the document that contains
# the root toctree directive.
root_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinxcontrib.bibtex",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "sphinx_issues",
]

source_suffix = ".rst"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# Sphinx will warn about all references where the target cannot be found.
nitpicky = True
nitpick_ignore = []

# The name of a reST role (builtin or Sphinx extension) to use as the default
# role, that is, for text marked up `like this`. This can be set to 'py:obj' to
# make `filter` a cross-reference to the Python function “filter”.
default_role = "py:obj"

# -- options for HTML output -------------------------------------------------

# HTML options (e.g., theme)
# see: https://sphinx-bootstrap-theme.readthedocs.io/en/latest/README.html
# Clean up sidebar: Do not show "Source" link
html_show_sourcelink = False
html_copy_source = False

html_theme = "pydata_sphinx_theme"

# Add any paths that contain templates here, relative to this directory.
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    "icon_links": [
        dict(
            name="GitHub",
            url=gh_url,
            icon="fab fa-github-square",
        ),
    ],
    "use_edit_page_button": False,
    "navigation_with_keys": False,
    "show_toc_level": 1,
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
}
# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    "index": ["search-field.html"],
}

html_context = {
    "versions_dropdown": {
        "dev": "v0.4 (devel)",
        "stable": "v0.3",
        "v0.2": "v0.2",
        "v0.1": "v0.1",
    },
}

# variables to pass to HTML templating engine
html_context = {
    'pygment_light_style': 'tango',
    'pygment_dark_style': 'native',
}

# -- autosummary -------------------------------------------------------------
autosummary_generate = True

# -- autodoc -----------------------------------------------------------------
autoclass_content = "class"
autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_warningiserror = True
autodoc_default_options = {"inherited-members": None}

# -- numpydoc ----------------------------------------------------------------

# needed to prevent errors
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True

# x-ref
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    # Python
    "Path": "pathlib.Path",
    "bool": ":class:`python:bool`",
    # MNE
    "Epochs": "mne.Epochs",
    "ICA": "mne.preprocessing.ICA",
    "Info": "mne.Info",
    "Raw": "mne.io.Raw",
}
numpydoc_xref_ignore = {
    "of",
    "shape",
    "n_components",
    "n_pixels",
    "n_classes",
    "instance",
    "ICAComponentLabeler",
}

# validation
# https://numpydoc.readthedocs.io/en/latest/validation.html#validation-checks
error_ignores = {
    "GL01",  # docstring should start in the line immediately after the quotes
    "EX01",  # section 'Examples' not found
    "ES01",  # no extended summary found
    "SA01",  # section 'See Also' not found
    "RT02",  # The first line of the Returns section should contain only the type, unless multiple values are being returned  # noqa
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

# -- sphinx-copybutton -------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- intersphinx -------------------------------------------------------------
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

# -- sphinx-gallery ----------------------------------------------------------
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

# -- sphinxcontrib-bibtex ----------------------------------------------------
bibtex_bibfiles = ["./references.bib"]
bibtex_style = "unsrt"
bibtex_footbibliography_header = ""

# -- Sphinx-issues -----------------------------------------------------------
issues_github_path = "mne-tools/mne-icalabel"

# -- sphinx.ext.linkcode -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html

def linkcode_resolve(domain: str, info: Dict[str, str]) -> Optional[str]:
    """Determine the URL corresponding to a Python object.

    Parameters
    ----------
    domain : str
        One of 'py', 'c', 'cpp', 'javascript'.
    info : dict
        With keys "module" and "fullname".

    Returns
    -------
    url : str | None
        The code URL. If None, no link is added.
    """
    if domain != "py":
        return None  # only document python objects

    # retrieve pyobject and file
    try:
        module = import_module(info["module"])
        pyobject = module
        for elt in info["fullname"].split("."):
            pyobject = getattr(pyobject, elt)
        fname = inspect.getsourcefile(pyobject).replace("\\", "/")
    except Exception:
        # Either the object could not be loaded or the file was not found.
        # For instance, properties will raise.
        return None

    # retrieve start/stop lines
    source, start_line = inspect.getsourcelines(pyobject)
    lines = "L%d-L%d" % (start_line, start_line + len(source) - 1)

    # create URL
    if "dev" in release:
        branch = "main"
    else:
        return None  # alternatively, link to a maint/version branch
    fname = fname.split("/mne_icalabel/")[1]
    url = f"{gh_url}/blob/{branch}/mne_icalabel/{fname}#{lines}"
    return url
