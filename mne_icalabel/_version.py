"""Version number."""

# TODO: Remove try/except once the minimum python requirement is bumped to 3.8
try:
    from importlib.metadata import version  # type: ignore
except ImportError:
    from importlib_metadata import version  # type: ignore

__version__ = version(__package__)
