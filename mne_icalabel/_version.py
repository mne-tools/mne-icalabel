"""Version number."""

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

__version__ = version(__package__)
