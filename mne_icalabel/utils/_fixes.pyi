from typing import Any

class WrapStdOut:
    """Dynamically wrap to sys.stdout.

    This makes packages that monkey-patch sys.stdout (e.g.doctest, sphinx-gallery) work
    properly.
    """

    def __getattr__(self, name: str) -> Any:
        ...