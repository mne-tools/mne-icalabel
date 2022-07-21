"""Test _docs.py"""

import pytest

from mne_icalabel.utils._docs import copy_doc, fill_doc


def test_fill_doc():
    """Test decorator to fill docstring."""
    # test filling docstring
    @fill_doc
    def foo(verbose):
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """
        pass

    assert "verbose : bool | str | int | None" in foo.__doc__

    # test filling docstring with invalid key
    with pytest.raises(RuntimeError, match="Error documenting"):

        @fill_doc
        def foo(verbose):
            """My doc.

            Parameters
            ----------
            %(invalid_key)s
            """
            pass


def test_copy_doc():
    """Test decorator to copy docstring."""
    # test copy of docstring
    def foo(x, y):
        """
        My doc.
        """
        pass

    @copy_doc(foo)
    def foo2(x, y):
        pass

    assert "My doc." in foo2.__doc__

    # test copy of docstring from a function without docstring
    def foo(x, y):
        pass

    with pytest.raises(RuntimeError, match="The docstring from foo could not"):

        @copy_doc(foo)
        def foo2(x, y):
            pass
