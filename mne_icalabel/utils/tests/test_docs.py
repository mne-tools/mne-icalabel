"""Test _docs.py"""

import re
from pathlib import Path

import pytest
from mne.utils import verbose

from mne_icalabel.utils._docs import _KEYS_MNE, copy_doc, docdict, fill_doc


def test_fill_doc_function():
    """Test decorator to fill docstring on functions."""

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

    # test filling empty-docstring
    @fill_doc
    def foo():
        pass

    assert foo.__doc__ is None

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

    # test filling docstring with verbose
    @fill_doc
    @verbose
    def foo(verbose=None):
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """
        pass

    assert "verbose : bool | str | int | None" in foo.__doc__


def test_fill_doc_class():
    """Test decorator to fill docstring on classes."""

    @fill_doc
    class Foo:
        """My doc.

        Parameters
        ----------
        %(verbose)s
        """

        def __init__(self, verbose=None):
            pass

        @fill_doc
        def method(self, verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """
            pass

        @fill_doc
        @verbose
        def method_decorated(self, verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """
            pass

        @staticmethod
        @fill_doc
        @verbose
        def method_decorated_static(verbose=None):
            """My method doc.

            Parameters
            ----------
            %(verbose)s
            """
            pass

    assert "verbose : bool | str | int | None" in Foo.__doc__
    assert "verbose : bool | str | int | None" in Foo.method.__doc__
    assert "verbose : bool | str | int | None" in Foo.method_decorated.__doc__
    assert "verbose : bool" in Foo.method_decorated_static.__doc__
    foo = Foo()
    assert "verbose : bool | str | int | None" in foo.__doc__
    assert "verbose : bool | str | int | None" in foo.method.__doc__
    assert "verbose : bool | str | int | None" in foo.method_decorated.__doc__


def test_copy_doc_function():
    """Test decorator to copy docstring on functions."""

    # test copy of docstring
    def foo(x, y):
        """My doc."""
        pass

    @copy_doc(foo)
    def foo2(x, y):
        pass

    @copy_doc(foo)
    def foo3(x, y):
        """Doc of foo3."""
        pass

    assert foo.__doc__ == foo2.__doc__
    assert foo.__doc__ + "Doc of foo3." == foo3.__doc__

    # test copy of docstring from a function without docstring
    def foo(x, y):
        pass

    with pytest.raises(RuntimeError, match="The docstring from foo could not"):

        @copy_doc(foo)
        def foo2(x, y):
            pass

    # test copy docstring of decorated function
    @verbose
    def foo(verbose=None):
        """My doc."""
        pass

    @copy_doc(foo)
    def foo2(verbose=None):
        pass

    @copy_doc(foo)
    @verbose
    def foo3(verbose=None):
        pass

    assert foo.__doc__ == foo2.__doc__
    assert foo.__doc__ == foo3.__doc__


def test_copy_doc_class():
    """Test decorator to copy docstring on classes."""

    class Foo:
        """My doc."""

        def __init__(self):
            pass

        def method1(self):
            """Super 101 doc."""
            pass

    @copy_doc(Foo)
    class Foo2:
        def __init__(self):
            pass

        @copy_doc(Foo.method1)
        def method2(self):
            pass

        @copy_doc(Foo.method1)
        @verbose
        def method3(self, verbose=None):
            pass

        @staticmethod
        @copy_doc(Foo.method1)
        @verbose
        def method4(verbose=None):
            pass

    assert Foo.__doc__ == Foo2.__doc__
    assert Foo.method1.__doc__ == Foo2.method2.__doc__
    assert Foo.method1.__doc__ == Foo2.method3.__doc__
    assert Foo.method1.__doc__ == Foo2.method4.__doc__
    foo = Foo()
    foo2 = Foo2()
    assert foo.__doc__ == foo2.__doc__
    assert foo.method1.__doc__ == foo2.method2.__doc__
    assert foo.method1.__doc__ == foo2.method3.__doc__
    assert foo.method1.__doc__ == foo2.method4.__doc__


def test_docdict_order():
    """Test that docdict is alphabetical."""
    # read the file as text, and get entries via regex
    docdict_ = docdict.copy()
    for key in _KEYS_MNE:
        del docdict_[key]
    docs_path = Path(__file__).parents[1] / "_docs.py"
    assert docs_path.is_file()
    with open(docs_path, encoding="UTF-8") as fid:
        docs = fid.read()
    entries = re.findall(r'docdict\[(?:\n    )?["\'](.+)["\']\n?\] = ', docs)
    # test length, uniqueness and order
    assert len(docdict_) == len(entries)
    assert sorted(entries) == entries
