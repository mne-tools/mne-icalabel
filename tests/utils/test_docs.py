import pytest
from mne.utils import verbose

from mne_icalabel.utils._docs import copy_doc


def test_copy_doc_function():
    """Test decorator to copy docstring on functions."""

    # test copy of docstring
    def foo(x, y):
        """My doc."""

    @copy_doc(foo)
    def foo2(x, y):
        pass

    @copy_doc(foo)
    def foo3(x, y):
        """Doc of foo3."""

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
