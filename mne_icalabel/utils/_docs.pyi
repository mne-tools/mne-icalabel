from collections.abc import Callable
from typing import Any

docdict: dict[str, str]
_KEYS_MNE: tuple[str, ...]
entry: str
docdict_indented: dict[int, dict[str, str]]

def fill_doc(f: Callable[..., Any]) -> Callable[..., Any]:
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of (modified in place).

    Returns
    -------
    f : callable
        The function, potentially with an updated __doc__.
    """

def _indentcount_lines(lines: list[str]) -> int:
    """Minimum indent for all lines in line list.

    >>> lines = [" one", "  two", "   three"]
    >>> indentcount_lines(lines)
    1
    >>> lines = []
    >>> indentcount_lines(lines)
    0
    >>> lines = [" one"]
    >>> indentcount_lines(lines)
    1
    >>> indentcount_lines(["    "])
    0
    """

def copy_doc(source: Callable[..., Any]) -> Callable[..., Any]:
    """Copy the docstring from another function (decorator).

    The docstring of the source function is prepepended to the docstring of the function
    wrapped by this decorator.

    This is useful when inheriting from a class and overloading a method. This decorator
    can be used to copy the docstring of the original method.

    Parameters
    ----------
    source : callable
        The function to copy the docstring from.

    Returns
    -------
    wrapper : callable
        The decorated function.

    Examples
    --------
    >>> class A:
    ...     def m1():
    ...         '''Docstring for m1'''
    ...         pass
    >>> class B(A):
    ...     @copy_doc(A.m1)
    ...     def m1():
    ...         '''this gets appended'''
    ...         pass
    >>> print(B.m1.__doc__)
    Docstring for m1 this gets appended
    """
