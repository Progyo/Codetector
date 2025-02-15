# Generic: https://mypy.readthedocs.io/en/stable/generics.html

from typing import TypeVar, Generic
from abc import ABC, abstractmethod

T = TypeVar('T')
P = TypeVar('P')

class UsecaseWithParameters(ABC,Generic[T,P]):
    """
    Interface for use cases that have parameters.
    """

    @abstractmethod
    def __call__(self, params:T) -> P:
        pass


class UsecaseWithoutParameters(ABC,Generic[T]):
    """
    Interface for use cases that don't have parameters.
    """

    @abstractmethod
    def __call__(self) -> T:
        pass