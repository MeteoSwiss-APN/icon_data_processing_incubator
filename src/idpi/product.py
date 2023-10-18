"""Product base classes."""
# Standard library
import dataclasses as dc
from abc import ABCMeta, abstractmethod
from typing import NamedTuple

# Local
from . import tasking


class Request(NamedTuple):
    param: str
    levtype: str | None = None


@dc.dataclass
class ProductDescriptor:
    input_fields: list[Request]


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self,
        input_fields: list[Request],
        delay_entire_product: bool = False,
    ):
        self._desc = ProductDescriptor(input_fields=input_fields)
        self._delay_entire_product = delay_entire_product

    @abstractmethod
    def _run(self, **args):
        pass

    def __call__(self, *args):
        if self._delay_entire_product:
            return tasking.delayed(self._run)(*args)
        else:
            return self._run(*args)

    @property
    def descriptor(self):
        return self._desc
