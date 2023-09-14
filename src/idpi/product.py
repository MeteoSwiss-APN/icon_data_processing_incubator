"""Product base classes."""

# Standard library
import dataclasses as dc
from abc import ABCMeta
from abc import abstractmethod
from functools import partial

# Third-party
import dask


@dc.dataclass
class ProductDescriptor:
    input_fields: list[dict]


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self,
        input_fields: list[dict],
        delay_entire_product: bool = False,
    ):
        self._desc = ProductDescriptor(input_fields=input_fields)
        self._base_delayed = (
            partial(dask.delayed, pure=True) if delay_entire_product else lambda x: x
        )

    @abstractmethod
    def _run(self, **args):
        pass

    def __call__(self, *args):
        return self._base_delayed(self._run)(*args)

    @property
    def descriptor(self):
        return self._desc
