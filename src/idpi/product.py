"""Product base classes."""

# Standard library
from abc import ABCMeta
from abc import abstractmethod
import dataclasses as dc

# Third-party
import dask


@dc.dataclass
class ProductDescriptor:
    input_fields: list[str]


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self,
        input_fields: list[str],
        delay_entire_product: bool = False,
    ):
        self._desc = ProductDescriptor(input_fields=input_fields)
        self._base_delayed = dask.delayed if delay_entire_product else lambda x: x

    # avoid a possible override from inheriting classes
    @property
    def delay_entire_product(self):
        return self._base_delayed

    @abstractmethod
    def _run(self, **args):
        pass

    def __call__(self, *args):
        return self._base_delayed(self._run)(*args)

    @property
    def descriptor(self):
        return self._desc
