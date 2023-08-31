"""Product base classes."""

# Standard library
import typing
from abc import ABCMeta
from abc import abstractmethod
from itertools import accumulate

# Third-party
import dask


class OperatorRegistry:
    """Registry for operators that can be used to cache operations in dask."""

    def __init__(self, delay: bool = False):
        self.regdict: dict[str, typing.Any] = {}
        self._delayed = dask.delayed if delay else lambda x: x

    def reg(self, fn, *arg):
        key = frozenset([id(x) for x in arg])

        return self.regdict.setdefault(key, self._delayed(fn)(*arg))


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self,
        input_fields: list[str],
        reg: OperatorRegistry | None = None,
        delay_entire_product: bool = False,
    ):
        self._input_fields = input_fields
        if not reg:
            self.reg = OperatorRegistry()
        else:
            self.reg = reg
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
    def input_fields(self):
        return self._input_fields
