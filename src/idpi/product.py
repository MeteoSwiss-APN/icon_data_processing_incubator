"""Product base classes."""

# Standard library
from abc import ABCMeta
from abc import abstractmethod
from itertools import accumulate

# Third-party
import dask


class Register:
    """Register methods that are (dask) delayed for caching."""

    def __init__(self, delay: bool = False):
        self.regdict = {}
        self._delayed = dask.delayed if delay else lambda x: x

    def reg(self, fn, *arg):
        key = list(
            accumulate([hash(id(x)) for x in arg], lambda x, acc: hash(x ^ acc))
        )[-1]

        return self.regdict.setdefault(key, self._delayed(fn)(*arg))


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self,
        input_fields: list[str],
        reg: Register | None = None,
        delay_entire_product: bool = False,
    ):
        self._input_fields = input_fields
        if not reg:
            self.reg = Register()
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
