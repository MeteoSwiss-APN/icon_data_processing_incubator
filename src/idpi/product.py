"""Product base classes."""

# Standard library
from abc import ABC
from abc import abstractmethod
from itertools import accumulate

# Third-party
from dask import delayed


class Register:
    """Register methods that are (dask) delayed for caching."""

    def __init__(self):
        self.regdict = {}

    def reg(self, fn, *arg):
        key = list(
            accumulate([hash(id(x)) for x in arg], lambda x, acc: hash(x ^ acc))
        )[-1]

        return self.regdict.setdefault(key, delayed(fn)(*arg))


class Product(ABC):
    """Abstract base class for products."""

    def __init__(self, reg: Register | None = None):
        if not reg:
            self.reg = Register()
        else:
            self.reg = reg

    @property
    @abstractmethod
    def input_fields(self):
        ...
