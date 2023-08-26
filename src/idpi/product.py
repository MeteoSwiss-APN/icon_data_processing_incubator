"""Product base classes."""

# Standard library
from abc import ABCMeta, abstractmethod
from itertools import accumulate

# Third-party
import dask


class Register:
    """Register methods that are (dask) delayed for caching."""

    def __init__(self):
        self.regdict = {}

    def reg(self, fn, *arg):
        key = list(
            accumulate([hash(id(x)) for x in arg], lambda x, acc: hash(x ^ acc))
        )[-1]

        return self.regdict.setdefault(key, dask.delayed(fn)(*arg))


class Product(metaclass=ABCMeta):
    """Base class for products."""

    def __init__(
        self, input_fields: list[str], reg: Register | None = None, delay: bool = False
    ):
        self._input_fields = input_fields
        if not reg:
            self.reg = Register()
        else:
            self.reg = reg
        self._delay = delay

    @abstractmethod
    def _run(self, **args):
        pass

    def __call__(self, *args):
        if self._delay:
            return dask.delayed(self._run)(*args)
        return self._run(*args)

    @property
    def input_fields(self):
        return self._input_fields
