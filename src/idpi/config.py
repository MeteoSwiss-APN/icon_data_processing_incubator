"""global configuration for idpi."""

# Standard library
from typing import Any
from typing import Literal

config: dict = {}


class set:
    """Temporarily set configuration values within a context manager.

    Parameters
    ----------
    config : dict, optional
        dictionary object use to hold the configuration.
        Default will use the config object in this module
    **kwargs :
        the configuration key-value pairs to set.

    """

    def __init__(self, config: dict = config, **kwargs):
        self._config = config
        self._record: list[tuple[Literal["insert", "replace"], str, Any]] = []

        if kwargs:
            for key, value in kwargs.items():
                if key in self._config:
                    self._record.append(("replace", key, self._config[key]))
                else:
                    self._record.append(("insert", key, None))

                self._config[key] = value

    def __enter__(self) -> dict:
        return self._config

    def __exit__(self, type, value, traceback):
        for op, key, value in reversed(self._record):
            d = self._config
            if op == "replace":
                d[key] = value
            else:  # insert
                d.pop(key, None)


def get(
    key: str,
    default: Any = None,
    config: dict = config,
) -> Any:
    """Get values from global config.

    Parameters
    ----------
    key: str
        specifies the name of the key for which the value is requested
    default: Any
        default value to be returned in case the key does not exist in config
    config: dict, optional
        config object holding the mapping. Default value is the global config

    """
    return config.get(key, default)
