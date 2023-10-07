"""functionality for tasking and parallel computing."""
# Third-party
import dask

# First-party
import idpi.config


def delayed(fn):
    return dask.delayed(fn) if idpi.config.get("enable_dask", False) else fn


def compute(*x):
    return dask.compute(*x) if idpi.config.get("enable_dask", False) else x
