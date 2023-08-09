"""Various time operators."""
# Third-party
import numpy as np
import xarray as xr


def time_rate(var: xr.DataArray, dtime: np.timedelta64):
    """Compute a time rate for a given delta in time.

    It assumes the input data is an accumulated value
    between two time steps of the time coordinate

    Args:
        var: variable that contains the input data
        dtime: delta time of the desired output time rate

    """
    coord = var.valid_time
    result = var.diff(dim="time") / (coord.diff(dim="time") / dtime)
    result.attrs = var.attrs
    return result


def _nsteps(valid_time: xr.DataArray, dtime: np.timedelta64) -> int:
    dt = valid_time.diff(dim="time")
    uniform = np.all(dt == dt[0]).item()

    if not uniform:
        msg = "Given field has an irregular time step."
        raise ValueError(msg)

    condition = valid_time - valid_time[0] == dtime
    try:
        [index] = np.nonzero(condition.values)
    except ValueError:
        msg = "Provided dtime is not a multiple of the time step."
        raise ValueError(msg)

    return index.item()


def delta(field: xr.DataArray, dtime: np.timedelta64) -> xr.DataArray:
    """Compute difference for a given delta in time.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    dtime : np.timedelta64
        Time delta for which to evaluate the difference.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The field difference for the given time delta.

    """
    nsteps = _nsteps(field.valid_time, dtime)
    result = field - field.shift(time=nsteps)
    result.attrs = field.attrs
    return result


def min(field: xr.DataArray, dtime: np.timedelta64) -> xr.DataArray:
    """Compute minimum aggregate for a given delta in time.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    dtime : np.timedelta64
        Time delta for which to evaluate the minimum.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The field minimum for the given time delta.

    """
    nsteps = _nsteps(field.valid_time, dtime)
    return field.rolling(time=nsteps).min()


def max(field: xr.DataArray, dtime: np.timedelta64) -> xr.DataArray:
    """Compute maximum aggregate for a given delta in time.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    dtime : np.timedelta64
        Time delta for which to evaluate the maximum.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The field maximum for the given time delta.

    """
    nsteps = _nsteps(field.valid_time, dtime)
    return field.rolling(time=nsteps).max()


def avg(field: xr.DataArray, dtime: np.timedelta64) -> xr.DataArray:
    """Compute average aggregate for a given delta in time.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    dtime : np.timedelta64
        Time delta for which to evaluate the average.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The field average for the given time delta.

    """
    nsteps = _nsteps(field.valid_time, dtime)
    return field.rolling(time=nsteps).mean()


def sum(field: xr.DataArray, dtime: np.timedelta64) -> xr.DataArray:
    """Compute sum aggregate for a given delta in time.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    dtime : np.timedelta64
        Time delta for which to evaluate the sum.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The field sum for the given time delta.

    """
    nsteps = _nsteps(field.valid_time, dtime)
    return field.rolling(time=nsteps).sum()


def resample(field: xr.DataArray, period: np.timedelta64) -> xr.DataArray:
    """Resample field.

    The period must be a multiple of the current time step.
    No interpolation is performed.

    Parameters
    ----------
    field : xr.DataArray
        Field that contains the input data.
    period : np.timedelta64
        Output sample period.

    Raises
    ------
    ValueError
        if dtime is not multiple of the field time step
        or if the time step is not regular.

    Returns
    -------
    xr.DataArray
        The resampled field.

    """
    nsteps = _nsteps(field.valid_time, period)
    return field.sel(time=slice(None, None, nsteps))
