"""algorithm for destaggering a field."""
# Standard library
from typing import Literal

# Third-party
import numpy as np
import xarray as xr

from idpi.iarray import Iarray

ExtendArg = Literal["left", "right", "both"] | None


# def _intrp_mid(a: np.ndarray) -> np.ndarray:
#     return 0.5 * (a[..., :-1] + a[..., 1:])


# def _left(a: np.ndarray) -> np.ndarray:
#     t = a.copy()
#     t[..., 1:] = _intrp_mid(a)
#     return t


# def _right(a: np.ndarray) -> np.ndarray:
#     t = a.copy()
#     t[..., :-1] = _intrp_mid(a)
#     return t


# def _both(a: np.ndarray) -> np.ndarray:
#     *m, n = a.shape
#     t = np.empty((*m, n + 1))
#     t[..., 0] = a[..., 0]
#     t[..., -1] = a[..., -1]
#     t[..., 1:-1] = _intrp_mid(a)
#     return t


def _intrp_mid(a: Iarray, dim: str) -> Iarray:
    return 0.5 * (a.isel({dim: slice(0, -1)}) + a.isel({dim: slice(1, None)}))


def _left(a: Iarray, dim: str) -> Iarray:
    t = a.copy()
    t[{dim: slice(1, None)}] = _intrp_mid(a, dim)
    return t


def _right(a: Iarray, dim: str) -> Iarray:
    t = a.copy()
    t[{dim: slice(0, -1)}] = _intrp_mid(a, dim)
    return t


def _both(a: Iarray, dim: str) -> Iarray:
    index_dim = a.dims(dim)
    shape = [x if idx != index_dim else x + 1 for idx, x in enumerate(a.shape)]
    dim_index = a.dims.index(dim)
    shape[dim_index] = shape[dim_index] + 1
    t = Iarray(dims=a.dims, coords=a.coords, data=np.empty(shape))
    t[{dim: 0}] = a.isel({dim: 0})
    t[{dim: 0}] = a.isel({dim: 0})

    t[..., -1] = a[..., -1]
    t[..., 1:-1] = _intrp_mid(a, dim)
    return t


def interpolate_midpoint(array: Iarray, dim: str, extend: ExtendArg = None) -> Iarray:
    """Interpolate field values onto the midpoints.

    The interpolation is only done on the last dimension of the given array.
    The first or last values can optionally duplicated as per the extend argument.

    Parameters
    ----------
    array : np.ndarray
        Array of field values
    extend : None | Literal["left", "right", "both"]
        Optionally duplicate values on the left, right or both sides.
        Defaults to None.

    Raises
    ------
    ValueError
        If the extend argument is not recognised.

    Returns
    -------
    np.ndarray
        Values of the field interpolated to the midpoint on the last dimension.

    """
    f_map = {
        None: _intrp_mid,
        "left": _left,
        "right": _right,
        "both": _both,
    }
    if extend not in f_map:
        raise ValueError(f"extend arg not in {tuple(f_map.keys())}")
    return f_map[extend](array, dim)


def destagger(
    field: Iarray,
    dim: Literal["x", "y", "z"],
) -> Iarray:
    """Destagger a field.

    Note that, in the x and y directions, it is assumed that one element
    of the destaggered field is missing on the left side of the domain.
    The first element is thus duplicated to fill the blank.

    Parameters
    ----------
    field : Iarray
        Field to destagger
    dim : Literal["x", "y", "z"]
        Dimension along which to destagger

    Raises
    ------
    ValueError
        Raises ValueError if dim argument is not one of
        {"x","y","z"}.

    Returns
    -------
    Iarray
        destaggered field with dimensions in
        {"x","y","z"}

    """
    if dim == "x" or dim == "y":
        return interpolate_midpoint(field, dim=dim, extend="left")
    elif dim == "z":
        return interpolate_midpoint(field, dim=dim, extend=None)

    raise ValueError("Unknown dimension", dim)
