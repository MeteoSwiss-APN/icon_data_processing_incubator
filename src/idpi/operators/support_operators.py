"""Algorithms to support operations on a field."""


# Standard library
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional

# Third-party
import numpy as np
import xarray as xr


def init_field_with_vcoord(
    parent: xr.DataArray,
    tc_values: np.ndarray,
    fill_value: Any,
    dtype: Optional[np.dtype] = None,
) -> xr.DataArray:
    """Initialize an xarray.DataArray with new vertical coordinates.

    Properties except for those related to the vertical coordinates,
    and optionally dtype, are inherited from the parent xarray.DataArray.

    Args:
        parent (xr.DataArray): parent field
        vcoord (dict[str, Any]): dictionary specifying new vertical coordinates;
            expected keys: "typeOfLevel" (string), "values" (list),
                "NV" (int), "attrs" (dict)
        fill_value (Any): value the data array of the new field
            is initialized with
        dtype (np.dtype, optional): fill value data type; defaults to None (in this case
            the data type is inherited from the parent field). Defaults to None.

    Raises:
        KeyError: _description_

    Returns:
        xr.DataArray: new field

    """
    # dims
    shape = list(len(parent[d]) if d != "z" else len(tc_values) for d in parent.dims)
    dims = parent.dims
    # coords
    coords = {"z": tc_values}

    # attrs
    attrs = dict()
    attrs["vcoord_type"] = "pressure"
    # TODO does this algorithm works also for staggered U,V?
    attrs["origin"] = parent.origin

    # dtype
    if dtype is None:
        dtype = parent.data.dtype

    return xr.DataArray(
        name=parent.name,
        data=np.full(tuple(shape), fill_value, dtype),
        dims=tuple(dims),
        coords=coords,
        attrs=attrs,
    )


def get_rotated_latitude(field: xr.DataArray) -> xr.DataArray:
    ny = field.attrs["GRIB_Ny"]
    lat_min = field.attrs["GRIB_latitudeOfFirstGridPointInDegrees"]
    dlat = field.attrs["GRIB_jDirectionIncrementInDegrees"]
    rlat = xr.DataArray(np.arange(ny, dtype=np.float32) * dlat + lat_min, dims="y")
    return rlat * np.pi / 180
