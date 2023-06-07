"""Algorithms to support operations on a field."""


# Standard library
import dataclasses as dc
from typing import Any
from typing import Optional
from typing import Literal
from typing import Sequence

# Third-party
import numpy as np
import xarray as xr


@dc.dataclass
class TargetCoordinatesAttrs:
    """Attributes to the target coordinates."""

    standard_name: str
    long_name: str
    units: str
    positive: Literal["up", "down"]


@dc.dataclass
class TargetCoordinates:
    """Target Coordinates."""

    type_of_level: str
    values: Sequence[float]
    attrs: TargetCoordinatesAttrs

    @property
    def size(self):
        return len(self.values)


def init_field_with_vcoord(
    parent: xr.DataArray,
    vcoord: TargetCoordinates,
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
    # TODO: test that vertical dim of parent is named "generalVerticalLayer"
    # or take vertical dim to replace as argument
    #       be aware that vcoord contains also xr.DataArray GRIB attributes;
    #  one should separate these from coordinate properties
    #       in the interface
    # attrs
    attrs = parent.attrs.copy()
    attrs["GRIB_typeOfLevel"] = vcoord.type_of_level
    if "GRIB_NV" in attrs:
        attrs["GRIB_NV"] = 0
    # dims
    sizes = {dim: size for dim, size in parent.sizes.items() if str(dim) in "xy"}
    sizes[vcoord.type_of_level] = vcoord.size
    # coords
    # ... inherit all except for the vertical coordinates
    coords = {c: v for c, v in parent.coords.items() if c != "generalVerticalLayer"}
    # ... initialize the vertical target coordinates
    coords[vcoord.type_of_level] = xr.IndexVariable(
        vcoord.type_of_level, vcoord.values, attrs=dc.asdict(vcoord.attrs)
    )
    # dtype
    if dtype is None:
        dtype = parent.data.dtype

    return xr.DataArray(
        name=parent.name,
        data=np.full(tuple(sizes.values()), fill_value, dtype),
        dims=tuple(sizes.keys()),
        coords=coords,
        attrs=attrs,
    )


def get_rotated_latitude(field: xr.DataArray) -> xr.DataArray:
    ny = field.attrs["GRIB_Ny"]
    lat_min = field.attrs["GRIB_latitudeOfFirstGridPointInDegrees"]
    dlat = field.attrs["GRIB_jDirectionIncrementInDegrees"]
    rlat = xr.DataArray(np.arange(ny, dtype=np.float32) * dlat + lat_min, dims="y")
    return rlat * np.pi / 180
