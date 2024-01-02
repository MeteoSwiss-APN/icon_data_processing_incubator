"""Horizontal cropping operator."""

# Third-party
import xarray as xr

# Local
from .. import metadata
from . import gis


def crop(field: xr.DataArray, bounds: tuple[int, ...]) -> xr.DataArray:
    """Crop the field to the given bounds.

    Only fields defined on regular grids in rotlatlon coordinates,
    without rotation nor flipped axes are supported.

    Parameters
    ----------
    field : xarray.DataArray
        The field to crop.
    bounds: tuple[int, ...]
        Bounds of the cropped area, xmin, xmax, ymin, ymax. All bounds are inclusive.

    Raises
    ------
    ValueError
        If there are any consistency issues with the provided bounds
        or any of the conditions on the input grid not met.

    Returns
    -------
    xarray.DataArray
        The data is set to cropped domain and the metadata is updated accordingly.

    """
    xmin, xmax, ymin, ymax = bounds

    sizes = field.sizes
    if (
        xmin > xmax
        or ymin > ymax
        or any(v < 0 for v in bounds)
        or xmax >= sizes["x"]
        or ymax >= sizes["y"]
    ):
        raise ValueError(f"Inconsistent bounds: {bounds}")

    grid = gis.get_grid(field.geography)
    lon_min = grid.rlon.isel(x=xmin).item()
    lat_min = grid.rlat.isel(y=ymin).item()
    ni = xmax - xmin + 1
    nj = ymax - ymin + 1

    return xr.DataArray(
        field.isel(x=slice(xmin, xmax + 1), y=slice(ymin, ymax + 1)),
        attrs=metadata.override(
            field.message,
            longitudeOfFirstGridPointInDegrees=lon_min,
            Ni=ni,
            latitudeOfFirstGridPointInDegrees=lat_min,
            Nj=nj,
        ),
    )
