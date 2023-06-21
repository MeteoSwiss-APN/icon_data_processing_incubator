"""Decoder for grib data."""
# Standard library
import os
from contextlib import contextmanager
from pathlib import Path

# Third-party
import earthkit.data  # type: ignore
import numpy as np
import xarray as xr
import cfgrib

def _is_ensemble(field) -> bool:
    try:
        return field.metadata("typeOfEnsembleForecast") == 192
    except KeyError:
        return False


def _gather_coords(field_map, dims):
    coords = zip(*field_map)
    cc = (sorted(set(coord)) for coord in coords)
    ny, nx = next(iter(field_map.values())).shape
    return {dim: c for dim, c in zip(dims, (*cc, np.arange(ny), np.arange(nx)))}



def load_data(
    params: list[str], datafiles: list[Path], ref_param: str
) -> dict[str, xr.DataArray]:
    """Load data from GRIB files.

    Parameters
    ----------
    params : list[str]
        List of fields to load from the data files.
    datafiles : list[Path]
        List of files from which to load the data.
    ref_param : str
        Parameter to use as a reference for the coordinates.

    Raises
    ------
    ValueError
        if ref_param is not included in params.
    RuntimeError
        if not all fields are found in the given datafiles.

    Returns
    -------
    dict[str, xr.DataArray]
        Mapping of fields by param name

    """
    fs = earthkit.data.from_source("file", [str(p) for p in datafiles])

    if ref_param not in params:
        raise ValueError(f"{ref_param} must be in {params}")

    hcoords = None
    metadata = {}
    dims: dict[str, tuple[str, ...]] = {}
    data: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
    for field in fs.sel(param=params):
        param = field.metadata("param")
        field_map = data.setdefault(param, {})
        dim_keys = (
            ("perturbationNumber", "step", "level")
            if _is_ensemble(field)
            else ("step", "level")
        )
        field_map[field.metadata(*dim_keys)] = field.to_numpy(dtype=np.float32)

        if param not in dims:
            dims[param] = dim_keys[:-1] + (field.metadata("typeOfLevel"), "y", "x")

        if param not in metadata:
            metadata[param] = field.metadata(
                namespace=["ls", "geography", "parameter", "time"]
            )
            if field.metadata('PVPresent') != 0:
                metadata[param].update(
                    {'NV':field.metadata('NV'),
                    'pv':field.metadata('pv')
                    })

        if hcoords is None and param == ref_param:
            hcoords = {
                dim: (("y", "x"), values) for dim, values in field.to_points().items()
            }

    if not set(params) == data.keys():
        raise RuntimeError(f"Missing params: {set(params) - data.keys()}")

    coords = {
        param: _gather_coords(field_map, dims[param])
        for param, field_map in data.items()
    }

    return {
        param: xr.DataArray(
            np.array([field_map.pop(key) for key in sorted(field_map)]).reshape(
                tuple(len(c) for c in coords[param].values())
            ),
            coords=coords[param] | hcoords,
            dims=list(coords[param].keys()),
            attrs=metadata[param],
        )
        for param, field_map in data.items()
    }
