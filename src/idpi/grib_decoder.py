"""Decoder for grib data."""
# Standard library
import os
from contextlib import contextmanager
from pathlib import Path

# Third-party
import earthkit.data  # type: ignore
import eccodes  # type: ignore
import numpy as np
import xarray as xr


@contextmanager
def cosmo_grib_defs():
    """Enable COSMO GRIB definitions."""
    prefix = os.environ["CONDA_PREFIX"]
    root_dir = Path(prefix) / "share"
    paths = (
        root_dir / "eccodes-cosmo-resources/definitions",
        root_dir / "eccodes/definitions",
    )
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"{path} does not exist")
    defs_path = ":".join(map(str, paths))
    restore = eccodes.codes_definition_path()
    eccodes.codes_set_definitions_path(defs_path)
    try:
        yield
    finally:
        eccodes.codes_set_definitions_path(restore)


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
    level_types = {}
    data: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for field in fs.sel(param=params):
        param = field.metadata("param")
        field_map = data.setdefault(param, {})
        field_map[field.metadata("step", "level")] = field.to_numpy(dtype=np.float32)

        if param not in level_types:
            level_types[param] = field.metadata("typeOfLevel")

        if param not in metadata:
            metadata[param] = field.metadata(
                namespace=["ls", "geography", "parameter", "time"]
            )

        if hcoords is None and param == ref_param:
            hcoords = {
                dim: (("y", "x"), values) for dim, values in field.to_points().items()
            }

    if not set(params) == data.keys():
        raise RuntimeError(f"Missing params: {set(params) - data.keys()}")

    shapes = {}
    for param, field_map in data.items():
        steps, levels = zip(*field_map)
        n_steps = len(set(steps))
        n_levels = len(set(levels))
        ny, nx = next(iter(field_map.values())).shape
        shapes[param] = n_steps, n_levels, ny, nx

    return {
        param: xr.DataArray(
            np.array([field_map.pop(key) for key in sorted(field_map)]).reshape(
                shapes[param]
            ),
            coords=hcoords,
            dims=["step", level_types[param], "y", "x"],
            attrs=metadata[param],
        )
        for param, field_map in data.items()
    }


def load_cosmo_data(
    params: list[str], datafiles: list[Path], ref_param: str = "HHL"
) -> dict[str, xr.DataArray]:
    """Load data from GRIB files.

    The COSMO definitions are enabled during the load.

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
    with cosmo_grib_defs():
        return load_data(params, datafiles, ref_param)
