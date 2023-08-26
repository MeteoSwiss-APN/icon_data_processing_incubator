"""Decoder for grib data."""
# Standard library
import datetime as dt
import sys
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
import dataclasses as dc
import dask

# Third-party
import earthkit.data  # type: ignore
import eccodes  # type: ignore
import numpy as np
import xarray as xr
import yaml

from idpi.product import Product

DIM_MAP = {
    "level": "z",
    "perturbationNumber": "eps",
    "step": "time",
}
VCOORD_TYPE = {
    "generalVertical": ("model_level", -0.5),
    "generalVerticalLayer": ("model_level", 0.0),
    "hybrid": ("hybrid", 0.0),
    "isobaricInPa": ("pressure", 0.0),
    "surface": ("surface", 0.0),
}
_ifs_allowed = True
_cosmo_allowed = True


@contextmanager
def cosmo_grib_defs():
    """Enable COSMO GRIB definitions."""
    prefix = sys.exec_prefix
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


def _is_ensemble(field) -> bool:
    try:
        return field.metadata("typeOfEnsembleForecast") == 192
    except KeyError:
        return False


def _gather_coords(field_map, dims):
    coord_values = zip(*field_map)
    unique = (sorted(set(values)) for values in coord_values)
    coords = {dim: c for dim, c in zip(dims[:-2], unique)}
    ny, nx = next(iter(field_map.values())).shape
    shape = tuple(len(v) for v in coords.values()) + (ny, nx)
    return coords, shape


def _parse_datetime(date, time):
    return dt.datetime.strptime(f"{date}{time:04d}", "%Y%m%d%H%M")


def _gather_tcoords(time_meta):
    time = None
    valid_time = []
    for step in sorted(time_meta):
        tm = time_meta[step]
        valid_time.append(_parse_datetime(tm["validityDate"], tm["validityTime"]))
        if time is None:
            time = _parse_datetime(tm["dataDate"], tm["dataTime"])

    return {"valid_time": ("time", valid_time), "ref_time": time}


def _extract_pv(pv):
    if pv is None:
        return {}
    i = len(pv) // 2
    return {
        "ak": xr.DataArray(pv[:i], dims="z"),
        "bk": xr.DataArray(pv[i:], dims="z"),
    }


@dc.dataclass
class Grid:
    lon: xr.DataArray
    lat: xr.DataArray
    longitudeOfFirstGridPointInDegrees: float
    latitudeOfFirstGridPointInDegrees: float


def _check_string_arg(obj):
    return bool(obj) and all(isinstance(elem, str) for elem in obj)


class GribReader:
    def __init__(
        self,
        datafiles: list[Path],
        ref_param="HHL",
        ifs: bool = False,
        delay: bool = False,
    ):
        self._datafiles = datafiles
        self._ifs = ifs
        self._delayed = dask.delayed if delay else (lambda x: x)
        self._grid = self.load_grid_reference(ref_param)

    def load_grid_reference(self, ref_param: str) -> Grid:
        """Load data from GRIB files.

        Parameters
        ----------
        params : list[str]
            List of fields to load from the data files.
        datafiles : list[Path]
            List of files from which to load the data.
        ref_param : str
            Parameter to use as a reference for the coordinates.
        extract_pv: str | None
            Optionally extract hybrid level coefficients from the given field.

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

        if self._ifs:
            mapping_path = files("idpi.data").joinpath("field_mappings.yml")
            mapping = yaml.safe_load(mapping_path.open())
            ref_param = mapping[ref_param]["ifs"]["name"]

        fs = earthkit.data.from_source("file", [str(p) for p in self._datafiles])

        for field in fs.sel(param=ref_param):
            lonlat_dict = {
                geo_dim: xr.DataArray(dims=("y", "x"), data=values)
                for geo_dim, values in field.to_latlon().items()
            }

            grid = Grid(
                lonlat_dict["lon"],
                lonlat_dict["lat"],
                *field.metadata(
                    "longitudeOfFirstGridPointInDegrees",
                    "latitudeOfFirstGridPointInDegrees",
                ),
            )

            return grid

        raise ValueError(
            f"reference field, {ref_param=} not found in {self._datafiles=}"
        )

    def load_pv(self, pv_param: str):
        if not self._ifs:
            raise ValueError("load_pv only available for IFS data")
        fs = earthkit.data.from_source("file", [str(p) for p in self._datafiles]).sel(
            param=pv_param
        )

        for field in fs:
            return field.metadata("pv")

    def _load_param(
        self,
        param: str,
    ):
        fs = earthkit.data.from_source("file", [str(p) for p in self._datafiles]).sel(
            param=param
        )

        hcoords = None
        metadata = {}
        time_meta: dict[int, dict] = {}
        dims: tuple[str, ...] = None
        field_map: dict[tuple[int, ...], np.ndarray] = {}

        for field in fs:
            dim_keys = (
                ("perturbationNumber", "step", "level")
                if _is_ensemble(field)
                else ("step", "level")
            )
            key = field.metadata(*dim_keys)
            field_map[key] = field.to_numpy(dtype=np.float32)

            step = key[-2]  # assume all members share the same time steps
            if step not in time_meta:
                time_meta[step] = field.metadata(namespace="time")

            if not dims:
                dims = tuple(DIM_MAP[d] for d in dim_keys) + ("y", "x")

            if not metadata:
                metadata = field.metadata(namespace=["geography", "parameter"])
                level_type = field.metadata("typeOfLevel")
                vcoord_type, zshift = VCOORD_TYPE.get(level_type, (level_type, 0.0))

                x0 = self._grid.longitudeOfFirstGridPointInDegrees % 360
                y0 = self._grid.latitudeOfFirstGridPointInDegrees
                geo = metadata["geography"]
                dx = geo["iDirectionIncrementInDegrees"]
                dy = geo["jDirectionIncrementInDegrees"]

                metadata |= {
                    "vcoord_type": vcoord_type,
                    "origin": {
                        "z": zshift,
                        "x": np.round(
                            (geo["longitudeOfFirstGridPointInDegrees"] % 360 - x0) / dx,
                            1,
                        ),
                        "y": np.round(
                            (geo["latitudeOfFirstGridPointInDegrees"] - y0) / dy, 1
                        ),
                    },
                }

        if not field_map:
            raise ValueError(f"requested {param=} not found.")

        coords, shape = _gather_coords(field_map, dims)
        tcoords = _gather_tcoords(time_meta)
        hcoords = {
            "lon": (("y", "x"), self._grid.lon.data),
            "lat": (("y", "x"), self._grid.lat.data),
        }

        array = xr.DataArray(
            np.array([field_map.pop(key) for key in sorted(field_map)]).reshape(shape),
            coords=coords | hcoords | tcoords,
            dims=dims,
            attrs=metadata,
        )

        return (
            array if array.vcoord_type != "surface" else array.squeeze("z", drop=True)
        )

    def load_dataset(
        self,
        params: list[str],
        extract_pv: str | None = None,
    ) -> dict[str, xr.DataArray]:
        """Load data from GRIB files.

        Parameters
        ----------
        params : list[str]
            List of fields to load from the data files.
        datafiles : list[Path]
            List of files from which to load the data.
        extract_pv: str | None
            Optionally extract hybrid level coefficients from the given field.

        Raises
        ------
        RuntimeError
            if not all fields are found in the given datafiles.

        Returns
        -------
        dict[str, xr.DataArray]
            Mapping of fields by param name

        """

        if not _check_string_arg(params):
            raise ValueError(f"wrong type for arg {params=}. Expected str")

        _params = set(params)
        if extract_pv is not None and extract_pv not in _params:
            raise ValueError(f"If set, {extract_pv=} must be in {_params=}")

        data: dict[str, dict[tuple[int, ...], np.ndarray]] = {}
        result = {}

        for param in _params:
            result[param] = self._delayed(self._load_param)(param)

        if not _params == result.keys():
            raise RuntimeError(f"Missing params: {_params - data.keys()}")

        if extract_pv:
            result = result | _extract_pv(self.load_pv(extract_pv))

        return result

    def load(
        self,
        products: list[Product],
        extract_pv: str | None = None,
    ):
        params = set()
        for product in products:
            params |= set(product.input_fields)

        if self._ifs:
            self.load_ifs_data(params, extract_pv)
        else:
            if extract_pv:
                raise ValueError(f"{extract_pv=} can only be set for ifs data")
            return self.load_cosmo_data(params)

    def load_cosmo_data(
        self,
        params: list[str],
    ) -> dict[str, xr.DataArray]:
        """Load data from GRIB files.

        The COSMO definitions are enabled during the load.

        Parameters
        ----------
        params : list[str]
            List of fields to load from the data files.
        datafiles : list[Path]
            List of files from which to load the data.
        extract_pv: str | None
            Optionally extract hybrid level coefficients from the given field.

        Raises
        ------
        RuntimeError
            if not all fields are found in the given datafiles.

        Returns
        -------
        dict[str, xr.DataArray]
            Mapping of fields by param name

        """
        if not _cosmo_allowed:
            raise RuntimeError(
                "GRIB cache contains IFS defs, respawn process to clear."
            )

        global _ifs_allowed
        _ifs_allowed = False  # due to incompatible data in cache

        with cosmo_grib_defs():
            return self.load_dataset(params, extract_pv=None)

    def load_ifs_data(
        self,
        params: list[str],
        extract_pv: str | None = None,
    ) -> dict[str, xr.DataArray]:
        """Load data from GRIB files.

        Expects IFS data.

        Parameters
        ----------
        params : list[str]
            List of fields to load from the data files.
        datafiles : list[Path]
            List of files from which to load the data.
        extract_pv: str | None
            Optionally extract hybrid level coefficients from the given field.

        Raises
        ------
        RuntimeError
            if not all fields are found in the given datafiles.

        Returns
        -------
        dict[str, xr.DataArray]
            Mapping of fields by param name

        """
        if not _ifs_allowed:
            raise RuntimeError(
                "GRIB cache contains cosmo defs, respawn process to clear."
            )

        global _cosmo_allowed
        _cosmo_allowed = False  # due to incompatible data in cache

        mapping_path = files("idpi.data").joinpath("field_mappings.yml")
        mapping = yaml.safe_load(mapping_path.open())
        missing = set(params) - mapping.keys()
        if missing:
            msg = f"Some params are not present in the field mappings: {missing}"
            raise ValueError(msg)
        params_map = {mapping[p]["ifs"]["name"]: p for p in params}

        def get_unit_factor(key):
            param = params_map.get(key)
            if param is None:
                return 1
            return mapping[param].get("cosmo", {}).get("unit_factor", 1)

        ifs_params = list(params_map.keys())
        ifs_extract_pv = (
            mapping[extract_pv]["ifs"]["name"] if extract_pv is not None else None
        )
        ds = self.load_dataset(ifs_params, ifs_extract_pv)
        with xr.set_options(keep_attrs=True):
            return {params_map.get(k, k): get_unit_factor(k) * v for k, v in ds.items()}
