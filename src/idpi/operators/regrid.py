"""Regridding operator."""

# Standard library
import dataclasses as dc
import typing

# Third-party
import numpy as np
import xarray as xr
from rasterio import transform
from rasterio import warp
from rasterio.crs import CRS

Resampling: typing.TypeAlias = warp.Resampling

CRS_ALIASES = {
    "geolatlon": "epsg:4326",
    "swiss": "epsg:21781",
    "swiss03": "epsg:21781",
    "swiss95": "epsg:2056",
    "boaga-west": "epsg:3003",
    "boaga-east": "epsg:3004",
}


def _get_crs(geo):
    if geo["gridType"] != "rotated_ll":
        raise NotImplementedError("Unsupported grid type")

    lon = geo["longitudeOfSouthernPoleInDegrees"]
    lat = -1 * geo["latitudeOfSouthernPoleInDegrees"]

    return CRS.from_string(
        f"+proj=ob_tran +o_proj=longlat +o_lat_p={lat} +lon_0={lon} +datum=WGS84"
    )


def _relimit(longitude: float) -> float:
    if longitude > 180:
        return longitude - 360
    return longitude


@dc.dataclass
class RegularGrid:
    crs: CRS
    nx: int
    ny: int
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def from_field(cls, field: xr.DataArray):
        geo = field.geography
        obj = cls(
            crs=_get_crs(geo),
            nx=geo["Ni"],
            ny=geo["Nj"],
            xmin=_relimit(geo["longitudeOfFirstGridPointInDegrees"]),
            xmax=_relimit(geo["longitudeOfLastGridPointInDegrees"]),
            ymin=geo["latitudeOfFirstGridPointInDegrees"],
            ymax=geo["latitudeOfLastGridPointInDegrees"],
        )
        if abs(obj.dx - geo["iDirectionIncrementInDegrees"]) > 1e-5:
            raise ValueError("Inconsistent grid parameters")
        if abs(obj.dy - geo["jDirectionIncrementInDegrees"]) > 1e-5:
            raise ValueError("Inconsistent grid parameters")
        return obj

    @classmethod
    def parse_regrid_operator(cls, op: str):
        crs_str, *grid_params = op.split(",")
        crs = CRS.from_string(CRS_ALIASES[crs_str])
        xmin, ymin, xmax, ymax, dx, dy = map(float, grid_params)
        nx = (xmax - xmin) / dx + 1
        ny = (ymax - ymin) / dy + 1
        if nx != int(nx) or ny != int(ny):
            raise ValueError("Inconsistent regrid parameters")
        return cls(crs, int(nx), int(ny), xmin, xmax, ymin, ymax)

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / (self.nx - 1)

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / (self.ny - 1)

    @property
    def transform(self) -> transform.Affine:
        return transform.from_origin(
            west=self.xmin - self.dx / 2,
            north=self.ymax + self.dy / 2,
            xsize=self.dx,
            ysize=self.dy,
        )


def regrid(field: xr.DataArray, dst: RegularGrid, resampling: Resampling):
    src = RegularGrid.from_field(field)

    def reproject_layer(field):
        output = np.zeros((dst.ny, dst.nx))
        warp.reproject(
            source=field[::-1],
            destination=output,
            src_crs=src.crs,
            src_transform=src.transform,
            dst_crs=dst.crs,
            dst_transform=dst.transform,
            resampling=resampling,
        )
        return output[::-1]

    return xr.apply_ufunc(
        reproject_layer,
        field,
        input_core_dims=[["y", "x"]],
        output_core_dims=[["y1", "x1"]],
        vectorize=True,
    ).rename({"x1": "x", "y1": "y"})
