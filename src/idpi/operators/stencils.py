"""Finite difference stencils on xarray dataarrays."""
# Standard library
from typing import Literal
from typing import Protocol

# Third-party
import xarray as xr
import numpy as np

Dim = Literal["x", "y", "z"]

STENCILS = {
    "full": {
        -1: slice(-2),
        0: slice(1, -1),
        1: slice(2, None),
    },
    "half": {
        -1: slice(-1),
        0: slice(None),
        1: slice(1, None),
    },
}


class Field(Protocol):
    def dx(self) -> xr.DataArray:
        ...

    def dy(self) -> xr.DataArray:
        ...

    def dz(self) -> xr.DataArray:
        ...


class PaddedField:
    def __init__(self, field: xr.DataArray):
        self.indexes = field.indexes
        tmp = field.pad({"z": 1}, mode="edge")
        self.field = tmp.pad({dim: 1 for dim in ("x", "y")}, constant_values=np.nan)

    def __getitem__(self, indices: tuple[int, int, int]) -> xr.DataArray:
        i, j, k = indices
        s = STENCILS["full"]
        return self.field[{"x": s[i], "y": s[j], "z": s[k]}].assign_coords(self.indexes)

    def dx(self) -> xr.DataArray:
        return 0.5 * (self[1, 0, 0] - self[-1, 0, 0])

    def dy(self) -> xr.DataArray:
        return 0.5 * (self[0, 1, 0] - self[0, -1, 0])

    def dz(self) -> xr.DataArray:
        result = self[0, 0, 1] - self[0, 0, -1]
        result[{"z": slice(1, -1)}] *= 0.5
        return result


class StaggeredField:
    """Should have one more element along the staggered dimension."""

    def __init__(self, field: xr.DataArray, padded: PaddedField, dim: Dim):
        if field.indexes:
            field = field.drop_indexes(field.indexes.keys())
        self.dim = dim
        self.field = field
        self.padded = padded

    def __getitem__(self, indices: tuple[int, int, int]) -> xr.DataArray:
        i, j, k = indices
        s = STENCILS["half"]
        return self.field[{"x": s[i], "y": s[j], "z": s[k]}]

    def dx(self):
        if self.dim == "x":
            return self[1, 0, 0] - self[-1, 0, 0]
        return self.padded.dx()

    def dy(self):
        if self.dim == "y":
            return self[0, 1, 0] - self[0, -1, 0]
        return self.padded.dy()

    def dz(self):
        if self.dim == "z":
            return self[0, 0, 1] - self[0, 0, -1]
        return self.padded.dz()


def destagger_z(field: xr.DataArray) -> xr.DataArray:
    return 0.5 * (field + field.shift(z=-1)).dropna("z", how="all")


class TotalDiff:
    def __init__(self, dlon: float, dlat: float, hhl: xr.DataArray):
        self.dlon = dlon
        self.dlat = dlat
        z = "generalVertical"
        hhl_pad = PaddedField(hhl.rename({z: "z"}))
        dh_dx = destagger_z(hhl_pad.dx())  # order is important
        dh_dy = destagger_z(hhl_pad.dy())  # diff then destagger
        self.sqrtg_r_s = -1 / hhl.diff(dim=z, label="lower").rename({z: "z"})
        self.dzeta_dlam = self.sqrtg_r_s / dlon * dh_dx
        self.dzeta_dphi = self.sqrtg_r_s / dlat * dh_dy

    def d_dlam(self, field: Field) -> xr.DataArray:
        return field.dx() / self.dlon + field.dz() * self.dzeta_dlam

    def d_dphi(self, field: Field) -> xr.DataArray:
        return field.dy() / self.dlat + field.dz() * self.dzeta_dphi

    def d_dzeta(self, field: Field) -> xr.DataArray:
        return field.dz() * self.sqrtg_r_s
