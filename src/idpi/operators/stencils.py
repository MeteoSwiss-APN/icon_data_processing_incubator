"""Finite difference stencils on xarray dataarrays."""
# Standard library
from typing import Literal
from typing import Protocol

# Third-party
import xarray as xr
import numpy as np

Vert = Literal["generalVerticalLayer", "generalVertical"]

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
    def __init__(self, field: xr.DataArray, vert: Vert = "generalVerticalLayer"):
        self.indexes = field.indexes
        self.vert = vert
        tmp = field.pad({vert: 1}, mode="edge")
        self.field = tmp.pad({dim: 1 for dim in ("x", "y")}, constant_values=np.nan)

    def __getitem__(self, indices: tuple[int, int, int]) -> xr.DataArray:
        i, j, k = indices
        s = STENCILS["full"]
        sel = {"x": s[i], "y": s[j], self.vert: s[k]}
        return self.field[sel].assign_coords(self.indexes)

    def dx(self) -> xr.DataArray:
        return 0.5 * (self[1, 0, 0] - self[-1, 0, 0])

    def dy(self) -> xr.DataArray:
        return 0.5 * (self[0, 1, 0] - self[0, -1, 0])

    def dz(self) -> xr.DataArray:
        result = self[0, 0, 1] - self[0, 0, -1]
        result[{self.vert: slice(1, -1)}] *= 0.5
        return result


class StaggeredField:
    """Should have one more element along the staggered dimension."""

    def __init__(self, field: xr.DataArray):
        if field.indexes:
            field = field.drop_indexes(field.indexes.keys())
        self.field = field
        self.padded = PaddedField(destagger_z(field))

    def __getitem__(self, indices: tuple[int, int, int]) -> xr.DataArray:
        i, j, k = indices
        s = STENCILS["half"]
        return self.field[{"x": s[i], "y": s[j], "generalVertical": s[k]}]

    def dx(self):
        return self.padded.dx()

    def dy(self):
        return self.padded.dy()

    def dz(self):
        return (self[0, 0, 1] - self[0, 0, -1]).rename(
            generalVertical="generalVerticalLayer"
        )


def destagger_z(field: xr.DataArray) -> xr.DataArray:
    z = "generalVertical"
    result = 0.5 * (field + field.shift({z: -1})).dropna(z, how="all")
    return result.rename({z: "generalVerticalLayer"})


class TotalDiff:
    def __init__(self, dlon: float, dlat: float, hhl: xr.DataArray):
        self.dlon = dlon
        self.dlat = dlat
        hhl_pad = PaddedField(hhl, "generalVertical")
        dh_dx = destagger_z(hhl_pad.dx())  # order is important
        dh_dy = destagger_z(hhl_pad.dy())  # diff then destagger

        dim = "generalVertical"
        rename_kw = {dim: "generalVerticalLayer"}
        self.sqrtg_r_s = -1 / hhl.diff(dim=dim, label="lower").rename(rename_kw)
        self.dzeta_dlam = self.sqrtg_r_s / dlon * dh_dx
        self.dzeta_dphi = self.sqrtg_r_s / dlat * dh_dy

    def d_dlam(self, field: Field) -> xr.DataArray:
        return field.dx() / self.dlon + field.dz() * self.dzeta_dlam

    def d_dphi(self, field: Field) -> xr.DataArray:
        return field.dy() / self.dlat + field.dz() * self.dzeta_dphi

    def d_dzeta(self, field: Field) -> xr.DataArray:
        return field.dz() * self.sqrtg_r_s
