"""Algorithm for the curl operator."""

# Standard library
from typing import cast

# Third-party
import numpy as np
import xarray as xr

# Local
from .. import constants as const

DEFAULT_DIFF_TYPE = "center"
STENCILS = {
    "left": {"m": slice(0, -2), "c": slice(1, -1), "p": slice(1, -1)},
    "center": {"m": slice(0, -2), "c": slice(1, -1), "p": slice(2, None)},
    "right": {"m": slice(1, -1), "c": slice(1, -1), "p": slice(2, None)},
}


def stpt(stencil: str, diff_type: str | None = None) -> dict[str, slice]:
    """Get stencil points.

    This dictionary can be used for selecting values from a data array.
    Consider the following example, where ``A`` is a data array. ::

        B = A[stpt('ccc')] + A[stpt('cpm')]

    Here we add for every point a in ``A`` its upper (y+1) neighbor, one level
    below (z-1) and store the result in ``B``.
    In this method we allow 3d neighbor stencils. Points on the border won't
    be included.

    The stencil definition string must have three characters, corresponding to
    the 'x', 'y' and 'z' dimensions respectively.
    The characters can be
    - 'c': center (+/-0)
    - 'm': below (-1)
    - 'p': above (+1)

    Depending on the ``diff_type`` (set with ``set_diff_type``), the stencils
    will be interpreted differently.
    For 'left', positive stencils (+1, 'p') will be mapped to the original point.
    For 'right', negative stencils (-1, 'm') will be mapped to the original point.

    Args:
        stencil (str): A string defining the stencil.
        diff_type (str): A string indicating the difference type.

    """
    if diff_type is None:
        diff_type = DEFAULT_DIFF_TYPE
    s = STENCILS[diff_type]
    return {d: s[k] for d, k in zip("xyz", stencil)}


class FDField:
    """Syntaxic sugar to apply the stencil to a field."""

    def __init__(self, field: xr.DataArray, diff_type: str = DEFAULT_DIFF_TYPE):
        """Init FDField."""
        self.field = field
        self.diff_type = diff_type

    def __getitem__(self, stencil: str):
        return self.field[stpt(stencil, diff_type=self.diff_type)]

    @classmethod
    def from_fields(cls, *fields, diff_type: str = DEFAULT_DIFF_TYPE):
        for field in fields:
            yield cls(field, diff_type)


def curl(
    U: xr.DataArray,
    V: xr.DataArray,
    W: xr.DataArray,
    HHL: xr.DataArray,
    inv_dlon: float,
    inv_dlat: float,
    wk: float,
    sqrtg_r_s: xr.DataArray,
    lat: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute the curl of the velocity field."""
    r_earth_inv = 1 / const.earth_radius
    deg2rad = np.pi / 180
    acrlat = cast(xr.DataArray, 1 / (np.cos(lat * deg2rad) * const.earth_radius))
    tgrlat = cast(xr.DataArray, np.tan(lat * deg2rad))

    u, v, w, hhl = FDField.from_fields(U, V, W, HHL)

    dzeta_dlam = (
        0.25
        * inv_dlon
        * sqrtg_r_s
        * ((hhl["pcc"] - hhl["mcc"]) + (hhl["pcp"] - hhl["mcp"]))
    )

    dzeta_dphi = (
        0.25
        * inv_dlat
        * sqrtg_r_s
        * ((hhl["cpc"] - hhl["cmc"]) + (hhl["cpp"] - hhl["cmp"]))
    )

    # compute weighted derivatives for FD

    du_dphi = ((u["cpc"] + u["mpc"]) - (u["cmc"] + u["mmc"])) * 0.25 * inv_dlat
    du_dzeta = ((u["ccp"] + u["mcp"]) - (u["ccm"] + u["mcm"])) * 0.5 * wk

    dv_dlam = ((v["pcc"] + v["pmc"]) - (v["mcc"] + v["mmc"])) * 0.25 * inv_dlon
    dv_dzeta = ((v["ccp"] + v["cmp"]) - (v["ccm"] + v["cmm"])) * 0.5 * wk

    dw_dlam = ((w["pcp"] + w["pcc"]) - (w["mcp"] + w["mcc"])) * 0.25 * inv_dlon
    dw_dphi = ((w["cpp"] + w["cpc"]) - (w["cmp"] + w["cmc"])) * 0.25 * inv_dlat
    dw_dzeta = w["ccp"] - w["ccc"]

    # compute curl
    curl1 = acrlat * (
        r_earth_inv * (dw_dphi + dzeta_dphi * dw_dzeta)
        + sqrtg_r_s * dv_dzeta
        - r_earth_inv * 0.5 * (v["ccc"] + v["cmc"])
    )
    curl2 = r_earth_inv * (
        -sqrtg_r_s * du_dzeta
        - acrlat * (dw_dlam + dzeta_dlam * dw_dzeta)
        + r_earth_inv * 0.5 * (u["ccc"] + u["mcc"])
    )
    curl3 = (
        acrlat * (dv_dlam + dzeta_dlam * dv_dzeta)
        - r_earth_inv * (du_dphi + dzeta_dphi * du_dzeta)
        + r_earth_inv * tgrlat * 0.5 * (u["ccc"] + u["mcc"])
    )

    return curl1, curl2, curl3
