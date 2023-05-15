"""Algorithm for the curl operator."""

# Standard library
from typing import cast

# Third-party
import numpy as np
import xarray as xr

# Local
from .. import constants as const
from .destagger import destagger
from .stencils import PaddedField
from .stencils import StaggeredField
from .stencils import TotalDiff


def curl(
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray,
    rlat: xr.DataArray,
    total_diff: TotalDiff,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute the curl of the velocity field."""
    r_earth_inv = 1 / const.earth_radius
    acrlat = cast(xr.DataArray, 1 / (np.cos(rlat) * const.earth_radius))
    tgrlat = cast(xr.DataArray, np.tan(rlat))

    # compute weighted derivatives for FD
    u_f = destagger(u, "x")
    v_f = destagger(v, "y")

    u_p = PaddedField(u_f)
    v_p = PaddedField(v_f)
    w_s = StaggeredField(w)

    du_dphi = total_diff.d_dphi(u_p)
    du_dzeta = total_diff.d_dzeta(u_p)
    dv_dlam = total_diff.d_dlam(v_p)
    dv_dzeta = total_diff.d_dzeta(v_p)
    dw_dlam = total_diff.d_dlam(w_s)
    dw_dphi = total_diff.d_dphi(w_s)

    # compute curl
    curl1 = acrlat * (r_earth_inv * dw_dphi + dv_dzeta - r_earth_inv * v_f)
    curl2 = r_earth_inv * (-du_dzeta - acrlat * dw_dlam + r_earth_inv * u_f)
    curl3 = acrlat * dv_dlam + r_earth_inv * (-du_dphi + tgrlat * u_f)

    return curl1, curl2, curl3
