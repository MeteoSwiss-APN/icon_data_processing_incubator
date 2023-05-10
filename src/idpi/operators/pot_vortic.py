"""Implementation of the potential vorticity operator."""

# Third-party
import numpy as np
import xarray as xr

# Local
from .. import constants as const
from .curl import curl
from .stencils import TotalDiff, PaddedField


def fpotvortic(
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray,
    theta: xr.DataArray,
    rho_tot: xr.DataArray,
    total_diff: TotalDiff,
) -> xr.DataArray:
    r"""Compute the potential vorticity.

    The potential vorticity is computed with the following formula:

    .. math::
        v_p = \frac{1}{\rho} * \frac{\partial \Theta}{\partial \z} * (c_v + 2 \Omega)

    where
    :math:`\rho` is the total air density,
    :math:`\frac{\partial \Theta}{\partial \z}`
    is the vertical gradient of the potential temperature,
    :math:`c_v` is the curl of the wind in y direction and
    :math`\Omega` is the coriolis term.

    Args:
        U (xr.DataArray): Wind in x direction
        V (xr.DataArray): Wind in y direction
        W (xr.DataArray): Wind in z direction
        P (xr.DataArray): Pressure
        T (xr.DataArray): Temperature
        HHL (xr.DataArray): Height of half-layers
        QV (xr.DataArray): Specific humidity [kg/kg]
        QC (xr.DataArray): Specific cloud water content [kg/kg]
        QI (xr.DataArray): Specific cloud ice content [kg/kg].
        QW_load (xr.DataArray, optional): Specific precipitable components
            content [kg/kg]. Defaults to None.
        diff_type (str, optional): The type of differentiation. Defaults to None.

    Returns:
        xr.DataArray: The potential vorticity

    """
    # target coordinates
    deg2rad = np.pi / 180
    lat = (rho_tot["latitude"] * deg2rad).astype(np.float32)

    # compute curl
    curl1, curl2, curl3 = curl(u, v, w, lat, total_diff)

    # coriolis terms
    cor2 = 2 * const.pc_omega / const.earth_radius * np.cos(lat)
    cor3 = 2 * const.pc_omega * np.sin(lat)

    t = PaddedField(theta.rename(generalVerticalLayer="z"))
    dt_dlam = total_diff.d_dlam(t)
    dt_dphi = total_diff.d_dphi(t)
    dt_dzeta = total_diff.d_dzeta(t)

    # potential vorticity
    out = (
        dt_dlam * curl1 + dt_dphi * (curl2 + cor2) - dt_dzeta * (curl3 + cor3)
    ) / rho_tot.rename(generalVerticalLayer="z")

    return out.rename(z="generalVerticalLayer")
