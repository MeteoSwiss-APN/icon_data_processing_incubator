"""Implementation of the potential vorticity operator."""

# Standard library
from functools import partial
from typing import Callable
from typing import cast

# Third-party
import numpy as np
import xarray as xr

# Local
from .. import constants as const
from .curl import curl
from .curl import stpt as stpt_
from .rho import f_rho_tot
from .theta import ftheta


def fpotvortic(
    U: xr.DataArray,
    V: xr.DataArray,
    W: xr.DataArray,
    P: xr.DataArray,
    T: xr.DataArray,
    HHL: xr.DataArray,
    QV: xr.DataArray,
    QC: xr.DataArray,
    QI: xr.DataArray,
    QW_load: xr.DataArray | None = None,
    diff_type: str | None = None,
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

    The effect of water loading is ignored if ``QW_load`` is not specified.

    The type of the numerical derivatives can be specified by setting ``diff_type`` to
    either 'left', 'right' or 'center' (default).
    For 'center', both neighboring values will be used.
    With 'left', only the left neighbors will be used and the right neighbor will be
    assumed to be equal to the current value.
    With 'right', it's the other way around.
    The left neighbor's index is one lower than the current index, while the right
    neighbor's index is one higher.

    All input arrays must have the same coordinates and sizes.
    The output will be the same orthotope as the inputs, where the boundary was removed.
    The boundaries of some inputs are used for computing the numerical derivatives and
    hence, they cannot be included in the result.

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
    stpt = cast(
        Callable[[str], dict[str, slice]],
        partial(stpt_, diff_type=diff_type) if diff_type else stpt_,
    )

    # prepare parameters
    dlon = U.attrs["GRIB_iDirectionIncrementInDegrees"]
    dlat = U.attrs["GRIB_jDirectionIncrementInDegrees"]
    deg2rad = np.pi / 180
    inv_dlon = 1 / (dlon * deg2rad)
    inv_dlat = 1 / (dlat * deg2rad)

    # target coordinates
    # the output array is missing the border
    mask = {"x": slice(1, -1), "y": slice(1, -1)}
    lat = P["latitude"][mask]

    # FD (finite differences) weights
    wi = 0.5 * inv_dlon
    wj = 0.5 * inv_dlat
    wk = 0.5 if diff_type == "center" else 1.0

    zmax = U.sizes.get("generalVerticalLayer")

    def prepare_array(array):
        if "generalVertical" in array.dims:
            return (
                array.rename(generalVertical="z")
                .isel({"z": slice(zmax)})
                .drop_indexes("z")
            )
        elif "generalVerticalLayer" in array.dims:
            return array.rename(generalVerticalLayer="z").drop_indexes("z")
        return array

    U, V, W, P, T, HHL, QV, QC, QI = map(
        prepare_array,
        (U, V, W, P, T, HHL, QV, QC, QI),
    )
    if QW_load:
        QW_load = QW_load.drop_indexes("z")

    # computation

    # metric coefficients at the scalar grid positions

    # shortcut for stencils of hhl
    def hhl(s: str) -> xr.DataArray:
        return HHL[stpt(s)]

    sqrtg_r_s = 1.0 / (hhl("ccc") - hhl("ccp"))
    dzeta_dlam = (
        0.25
        * inv_dlon
        * sqrtg_r_s
        * ((hhl("pcc") - hhl("mcc")) + (hhl("pcp") - hhl("mcp")))
    )
    dzeta_dphi = (
        0.25
        * inv_dlat
        * sqrtg_r_s
        * ((hhl("cpc") - hhl("cmc")) + (hhl("cpp") - hhl("cmp")))
    )

    # curl
    curl1, curl2, curl3 = curl(U, V, W, HHL, inv_dlon, inv_dlat, wk, sqrtg_r_s, lat)

    # potential temperature
    theta = ftheta(P, T)

    # shortcut for stencils of theta
    def t(s: str) -> xr.DataArray:
        return theta[stpt(s)]

    # coriolis terms
    cor2 = 2 * const.pc_omega / const.earth_radius * np.cos(lat * deg2rad)
    cor3 = 2 * const.pc_omega * np.sin(lat * deg2rad)

    # total air density
    rho = f_rho_tot(T, P, QV, QC, QI, QW_load)

    # potential vorticity
    out = xr.full_like(rho, np.nan)
    out[stpt("ccc")] = (
        ((t("pcc") - t("mcc")) * wi + (t("ccp") - t("ccm")) * wk * dzeta_dlam) * curl1
        + ((t("cpc") - t("cmc")) * wj + (t("ccp") - t("ccm")) * wk * dzeta_dphi)
        * (curl2 + cor2)
        + ((t("ccp") - t("ccm")) * wk * (-sqrtg_r_s)) * (curl3 + cor3)
    ) / rho[stpt("ccc")]

    return out.rename(z="generalVerticalLayer")
