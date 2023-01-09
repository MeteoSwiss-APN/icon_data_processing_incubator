from .. import constants as const
import numpy as np
import xarray as xr
from .curl import curl, set_diff_type, stpt
from .rho import f_rho_tot
from .theta import ftheta

mc_deg_to_rad = np.pi / 180
mc_rad_to_deg = 180 / np.pi


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
    diff_type=None,
) -> xr.DataArray:
    """Compute the potential vorticity.

    The potential vorticity is computed with the following formula:

    .. math::
        v_p = \frac{1}{\rho} * \frac{\partial \Theta}{\partial \z} * (c_v + 2 \Omega)

    where :math:`\rho` is the total air density, :math:`\frac{\partial \Theta}{\partial \z}` is the gradient of the potential temperature,
    :math:`c_v` is the curl of the wind in y direction and :math`\Omega` is the coriolis term.

    The effect of water loading is ignored if ``QW_load`` is not specified.

    The type of the numerical derivatives can be specified by setting ``diff_type`` to either 'left', 'right' or 'center' (default).
    For 'center', both neighboring values will be used.
    With 'left', only the left neighbors will be used and the right neighbor will be assumed to be equal to the current value.
    With 'right', it's the other way around.
    The left neighbor's index is one lower than the current index, while the right neighbor's index is one higher.

    All input arrays must have the same coordinates and sizes.
    The output will be the same orthotope as the inputs, where the boundary was removed.
    The boundaries of some inputs are used for computing the numerical derivatives and hence, they cannot be included in the result.

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
        QW_load (xr.DataArray | None, optional): Specific precipitable components content [kg/kg]. Defaults to None.
        diff_type (_type_, optional): The type of differentiation. Defaults to None.

    Returns:
        xr.DataArray: The potential vorticity
    """

    # prepare parameters

    lon = U["longitude"]
    lat = U["latitude"]
    mdeg2rad = 1e-6 * mc_deg_to_rad
    inv_dlon = 1 / ((lon[1, 0] - lon[0, 0]) * mdeg2rad)
    inv_dlat = 1 / ((lat[0, 1] - lat[0, 0]) * mdeg2rad)

    # FD (finite differences) weights
    wi = 0.5 * inv_dlon
    wj = 0.5 * inv_dlat
    wk = 0.5 if diff_type == "center" else 1.0

    # other preparations

    set_diff_type(diff_type)

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
    curl1, curl2, curl3 = curl(U, V, W)

    # potential temperature
    theta = ftheta(T, P)

    # shortcut for stencils of theta
    def t(s: str) -> xr.DataArray:
        return theta[stpt(s)]

    # coriolis terms
    cor2 = 2 * const.pc_omega / const.earth_radius * np.cos(lat * mc_deg_to_rad)
    cor3 = 2 * const.pc_omega * np.sin(lat * mc_deg_to_rad)

    # total air density
    if QW_load is None:
        rho = f_rho_tot(T, P, QV, QC, QI)
    else:
        rho = f_rho_tot(T, P, QV, QC, QI, QW_load)

    # potential vorticity
    out = (
        ((t("pcc") - t("mcc")) * wi + (t("ccp") - t("ccm")) * wk * dzeta_dlam) * curl1
        + ((t("cpc") - t("cmc") * wj) + (t("ccp") - t("ccm")) * wk * dzeta_dphi)
        * (curl2 + cor2)
        + ((t("ccp") - t("ccm")) * wk * (-sqrtg_r_s)) * (curl3 + cor3)
    ) / rho

    # set coordinates of output
    stpt2 = stpt("ccc")
    stpt2.pop("z")
    out["longitude"] = lon[stpt2]
    out["latitude"] = lat[stpt2]

    return out
