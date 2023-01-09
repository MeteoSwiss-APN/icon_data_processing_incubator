import constants as const
import numpy as np
import xarray as xr
from operators.curl import curl, set_diff_type, stpt
from operators.rho import f_rho_tot
from operators.theta import ftheta

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
    diff_type="center",
) -> xr.DataArray:

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
