from operators.curl import curl, stpt, set_diff_type
from operators.theta import ftheta
import xarray as xr

def fpotvortic(QV, QC, QI, U, V, W, P, T) -> xr.DataArray:

    # prepare parameters
    curl1, curl2, curl3 = curl(U, V, W)
    theta = ftheta(T, P)

    # TODO
    rho = 0
    wi = 0
    wj = 0
    wk = 0
    sqrtg_r_s = 0
    dzeta_dlam = 0
    dzeta_dphi = 0
    cor2 = 0
    cor3 = 0

    # shortcut for stencils of theta
    def t(s: str) -> xr.DataArray:
        return theta[stpt(s)]

    out = (
        ((t("pcc") - t("mcc")) * wi + (t("ccp") - t("ccm")) * wk * dzeta_dlam) * curl1
        + ((t("cpc") - t("cmc") * wj) + (t("ccp") - t("ccm")) * wk * dzeta_dphi) * (curl2 + cor2)
        + ((t("ccp") - t("ccm")) * wk * (-sqrtg_r_s)) * (curl3 + cor3)
    ) / rho

    return out
