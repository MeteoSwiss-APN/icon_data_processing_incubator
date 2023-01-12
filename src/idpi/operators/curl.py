import xarray as xr
import numpy as np

from idpi import constants as const

__diff_type: str = "center"
__s: dict[str, slice] = {}


def set_diff_type(dt: str):
    """Set the type of differentiation to be used for functions of this module. Allowed values are 'left', 'center', 'right'."""
    global __diff_type, __s
    __diff_type = dt
    c = slice(1, -1)
    m = slice(0, -2) if dt != "right" else c
    p = slice(2, None) if dt != "left" else c
    __s = {"c": c, "m": m, "p": p}


set_diff_type("center")  # default


def stpt(stencil: str) -> dict[str, slice]:
    """
    Return a dictionary which indicates the stencil points of a data array as specified by the argument.

    This dictionary can be used for selecting values from a data array.
    Consider the following example, where ``A`` is a data array. ::

        B = A[stpt('ccc')] + A[stpt('cpm')]

    Here we add for every point a in ``A`` its upper (y+1) neighbor, one level below (z-1) and store the result in ``B``.
    In this method we allow 3d neighbor stencils. Points on the border won't be included.

    Depending on the ``diff_type`` (set with ``set_diff_type``), the stencils will be interpreted differently.
    For 'left', positive stencils (+1, 'p') will be mapped to the original point.
    For 'right', negative stencils (-1, 'm') will be mapped to the original point.

    Args:
        stencil (str): A string indicating the stencil.
        This string must have three characters, corresponding to the 'x', 'y' and 'z' dimension (in this order).
        The characters can be

        - 'c': center (+/-0)
        - 'm': below (-1)
        - 'p': above (+1)
    """
    global __s
    return {d: __s[k] for d, k in zip("xyz", stencil)}


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
    def win(s: str) -> xr.DataArray:
        """
        Retrun the application of the ``stpt`` function on either ``U``, ``V`` or ``W``, depending on the first character of the argument.

        Args:
            s (str): A 4-character string. The first character indicates the data array for which the stencil should be applied and can be 'u', 'v' or 'w'.
            The last 3 characters are passed to ``stpt``.
        """
        x = s[0]
        X = U if x == "u" else V if x == "v" else W
        return X[stpt(s[1:])]

    # prepare parameters

    r_earth_inv = 1 / const.earth_radius
    mdeg2rad = 1e-6 * np.pi / 180
    acrlat: xr.DataArray = 1 / (np.cos(lat * mdeg2rad) * const.earth_radius)  # type: ignore[assignment]
    tgrlat: xr.DataArray = np.tan(lat * mdeg2rad)  # type: ignore[assignment]

    dzeta_dlam = (
        0.25
        * inv_dlon
        * sqrtg_r_s
        * (
            (HHL[stpt("pcc")] - HHL[stpt("mcc")])
            + (HHL[stpt("pcp")] - HHL[stpt("mcp")])
        )
    )

    dzeta_dphi = (
        0.25
        * inv_dlat
        * sqrtg_r_s
        * (
            (HHL[stpt("cpc")] - HHL[stpt("cmc")])
            + (HHL[stpt("cpp")] - HHL[stpt("cmp")])
        )
    )

    # compute weighted derivatives for FD

    du_dphi = (
        ((win("ucpc") + win("umpc")) - (win("ucmc") + win("ummc"))) * 0.25 * inv_dlat
    )
    du_dzeta = ((win("uccp") + win("umcp")) - (win("uccm") + win("umcm"))) * 0.5 * wk

    dv_dlam = (
        ((win("vpcc") + (win("vpmc")) - (win("vmcc") + win("vmmc")))) * 0.25 * inv_dlon
    )
    dv_dzeta = ((win("vccp") + (win("vcmp")) - (win("vccm") + win("vcmm")))) * 0.5 * wk

    dw_dlam = (
        ((win("wpcp") + (win("wpcc")) - (win("wmcp") + win("wmcc")))) * 0.25 * inv_dlon
    )
    dw_dphi = (
        ((win("wcpp") + win("wcpc")) - (win("wcmp") + win("wcmc"))) * 0.25 * inv_dlat
    )
    dw_dzeta = win("wccp") - win("wccc")

    # compute curl
    curl1 = acrlat * (
        r_earth_inv * (dw_dphi + dzeta_dphi * dw_dzeta)
        + sqrtg_r_s * dv_dzeta
        - r_earth_inv * 0.5 * (win("vccc") + win("vcmc"))
    )
    curl2 = r_earth_inv * (
        -sqrtg_r_s * du_dzeta
        - acrlat * (dw_dlam + dzeta_dlam * dw_dzeta)
        + r_earth_inv * 0.5 * (win("uccc") + win("umcc"))
    )
    curl3 = (
        acrlat * (dv_dlam + dzeta_dlam * dv_dzeta)
        - r_earth_inv * (du_dphi + dzeta_dphi * du_dzeta)
        + r_earth_inv * tgrlat * 0.5 * (win("uccc") + win("umcc"))
    )

    return curl1, curl2, curl3
