"""algorithm for BRN operator."""
import numpy as np
import xarray as xr
from operators.destagger import destagger
from operators.thetav import fthetav

pc_g = 9.80665


def fbrn(p, t, qv, u, v, hhl, hsurf):
    """Bulk Richardson Number (BRN)."""
    nlevels = len(p.coords["generalVerticalLayer"])

    thetav = fthetav(p, t, qv)
    thetav_sum = thetav.isel(generalVerticalLayer=slice(None, None, -1)).cumsum(
        dim="generalVerticalLayer"
    )

    nlevels_xr = xr.DataArray(
        data=np.arange(nlevels, 0, -1), dims=["generalVerticalLayer"]
    )
    u_ = destagger(u, "x")
    v_ = destagger(v, "y")
    hfl = destagger(hhl, "generalVertical")

    brn = (
        pc_g
        * (hfl - hsurf)
        * (thetav - thetav.isel(generalVerticalLayer=79))
        * nlevels_xr
        / (thetav_sum * (u_**2 + v_**2))
    )

    return brn
