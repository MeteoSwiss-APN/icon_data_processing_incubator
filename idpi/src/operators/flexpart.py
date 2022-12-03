# Third-party
import numpy as np
from operators.omega_slope import omega_slope
from operators.time_operators import time_rate


def fflexpart(ds, istep):
    ds_out = {}
    for field in (
        "U",
        "V",
        "T",
        "QV",
        "PS",
        "U_10M",
        "V_10M",
        "T_2M",
        "TD_2M",
        "CLCT",
        "W_SNOW",
    ):
        ds_out[field] = ds[field].isel(step=istep)

    ds_out["TOT_CON"] = time_rate(
        ds["TOT_CON"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "h")
    )
    ds_out["TOT_CON"].attrs = ds["TOT_CON"].attrs
    ds_out["TOT_GSP"] = time_rate(
        ds["TOT_GSP"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "h")
    )

    ds_out["TOT_GSP"].attrs = ds["TOT_GSP"].attrs
    ds_out["SSR"] = time_rate(
        ds["SSR"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["SSR"].attrs = ds["SSR"].attrs
    ds_out["SSHF"] = time_rate(
        ds["SSHF"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["SSHF"].attrs = ds["SSHF"].attrs
    ds_out["EWSS"] = time_rate(
        ds["EWSS"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )

    ds_out["EWSS"].attrs = ds["EWSS"].attrs

    ds_out["OMEGA_SLOPE"] = omega_slope(
        ds["PS"].isel(step=istep), ds["ETADOT"].isel(step=istep), ds["ak"], ds["bk"]
    ).isel(hybrid=slice(39, 61))

    return ds_out
