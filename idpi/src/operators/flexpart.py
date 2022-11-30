import numpy as np
from operators.omega_slope import omega_slope


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

    ds_out["TOT_CON"] = (
        (ds["TOT_CON"].isel(step=istep) - ds["TOT_CON"].isel(step=istep - 1))
        * 0.333333
        * 1000
    )
    ds_out["TOT_CON"].attrs = ds["TOT_CON"].attrs
    ds_out["TOT_GSP"] = (
        (ds["TOT_GSP"].isel(step=istep) - ds["TOT_GSP"].isel(step=istep - 1))
        * 0.333333
        * 1000
    )
    ds_out["TOT_GSP"].attrs = ds["TOT_GSP"].attrs
    ds_out["SSR"] = (ds["SSR"].isel(step=istep) - ds["SSR"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["SSR"].attrs = ds["SSR"].attrs
    ds_out["SSHF"] = (ds["SSHF"].isel(step=istep) - ds["SSHF"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["SSHF"].attrs = ds["SSHF"].attrs
    ds_out["EWSS"] = (ds["EWSS"].isel(step=istep) - ds["EWSS"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["EWSS"].attrs = ds["EWSS"].attrs

    ds_out["OMEGA_SLOPE"] = omega_slope(
        ds["PS"].isel(step=istep), ds["ETADOT"].isel(step=istep), ds["ak"], ds["bk"]
    ).isel(hybrid=slice(39, 61))

    return ds_out
