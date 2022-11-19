import numpy as np
# similar to the subtract.accumulate but permute the order of the operans of the diff
# TODO implement as a ufunc
def cumdiff(A, axis):
    r = np.empty(np.shape(A))
    t = 0  # op = the ufunc being applied to A's  elements
    for i in range(np.shape(A)[axis]):
        t = np.take(A, i, axis) - t

        slices = []
        for dim in range(A.ndim):
            if dim == axis:
                slices.append(slice(i, i + 1))
            else:
                slices.append(slice(None))

        r[tuple(slices)] = np.expand_dims(t, axis=t.ndim)
    return r

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

    surface_pressure_ref = 101325.0

    ak1 = ds["ak"][1:].assign_coords(
        {"hybrid": ds["ak"][{"hybrid": slice(0, -1)}].hybrid}
    )
    bk1 = ds["bk"][1:].assign_coords(
        {"hybrid": ds["bk"][{"hybrid": slice(0, -1)}].hybrid}
    )

    omega_slope = (
        2.0
        * ds["PS"].isel(step=istep)
        * ds["ETADOT"].isel(step=istep)
        * ((ak1 - ds["ak"][0:-1]) / ds["PS"].isel(step=istep) + bk1 - ds["bk"][0:-1])
        / ((ak1 - ds["ak"][0:-1]) / surface_pressure_ref + bk1 - ds["bk"][0:-1])
    )

    ds_out["OMEGA_SLOPE"] = omega_slope.reduce(cumdiff, dim="hybrid").isel(
        hybrid=slice(39, 61)
    )

    return ds_out
