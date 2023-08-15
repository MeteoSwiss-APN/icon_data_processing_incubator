# Standard library
from typing import Literal

# Third-party
import numpy as np
import xarray as xr


def compute_weights(
    window_size: int,
    window_type: Literal["exp", "const"],
    window_shape: Literal["disk", "square"],
) -> xr.DataArray:
    n = window_size

    if n % 2 != 1:
        raise ValueError("window_size must be an odd number.")

    radius = n // 2
    yy, xx = np.mgrid[:n, :n] - radius
    dist = np.sqrt(xx**2 + yy**2)

    if window_type == "exp":
        kernel = np.exp(-dist)
    else:
        kernel = np.ones((n, n))

    weights = xr.DataArray(kernel, dims=["win_x", "win_y"])

    if window_shape == "disk":
        return weights.where(dist <= radius, 0.0)

    return weights


def compute_cond_mask(
    windows: xr.DataArray, weights: xr.DataArray, frac_val: float
) -> xr.DataArray:
    mask = weights > 0

    loc_x = windows.x + windows.win_x - windows.sizes["win_x"] // 2
    in_bnds_x = np.logical_and(0 <= loc_x, loc_x < windows.sizes["x"])

    loc_y = windows.y + windows.win_y - windows.sizes["win_y"] // 2
    in_bnds_y = np.logical_and(0 <= loc_y, loc_y < windows.sizes["y"])

    undef = windows.isnull()
    frac_undef = undef.where(mask).where(in_bnds_x).where(in_bnds_y).mean(weights.dims)
    return frac_undef <= frac_val


def fill_undef(field: xr.DataArray, radius: int, frac_val: float) -> xr.DataArray:
    n = 2 * radius + 1
    weights = compute_weights(n, "exp", "disk")

    # construct rolling windows
    dims = {"x": "win_x", "y": "win_y"}
    windows = field.rolling({"x": n, "y": n}, center=True).construct(dims)

    # compute conditional mask
    cond = compute_cond_mask(windows, weights, frac_val)

    # compute weighted mean skipping undefined values
    smoothed = windows.weighted(weights).mean(dims.values())

    # replace undefined values in input field
    return xr.where(field.isnull(), smoothed.where(cond), field)


def disk_avg(field: xr.DataArray, radius: int) -> xr.DataArray:
    n = 2 * radius + 1
    weights = compute_weights(n, "const", "disk")

    # construct rolling windows
    dims = {"x": "win_x", "y": "win_y"}
    windows = field.rolling({"x": n, "y": n}, center=True).construct(dims)

    # compute weighted mean skipping undefined values
    smoothed = windows.weighted(weights).mean(dims.values())

    return smoothed.where(~field.isnull())
