"""Various time operators."""
# Third-party
import numpy as np
import xarray as xr
import pandas as pd


def time_rate(var: xr.DataArray, dtime: np.timedelta64):
    """Compute a time rate for a given delta in time.

    It assumes the input data is an accumulated value
    between two time steps of the time coordinate

    Args:
        var: variable that contains the input data
        dtime: delta time of the desired output time rate

    """
    coord = xr.apply_ufunc(pd.to_datetime, var.coords["step"], kwargs={"unit": "h"})              
    return (
        var.isel(step=slice(1, None))
        - var.isel(step=slice(0, -1)).assign_coords(
            {"step": var[{"step": slice(1, None)}].step}
        )
    ) / (
        (
            coord.isel(step=slice(1, None))
            - coord.isel(step=slice(0, -1)).assign_coords(
                {"step": coord[{"step": slice(1, None)}].step}
            )
        )
        / dtime
    )
