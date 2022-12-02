import xarray as xr
import numpy as np

def time_rate(var: xr.DataArray, dtime: np.timedelta64):
    coord=var.coords["step"]
    return (var.isel(step=slice(1,None)) - var.isel(step=slice(0,-1)).assign_coords({"step": var[{"step": slice(1, None)}].step})  ) / ( (coord.isel(step=slice(1,None)) - coord.isel(step=slice(0,-1)).assign_coords({"step": coord[{"step": slice(1, None)}].step}) ) / dtime )
