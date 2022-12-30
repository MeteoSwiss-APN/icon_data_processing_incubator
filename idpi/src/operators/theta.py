import xarray as xr

pc_r_d = 287.05
pc_cp_d = 1005.0

def ftheta(T: xr.DataArray, P: xr.DataArray):
    exp = pc_r_d/pc_cp_d
    out = T * (1e5/P)**exp
    return out