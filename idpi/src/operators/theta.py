import xarray as xr
import constants as const

def ftheta(T: xr.DataArray, P: xr.DataArray):
    exp = const.pc_r_d/const.pc_cp_d
    out = T * (const.p0/P)**exp
    return out