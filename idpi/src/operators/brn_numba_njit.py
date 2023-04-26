"""algorithm for BRN operator."""
import numpy as np
import xarray as xr
from operators.destagger import destagger
from operators.thetav import fthetav, fthetav_np
from numba import njit
from numba import prange

pc_g = 9.80665
pc_r_d = 287.05  # Gas constant for dry air [J kg-1 K-1]
pc_r_v = 461.51  # Gas constant for water vapour[J kg-1 K-1]
pc_cp_d = 1005.0  # Specific heat capacity of dry air at 0 deg C and constant pressure [J kg-1 K-1]

# Derived quantities
pc_rvd = pc_r_v / pc_r_d
pc_rdocp = pc_r_d / pc_cp_d
pc_rvd_o = pc_rvd - 1.0

# Reference surface pressure for computation of potential temperature
p0 = 1.0e5

@njit
def fbrn(p, t, qv, u, v, hhl, hsurf):
    """Bulk Richardson Number (BRN)."""
    nlevels = p.shape[0]
    
    thetav = (p0 / p) ** pc_rdocp * t * (1.0 + (pc_rvd_o * qv / (1.0 - qv)))

    thetav_sum = np.sum(thetav, axis=0) 

    nlevels_xr = np.arange(nlevels, 0, -1)

    u_ = np.copy(u)
    v_ = np.copy(v)
    u_[:,:,1:] = ((u[:,:,0:-1] + u[:,:,1:]) * np.float32(0.5))
    v_[:,1:,:] = (v[:,0:-1,:] + v[:,1:,:]) * np.float32(0.5)
    hfl = (hhl[0:-1,:,:] + hhl[1:,:,:]) * np.float32(0.5)

    nlevels_xr_bc = np.expand_dims(np.expand_dims(nlevels_xr,-1),-1)

    brn = pc_g * (hfl-hsurf) * (thetav - thetav[-1,:,:]) * nlevels_xr_bc /  \
        (thetav_sum * (u_**2 + v_**2))

    return brn
