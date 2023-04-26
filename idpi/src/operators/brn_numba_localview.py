"""algorithm for BRN operator."""
import numpy as np
import xarray as xr
from operators.destagger import destagger
from operators.thetav import fthetav, fthetav_np
from numba import njit
from numba import prange

pc_g = 9.80665

@njit(parallel=True)
def cpu_cumsum(data):
    output = np.zeros(data.shape)
    output[0,:,:] = data[0,:,:]

    for i in prange(data.shape[2]):
        for j in prange(data.shape[1]):
            for k in prange(1, data.shape[0]):
                output[k,j,i] = data[k,j,i] + output[k-1,j,i]
    return output

pc_r_d = 287.05  # Gas constant for dry air [J kg-1 K-1]
pc_r_v = 461.51  # Gas constant for water vapour[J kg-1 K-1]
pc_cp_d = 1005.0  # Specific heat capacity of dry air at 0 deg C and constant pressure [J kg-1 K-1]

# Derived quantities
pc_rvd = pc_r_v / pc_r_d
pc_rdocp = pc_r_d / pc_cp_d
pc_rvd_o = pc_rvd - 1.0

# Reference surface pressure for computation of potential temperature
p0 = 1.0e5

@njit(parallel=True, fastmath=True, nogil=True)
def fbrn(p, t, qv, u, v, hhl, hsurf):
    """Bulk Richardson Number (BRN)."""
    brn= np.zeros(t.shape)
    hsurf_bc = np.broadcast_to(hsurf,t.shape)
    thetav_sum = np.zeros(t.shape[1:])
  
    for k in range(t.shape[0]-1,-1,-1):
        for j in prange(t.shape[1]):
            for i in prange(t.shape[2]):
                thetav = (p0 / p[k,j,i]) ** pc_rdocp * t[k,j,i] * (1.0 + (pc_rvd_o * qv[k,j,i] / (1.0 - qv[k,j,i])))

                if k == t.shape[0]-1:
                    thetav_surf = thetav
                                      
                u_ = u[k,j,i]
                v_ = v[k,j,i]
                if i != 0:
                    u_ = (u[k,j,i-1] + u[k,j,i]) * np.float32(0.5)
                if j != 0:
                    v_ = (v[k,j-1,i] + v[k,j,i]) * np.float32(0.5)

                thetav_sum[j,i] = thetav_sum[j,i] + thetav

                brn[k,j,i] = pc_g*( (hhl[k,j,i] + hhl[k+1,j,i]) * np.float32(0.5) - hsurf_bc[k, j,i]) * (thetav - thetav_surf) * \
                    (t.shape[0] - k) / (u_**2 + v_**2 )

    for k in range(t.shape[0]-1,-1,-1):
        for j in prange(t.shape[1]):
            for i in prange(t.shape[2]):
                brn[k,j,i] /= thetav_sum[j,i]
                 
    return brn
