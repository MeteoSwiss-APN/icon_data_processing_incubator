from .. import constants as const
import xarray as xr


def f_rho_tot(
    T: xr.DataArray,
    P: xr.DataArray,
    QV: xr.DataArray,
    QC: xr.DataArray,
    QI: xr.DataArray | int = 0,
    QP: xr.DataArray | int = 0,
) -> xr.DataArray:
    """
    Total density of air mixture (perfect gas law, pressure as sum of partial pressures). Result is in [kg/m**3].

    Args:
        T (xr.DataArray): Temperature [Kelvin]
        P (xr.DataArray): Pressure [Pascal]
        QV (xr.DataArray): Specific humidity [kg/kg]
        QC (xr.DataArray): Specific cloud water content [kg/kg]
        QI (xr.DataArray, optional): Specific cloud ice content [kg/kg]. Defaults to 0.
        QP (xr.DataArray, optional): Specific precipitable components content [kg/kg]. Defaults to 0.
    """
    return P / (const.pc_r_d * T * (1.0 + const.pc_rvd_o * QV - QC - QI - QP))
