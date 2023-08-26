# Third-party
import xarray as xr

# First-party
from idpi.operators.atmo import pv_sw
from idpi.operators.atmo import qv_pvp


def relhum(qv, t, p, clipping=True):
    attrs = t.attrs.copy()
    max = 100 if clipping else None

    return xr.DataArray((100.0 * qv / qv_pvp(pv_sw(t), p)).clip(0, max), attrs=attrs)
