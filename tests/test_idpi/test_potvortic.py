# Third-party
import numpy as np
from numpy.testing import assert_allclose

# First-party
import idpi.operators.pot_vortic as pv
from idpi.grib_decoder import GribReader
from idpi.operators.rho import f_rho_tot
from idpi.operators.theta import ftheta
from idpi.operators.total_diff import TotalDiff


def test_pv(data_dir, fieldextra):
    datafile = data_dir / "COSMO-1E/1h/ml_sl/000/lfff00000000"
    cdatafile = data_dir / "COSMO-1E/1h/const/000/lfff00000000c"

    reader = GribReader([cdatafile, datafile])
    ds = reader.load_fieldnames(["U", "V", "W", "P", "T", "QV", "QC", "QI", "HHL"])

    theta = ftheta(ds["P"], ds["T"])
    rho_tot = f_rho_tot(ds["T"], ds["P"], ds["QV"], ds["QC"], ds["QI"])

    geo = ds["HHL"].attrs["geography"]
    dlon = geo["iDirectionIncrementInDegrees"]
    dlat = geo["jDirectionIncrementInDegrees"]
    deg2rad = np.pi / 180

    total_diff = TotalDiff(dlon * deg2rad, dlat * deg2rad, ds["HHL"])

    observed = pv.fpotvortic(ds["U"], ds["V"], ds["W"], theta, rho_tot, total_diff)

    fs_ds = fieldextra("POT_VORTIC")

    assert_allclose(
        fs_ds["POT_VORTIC"],
        observed,
        rtol=1e-4,
        atol=1e-8,
    )
