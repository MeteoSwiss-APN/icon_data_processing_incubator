# Third-party
from numpy.testing import assert_allclose

# First-party
import idpi.operators.thetav as mthetav
from idpi import grib_decoder


def test_thetav(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"

    ref_grid = grib_decoder.load_grid_reference("P", [datafile])
    ds = grib_decoder.load_cosmo_data(ref_grid, ["P", "T", "QV"], [datafile])

    thetav = mthetav.fthetav(ds["P"], ds["T"], ds["QV"])

    fs_ds = fieldextra("THETAV")

    assert_allclose(fs_ds["THETA_V"], thetav, rtol=1e-6)
