# Third-party
from numpy.testing import assert_allclose

# First-party
import idpi.operators.theta as mtheta
from idpi.grib_decoder import GribReader


def test_theta(data_dir, fieldextra):
    datafile = data_dir / "COSMO-1E/1h/ml_sl/000/lfff00000000"
    reader = GribReader([datafile], ref_param="P")

    ds = reader.load_cosmo_data(["P", "T"])

    theta = mtheta.ftheta(ds["P"], ds["T"])

    fs_ds = fieldextra("THETA")

    assert_allclose(fs_ds["THETA"], theta, rtol=1e-6)
