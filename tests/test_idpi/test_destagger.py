# Third-party
from numpy.testing import assert_allclose

# First-party
from idpi import grib_decoder
from idpi.operators.destagger import destagger


def test_destagger(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ref_grid = grib_decoder.load_grid_reference("HHL", [cdatafile])
    ds = grib_decoder.load_cosmo_data(
        ref_grid,
        ["U", "V", "HHL"],
        [datafile, cdatafile],
    )

    u = destagger(ds["U"], "x")
    v = destagger(ds["V"], "y")
    hfl = destagger(ds["HHL"].isel(time=0), "z")

    fs_ds = fieldextra("destagger")

    assert_allclose(fs_ds["U"], u, rtol=1e-12, atol=1e-9)
    assert_allclose(fs_ds["V"], v, rtol=1e-12, atol=1e-9)
    assert_allclose(fs_ds["HFL"], hfl, rtol=1e-12, atol=1e-9)

    assert u.origin["x"] == 0.0
    assert v.origin["y"] == 0.0
    assert hfl.origin["z"] == 0.0
