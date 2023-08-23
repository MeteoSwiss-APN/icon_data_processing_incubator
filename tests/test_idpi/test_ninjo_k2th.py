# Third-party
from numpy.testing import assert_allclose

# First-party
import idpi.products.ninjo_k2th as ninjo
from idpi import grib_decoder


def test_ninjo_k2th(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ref_grid = grib_decoder.load_grid_reference("HHL", [cdatafile])
    ds = grib_decoder.load_cosmo_data(
        ref_grid,
        ["U", "V", "W", "P", "T", "QV", "QC", "QI", "HHL"],
        [datafile, cdatafile],
    )

    observed_mean, observed_at_theta = ninjo.ninjo_k2th(
        ds["U"],
        ds["V"],
        ds["W"],
        ds["T"],
        ds["P"],
        ds["QV"],
        ds["QC"],
        ds["QI"],
        ds["HHL"],
    )

    fs_ds = fieldextra("ninjo_k2th")

    assert_allclose(
        fs_ds["POT_VORTIC_MEAN"].isel(z_1=0),
        observed_mean,
        atol=1e-6,
    )

    assert_allclose(
        fs_ds["POT_VORTIC_AT_THETA"],
        observed_at_theta["pot_vortic"],
        atol=1e-9,
        rtol=1e-5,
    )

    assert_allclose(
        fs_ds["P"],
        observed_at_theta["p"],
        atol=1e-3,
        rtol=1e-4,
    )

    assert_allclose(
        fs_ds["U"],
        observed_at_theta["u"],
        atol=1e-9,
        rtol=1e-5,
    )

    assert_allclose(
        fs_ds["V"],
        observed_at_theta["v"],
        atol=5e-4,
        rtol=1e-4,
    )
