# Third-party
import pytest
from numpy.testing import assert_allclose

# First-party
from idpi import grib_decoder
from idpi.operators.hzerocl import fhzerocl


@pytest.mark.parametrize("extrapolate", [True, False])
def test_hzerocl(data_dir, fieldextra, extrapolate):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ds = grib_decoder.load_cosmo_data(
        ["T", "HHL"],
        [datafile, cdatafile],
    )

    hzerocl = fhzerocl(ds["T"], ds["HHL"], extrapolate)

    fs_ds = fieldextra(
        "hzerocl",
        h0cl_extrapolate=".true." if extrapolate else ".false.",
    )

    assert_allclose(
        fs_ds["HZEROCL"],
        hzerocl,
        rtol=5e-6,
        atol=1e-5,
        equal_nan=True,
    )
