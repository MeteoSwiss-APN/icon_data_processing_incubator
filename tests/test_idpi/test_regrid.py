# Third-party
from numpy.testing import assert_allclose

# First-party
from idpi.operators import regrid
from idpi import grib_decoder
from idpi.operators.hzerocl import fhzerocl


def test_regrid(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    reader = grib_decoder.GribReader([cdatafile, datafile])
    ds = reader.load_cosmo_data(["T", "HHL"])

    hzerocl = fhzerocl(ds["T"], ds["HHL"], extrapolate=True)
    dst = regrid.RegularGrid.parse_regrid_operator(
        "swiss,479500,69500,840500,300500,1000,1000"
    )
    hzerocl.attrs["geography"] = ds["HHL"].geography
    observed = regrid.regrid(hzerocl, dst, regrid.Resampling.bilinear)

    fx_ds = fieldextra("regrid")
    expected = fx_ds["HZEROCL"]

    assert_allclose(observed, expected, rtol=2e-6)
