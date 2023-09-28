# Third-party
from numpy.testing import assert_allclose

# First-party
from idpi import grib_decoder
from idpi.operators import regrid
from idpi.operators.hzerocl import fhzerocl


def test_regrid(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    reader = grib_decoder.GribReader([cdatafile, datafile])
    ds = reader.load_cosmo_data(["T", "HHL"])

    hzerocl = fhzerocl(ds["T"], ds["HHL"], extrapolate=True)
    out_regrid_target = "swiss,549500,149500,650500,250500,1000,1000"
    dst = regrid.RegularGrid.parse_regrid_operator(out_regrid_target)
    hzerocl.attrs["geography"] = ds["HHL"].geography
    observed = regrid.regrid(hzerocl, dst, regrid.Resampling.bilinear)

    fx_ds = fieldextra("regrid", out_regrid_target=out_regrid_target)
    expected = fx_ds["HZEROCL"]

    assert_allclose(observed, expected, rtol=5e-4)
