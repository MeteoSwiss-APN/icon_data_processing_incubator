# Third-party
from numpy.testing import assert_allclose

# First-party
import idpi.operators.lateral_operators as lat_ops
from idpi import grib_decoder
from idpi.operators.hzerocl import fhzerocl


def test_fill_undef(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ds = grib_decoder.load_cosmo_data(
        ["T", "HHL"],
        [datafile, cdatafile],
    )

    hzerocl = fhzerocl(ds["T"], ds["HHL"])

    observed = lat_ops.fill_undef(hzerocl, 10, 0.3)

    fx_ds = fieldextra("lat_ops_fill_undef")
    expected = fx_ds["HZEROCL"]

    assert_allclose(observed, expected, rtol=2e-6)


def test_disk_avg(data_dir, fieldextra):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ds = grib_decoder.load_cosmo_data(
        ["T", "HHL"],
        [datafile, cdatafile],
    )

    hzerocl = fhzerocl(ds["T"], ds["HHL"])

    observed = lat_ops.disk_avg(hzerocl, 10)

    fx_ds = fieldextra("lat_ops_disk_avg")
    expected = fx_ds["HZEROCL"]

    assert_allclose(observed, expected, rtol=2e-6)
