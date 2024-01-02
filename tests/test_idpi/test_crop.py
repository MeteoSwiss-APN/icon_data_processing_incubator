# Third-party
from numpy.testing import assert_equal

# First-party
from idpi.grib_decoder import GribReader
from idpi.operators import crop


def test_crop(data_dir):
    cdatafile = data_dir / "COSMO-1E/1h/const/000/lfff00000000c"

    reader = GribReader.from_files([cdatafile])
    ds = reader.load_fieldnames(["HHL"])
    hhl = ds["HHL"]

    observed = crop.crop(hhl, (0, 1, 0, 2))

    expected_values = hhl.isel(x=[0, 1], y=[0, 1, 2]).values
    expected_geography = hhl.geography | {"Ni": 2, "Nj": 3}
    # eccodes normalises the value of the longitude
    expected_geography["longitudeOfFirstGridPointInDegrees"] += 360

    assert_equal(observed.values, expected_values)
    assert observed.geography == expected_geography
    assert observed.parameter == hhl.parameter
