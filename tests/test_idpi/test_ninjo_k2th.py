# Third-party
from numpy.testing import assert_allclose
import pytest

# First-party
import idpi.products.ninjo_k2th as ninjo
from idpi.grib_decoder import GribReader
from idpi.data_cache import DataCache
from idpi.data_source import DataSource


@pytest.fixture
def data(work_dir):
    datafile = "/store/s83/osm/KENDA-1/ANA22/det/laf2022020900"
    source = DataSource(datafiles=[datafile])
    fields = {
        "inputi": [(p, "ml") for p in ("U", "V", "W", "P", "T", "QV", "QC", "QI")],
        "inputc": [("HHL", "ml"), ("HSURF", "sfc"), ("FIS", "sfc")],
    }
    files = {
        "inputi": "lfff<ddhh>0000",
        "inputc": "lfff00000000c",
    }
    cache = DataCache(cache_dir=work_dir, fields=fields, files=files)
    cache.populate(source)
    reader = GribReader(source, ref_param=("HHL", "ml"))
    yield reader, cache
    # cache.clear()


def test_ninjo_k2th(data, fieldextra):
    reader, cache = data

    ds = reader.load_fieldnames(["U", "V", "W", "P", "T", "QV", "QC", "QI", "HHL"])
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

    conf_files = cache.conf_files | {"output": "<hh>_outfile.nc"}
    fs_ds = fieldextra("ninjo_k2th", conf_files=conf_files)

    assert_allclose(
        fs_ds["POT_VORTIC_MEAN"].isel(z_1=0),
        observed_mean,
        atol=3e-6,
    )

    assert_allclose(
        fs_ds["POT_VORTIC_AT_THETA"],
        observed_at_theta["pot_vortic"],
        atol=3e-7,
        rtol=1e-5,
    )

    assert_allclose(
        fs_ds["P"],
        observed_at_theta["p"],
        atol=1e-2,
        rtol=1e-3,
    )

    assert_allclose(
        fs_ds["U"],
        observed_at_theta["u"],
        atol=1e-8,
        rtol=1e-5,
    )

    assert_allclose(
        fs_ds["V"],
        observed_at_theta["v"],
        atol=1e-8,
        rtol=1e-4,
    )
