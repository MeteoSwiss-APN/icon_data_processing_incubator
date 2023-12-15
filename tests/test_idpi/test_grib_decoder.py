# Third-party
import xarray as xr

# First-party
from idpi import grib_decoder


def test_save(data_dir, tmp_path):
    datafile = data_dir / "lfff00000000c.ch"

    reader = grib_decoder.GribReader.from_files([datafile], ref_param="HHL")
    ds = reader.load_fieldnames(["HHL"])

    outfile = tmp_path / "output.grib"
    with outfile.open("wb") as f:
        grib_decoder.save(ds["HHL"], f)

    reader = grib_decoder.GribReader.from_files([outfile], ref_param="HHL")
    ds_new = reader.load_fieldnames(["HHL"])

    ds["HHL"].attrs.pop("metadata")
    ds_new["HHL"].attrs.pop("metadata")

    xr.testing.assert_identical(ds["HHL"], ds_new["HHL"])
