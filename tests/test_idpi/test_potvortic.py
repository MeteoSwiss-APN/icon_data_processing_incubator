# Third-party
import numpy as np
import xarray as xr

# First-party
import idpi.operators.pot_vortic as pv
from idpi import grib_decoder


def test_pv(data_dir, fieldextra, grib_defs):
    datafile = data_dir / "lfff00000000.ch"
    cdatafile = data_dir / "lfff00000000c.ch"

    ds: dict[str, xr.DataArray] = {}
    grib_decoder.load_data(
        ds, ["U", "V", "W", "P", "T", "QV", "QC", "QI"], datafile, chunk_size=None
    )
    grib_decoder.load_data(ds, ["HHL", "HSURF"], cdatafile, chunk_size=None)

    # rename vertical dimension
    keys = {"generalVerticalLayer", "generalVertical"}
    ds = {
        k: v.rename({key: "z" for key in keys if key in v.dims}) for k, v in ds.items()
    }

    # ensure same size for all input arrays
    ds = {k: v.isel(z=slice(80)) if "z" in v.dims else v for k, v in ds.items()}

    potv = pv.fpotvortic(
        ds["U"],
        ds["V"],
        ds["W"],
        ds["P"],
        ds["T"],
        ds["HHL"],
        ds["QV"],
        ds["QC"],
        ds["QI"],
    )

    fs_ds = fieldextra("POT_VORTIC")

    pv_ref = (
        fs_ds.rename({"x_1": "x", "y_1": "y", "z_1": "z"})
        .squeeze(drop=True)
        .isel(x=slice(1, -1), y=slice(1, -1))  # discard the (undefined) border
    )
    assert np.allclose(pv_ref, potv, rtol=3e-3, atol=5e-2, equal_nan=True)


if __name__ == "__main__":
    test_pv()  # type: ignore[call-arg]
