from idpi import grib_decoder
import numpy as np
import idpi.operators.pot_vortic as pv
import xarray as xr

from utils import fx_context, data_dir


def test_pv():
    with fx_context("PV") as context:
        datafile = data_dir + "/lfff00000000.ch"
        cdatafile = context.nl_const_input

        ds: dict[str, xr.DataArray] = {}
        grib_decoder.load_data(
            ds, ["U", "V", "W", "P", "T", "QV", "QC", "QI"], datafile, chunk_size=None
        )
        grib_decoder.load_data(ds, ["HHL", "HSURF"], cdatafile, chunk_size=None)

        # rename vertical dimension
        ds = {
            k: v.rename(generalVerticalLayer="z")
            if "generalVerticalLayer" in v.dims
            else v
            for k, v in ds.items()
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

        fs_ds = context.output
        pv_ref = (
            fs_ds["POT_VORTIC"]
            .rename({"x_1": "x", "y_1": "y", "z_1": "z"})
            .squeeze(drop=True)
        )
        # discard the (undefined) border
        pv_ref = pv_ref.isel(x=slice(1, -1), y=slice(1, -1))

        assert np.allclose(pv_ref, potv, rtol=3e-3, atol=5e-2, equal_nan=True)


if __name__ == "__main__":
    test_pv()
