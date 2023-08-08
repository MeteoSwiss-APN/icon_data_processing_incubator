# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

# First-party
import idpi.operators.time_operators as time_ops
from idpi import grib_decoder


@pytest.fixture
def data_dir():
    return Path("/project/s83c/rz+/icon_data_processing_incubator/data/temporal")


def test_delta(data_dir, fieldextra):
    steps = np.arange(0, 16, 3)
    dd, hh = np.divmod(steps, 24)
    datafiles = [data_dir / f"lfff{d:02d}{h:02d}0000" for d, h in zip(dd, hh)]

    ds = grib_decoder.load_cosmo_data(["TOT_PREC"], datafiles, ref_param="TOT_PREC")

    tot_prec_03h = time_ops.delta(ds["TOT_PREC"], np.timedelta64(3, "h"))

    # Negative values are replaced by zero as these are due to numerical inaccuracies.
    cond = np.logical_or(tot_prec_03h > 0.0, tot_prec_03h.isnull())
    observed = tot_prec_03h.where(cond, 0.0)

    fx_ds_h = fieldextra(
        "time_ops_delta",
        hh=hh.tolist(),
        conf_files={
            "inputi": data_dir / "lfff<DDHH>0000",
            "inputc": data_dir / "lfff00000000c",
            "output": "<HH>_time_ops_delta.nc",
        },
    )

    expected = xr.concat([fx_ds["tot_prec_03h"] for fx_ds in fx_ds_h], dim="time")

    assert_allclose(observed, expected.transpose("epsd_1", "time", ...))


def test_max(data_dir, fieldextra):
    steps = np.arange(34)
    dd, hh = np.divmod(steps, 24)
    datafiles = [data_dir / f"lfff{d:02d}{h:02d}0000" for d, h in zip(dd, hh)]

    ds = grib_decoder.load_cosmo_data(["VMAX_10M"], datafiles, ref_param="VMAX_10M")

    f = ds["VMAX_10M"]
    vmax_10m_24h = time_ops.max(f.where(f.time > 0), np.timedelta64(24, "h"))

    # Negative values are replaced by zero as these are due to numerical inaccuracies.
    cond = np.logical_or(vmax_10m_24h > 0.0, vmax_10m_24h.isnull())
    observed = vmax_10m_24h.where(cond, 0.0).sel(time=steps[::3], z=10)

    fx_ds_h = fieldextra(
        "time_ops_max",
        hh=steps[::3],
        conf_files={
            "inputi": data_dir / "lfff<DDHH>0000",
            "inputc": data_dir / "lfff00000000c",
            "output": "<HH>_time_ops_max.nc",
        },
    )

    expected = xr.concat([fx_ds["vmax_10m_24h"] for fx_ds in fx_ds_h], dim="time")

    assert_allclose(observed, expected.transpose("epsd_1", "time", ...))
