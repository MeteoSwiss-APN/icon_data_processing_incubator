# Third-party
import cfgrib
import numpy as np
import xarray as xr
import yaml
from operators.omega_slope import omega_slope
from operators.time_operators import time_rate


class ifs_data_loader:
    """Class for loading data from ifs and convert conventions to COSMO."""

    def __init__(self, field_mapping_file: str):
        with open(field_mapping_file) as f:
            self._field_map = yaml.safe_load(f)

    def open_ifs_to_cosmo(self, datafile: str, fields: list[str]):
        ds = {}

        read_keys = ["pv", "NV"]
        ifs_multi_ds = cfgrib.open_datasets(
            datafile,
            backend_kwargs={"read_keys": read_keys},
            encode_cf=("time", "geography", "vertical"),
        )

        for f in fields:
            ds[f] = self._get_da(self._field_map[f]["ifs"]["name"], ifs_multi_ds)
            if "cosmo" in self._field_map[f]:
                ufact = self._field_map[f]["cosmo"].get("unit_factor")

                if ufact:
                    ds[f] *= ufact

        return ds

    def _get_da(self, field, dss):
        for ds in dss:
            if field in ds:
                return ds[field]

def load_flexpart_data(fields, loader, datafile):
    ds = loader.open_ifs_to_cosmo(datafile, fields)

    ds["U"] = ds["U"].sel(hybrid=slice(40, 60))
    ds["V"] = ds["V"].sel(hybrid=slice(40, 60))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1, 60))
    ds["T"] = ds["T"].sel(hybrid=slice(40, 60))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40, 60))

    return ds


def append_pv(ds):
    """Compute ak, bk (weights that define the vertical coordinate) from pv."""
    NV = ds["U"].GRIB_NV
    ds["ak"] = (
        xr.DataArray(ds["U"].GRIB_pv[0 : int(NV / 2)], dims=("hybrid"))
        .sel(hybrid=slice(0, 61))
        .assign_coords(
            {"hybrid": np.append(ds["ETADOT"].hybrid, [len(ds["ETADOT"].hybrid) + 1])}
        )
    )
    ds["bk"] = (
        xr.DataArray(ds["U"].GRIB_pv[int(NV / 2) : NV], dims=("hybrid"))
        .sel(hybrid=slice(0, 61))
        .assign_coords(
            {"hybrid": np.append(ds["ETADOT"].hybrid, [len(ds["ETADOT"].hybrid) + 1])}
        )
    )

def fflexpart(ds, istep):
    ds_out = {}
    for field in (
        "U",
        "V",
        "T",
        "QV",
        "PS",
        "U_10M",
        "V_10M",
        "T_2M",
        "TD_2M",
        "CLCT",
        "W_SNOW",
    ):
        ds_out[field] = ds[field].isel(step=istep)

    ds_out["TOT_CON"] = time_rate(
        ds["TOT_CON"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "h")
    )
    ds_out["TOT_CON"].attrs = ds["TOT_CON"].attrs
    ds_out["TOT_GSP"] = time_rate(
        ds["TOT_GSP"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "h")
    )

    ds_out["TOT_GSP"].attrs = ds["TOT_GSP"].attrs
    ds_out["ASOB_S"] = time_rate(
        ds["ASOB_S"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["ASOB_S"].attrs = ds["ASOB_S"].attrs
    ds_out["ASHFL_S"] = time_rate(
        ds["ASHFL_S"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["ASHFL_S"].attrs = ds["ASHFL_S"].attrs
    ds_out["EWSS"] = time_rate(
        ds["EWSS"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )

    ds_out["EWSS"].attrs = ds["EWSS"].attrs

    ds_out["OMEGA_SLOPE"] = omega_slope(
        ds["PS"].isel(step=istep), ds["ETADOT"].isel(step=istep), ds["ak"], ds["bk"]
    ).isel(hybrid=slice(39, 61))

    return ds_out
