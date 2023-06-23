"""Flexpart operators."""

# Third-party
import cfgrib  # type: ignore
import numpy as np
import xarray as xr
import yaml
from importlib.resources import files

# First-party
from idpi.operators.omega_slope import omega_slope
from idpi.operators.time_operators import time_rate
from idpi import grib_decoder


class ifs_data_loader:
    """Class for loading data from ifs and convert conventions to COSMO."""

    def __init__(self):
        """Initialize the data loader.

        Args:
            field_mapping_file: mappings between IFS and internal var names

        """

        mapping_path = files("idpi.data").joinpath("field_mappings.yml")
        self._field_map = yaml.safe_load(mapping_path.open())

    def open_ifs_to_cosmo(
        self, datafile: str, fields: list[str], load_pv: bool = False
    ):
        """Load IFS data in a dictionary where the keys are COSMO variables.

        IFS and COSMO use different shortNames. IFS are lower case,
        while COSMO are upper case. In order to have a more homogeneous data management
        in idpi and operators, we convert the keys from IFS to COSMO conventions.
        Additionally it applies unit conversions from IFS to COSMO.
        """
        ds = {}

        fields_ = [self._field_map[f]["ifs"]["name"] for f in fields]

        ifs_multi_ds = grib_decoder.load_data(fields_, [datafile], ref_param = "t")

        for f in fields:
            ds[f] = ifs_multi_ds[self._field_map[f]["ifs"]["name"]]

            if "cosmo" in self._field_map[f]:
                ufact = self._field_map[f]["cosmo"].get("unit_factor")

                if ufact:
                    ds[f] *= ufact

        return ds



def load_flexpart_data(fields, loader, datafile):
    ds = loader.open_ifs_to_cosmo(datafile, fields)
    append_pv_raw(ds)

    ds["U"] = ds["U"].sel(hybrid=slice(40, 137))
    ds["V"] = ds["V"].sel(hybrid=slice(40, 137))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1, 137))
    ds["T"] = ds["T"].sel(hybrid=slice(40, 137))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40, 137))

    return ds


def append_pv_raw(ds):
    """Compute ak, bk (weights that define the vertical coordinate) from pv."""
    NV = ds["U"].NV

    ds["ak"] = xr.DataArray(
        ds["U"].pv[0 : int(NV / 2)], dims=("hybrid_pv")
    ).assign_coords(
        {
            "hybrid_pv": np.append(
                ds["ETADOT"].hybrid.data, [len(ds["ETADOT"].hybrid) + 1]
            ),
            "step": ds["ETADOT"].step.values[0]
        }
    )
    ds["bk"] = xr.DataArray(
        ds["U"].pv[int(NV / 2) : NV], dims=("hybrid_pv")
    ).assign_coords(
        {
            "hybrid_pv": np.append(
                ds["ETADOT"].hybrid.data, [len(ds["ETADOT"].hybrid) + 1]
            ),
            "step": ds["ETADOT"].step.values[0]
        }
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

    ds_out["OMEGA"] = omega_slope(
        ds["PS"].isel(step=istep),
        ds["ETADOT"].isel(step=istep),
        ds["ak"].isel(step=istep),
        ds["bk"].isel(step=istep),
    ).isel(hybrid=slice(39, 137))

    return ds_out
