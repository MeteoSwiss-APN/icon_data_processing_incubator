# Third-party
import pathlib
import typing as T

import cfgrib
import cfgrib.xarray_to_grib
import numpy as np
import xarray as xr
import yaml
from cfgrib import abc
from definitions import root_dir
from operators.omega_slope import omega_slope
from operators.time_operators import time_rate


def read_keys(
    first: abc.Field, keys: T.List[str], optional=False
) -> T.Dict[str, T.Any]:
    attributes = {}
    for key in keys:
        try:
            value = first[key]
            if value is not None:
                attributes["GRIB_" + key] = value
            elif not optional:
                raise RuntimeError("key not found", key)
        except Exception:
            if not optional:
                raise RuntimeError("key not found", key)
    return attributes


def myread_data_var_attrs(
    first: abc.Field, extra_keys: T.List[str]
) -> T.Dict[str, T.Any]:
    attributes = read_keys(first, extra_keys)

    if attributes["GRIB_edition"] == 1:
        # TODO support grib1
        return attributes

    if attributes.get("GRIB_PVPresent", 0) == 1:
        # TODO should we load pv for every field?
        rkeys = ["pv"]
        attributes |= read_keys(first, rkeys)

    # section 0
    rkeys = ["discipline"]
    #      master and local tables version
    rkeys += ["tablesVersion", "localTablesVersion"]
    # TODO check master version != 255

    rkeys += ["productionStatusOfProcessedData", "typeOfProcessedData"]

    # section 4
    rkeys += ["productDefinitionTemplateNumber"]

    attributes |= read_keys(first, rkeys)

    import yaml

    product_conf = None
    with open(
        (
            pathlib.Path(root_dir) / "share" / "./productDefinitionTemplate.yml"
        ).resolve(),
        "r",
    ) as stream:
        product_conf = yaml.safe_load(stream)

    prod_number = attributes["GRIB_productDefinitionTemplateNumber"]
    if prod_number not in product_conf["supported"]:
        raise RuntimeError(
            "productDefinitionTemplateNumber not supported: ", prod_number
        )
    prod_number_tag = product_conf["supported"][prod_number]

    def get_deps(prod_number_tag):
        res = []
        resopt = []
        for refid in product_conf["dependencies"][prod_number_tag]:
            if type(refid) is dict:
                if len(refid) > 1:
                    raise RuntimeError("multiple options in deps not supported")

                if "opt" in refid:
                    resopt += [refid["opt"]]
                elif "ref" in refid:
                    res += get_deps(refid["ref"])
                else:
                    raise RuntimeError("unkown key in dependency:", refid)
            else:
                res += [refid]

        return res, resopt

    deps, opt_deps = get_deps(prod_number_tag)
    attributes |= read_keys(first, deps)
    attributes |= read_keys(first, opt_deps, optional=True)

    return attributes


cfgrib.dataset.read_data_var_attrs = myread_data_var_attrs
cfgrib.dataset.EXTRA_DATA_ATTRIBUTES_KEYS = [
    "shortName",
    "units",
    "name",
    "cfName",
    "cfVarName",
    "missingValue",
    # "totalNumber",
    # TODO check these two
    # "numberOfDirections",
    # "numberOfFrequencies",
    "NV",
    "gridDefinitionDescription",
]

cfgrib.xarray_to_grib.MESSAGE_DEFINITION_KEYS = [
    # for the GRIB 2 sample we must set this before setting 'totalNumber'
    "productDefinitionTemplateNumber",
    # We need to set the centre before the units
    "centre",
    # NO IDEA WHAT IS GOING ON HERE: saving regular_ll_msl.grib results in the wrong `paramId`
    #   unless `units` is set before some other unknown key, this happens at random and only in
    #   Python 3.5, so it must be linked to dict key stability.
    "units",
]


def expand_dims(data_var: xr.DataArray) -> T.Tuple[T.List[str], xr.DataArray]:
    coords_names = []  # type: T.List[str]
    for coord_name in (
        cfgrib.dataset.ALL_HEADER_DIMS
        + cfgrib.xarray_to_grib.ALL_TYPE_OF_LEVELS
        + cfgrib.dataset.ALL_REF_TIME_KEYS
    ):
        # Needed for this fix https://github.com/ecmwf/cfgrib/pull/324
        if (
            coord_name in data_var.coords
            and data_var.coords[coord_name].size == 1
            and coord_name not in data_var.dims
        ):
            data_var = data_var.expand_dims(coord_name)
        if coord_name in data_var.dims:
            coords_names.append(coord_name)
    return coords_names, data_var


cfgrib.xarray_to_grib.expand_dims = expand_dims


class ifs_data_loader:
    """Class for loading data from ifs and convert conventions to COSMO."""

    def __init__(self, field_mapping_file: str):
        with open(field_mapping_file) as f:
            self._field_map = yaml.safe_load(f)

    def open_ifs_to_cosmo(self, datafile: str, fields: list[str]):
        ds = {}

        read_keys = [
            "edition",
            "productDefinitionTemplateNumber",
            "uvRelativeToGrid",
            "resolutionAndComponentFlags",
            "section4Length",
            "PVPresent",
            "productionStatusOfProcessedData",
        ]

        ifs_multi_ds = cfgrib.open_datasets(
            datafile,
            backend_kwargs={"read_keys": read_keys},
            encode_cf=("time", "geography", "vertical"),
        )

        for f in fields:
            ds[f] = self._get_da(self._field_map[f]["ifs"]["name"], ifs_multi_ds)
            if ds[f].GRIB_edition == 1:
                # Somehow grib1 loads a perturbationNumber=0 which sets a 'number' coordinate.
                # That will force in cfgrib setting the productDefinitionTemplateNumber to 1
                # https://github.com/ecmwf/cfgrib/blob/27071067bcdd7505b1abbcb2cea282cf23b36598/cfgrib/xarray_to_grib.py#L123
                ds[f] = ds[f].drop_vars("number")

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
    append_pv_raw(ds)

    ds["U"] = ds["U"].sel(hybrid=slice(40, 137))
    ds["V"] = ds["V"].sel(hybrid=slice(40, 137))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1, 137))
    ds["T"] = ds["T"].sel(hybrid=slice(40, 137))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40, 137))

    return ds


def append_pv_raw(ds):
    """Compute ak, bk (weights that define the vertical coordinate) from pv."""
    NV = ds["U"].GRIB_NV

    ds["ak"] = xr.DataArray(
        ds["U"].GRIB_pv[0 : int(NV / 2)], dims=("hybrid_pv")
    ).assign_coords(
        {
            "hybrid_pv": np.append(
                ds["ETADOT"].hybrid.data, [len(ds["ETADOT"].hybrid) + 1]
            ),
            "time": ds["ETADOT"].time,
            "step": ds["ETADOT"].step,
        }
    )
    ds["bk"] = xr.DataArray(
        ds["U"].GRIB_pv[int(NV / 2) : NV], dims=("hybrid_pv")
    ).assign_coords(
        {
            "hybrid_pv": np.append(
                ds["ETADOT"].hybrid.data, [len(ds["ETADOT"].hybrid) + 1]
            ),
            "time": ds["ETADOT"].time,
            "step": ds["ETADOT"].step,
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
        ds["PS"].isel(step=istep), ds["ETADOT"].isel(step=istep), ds["ak"], ds["bk"]
    ).isel(hybrid=slice(39, 137))

    return ds_out
