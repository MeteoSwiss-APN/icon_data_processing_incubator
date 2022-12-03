# Third-party
import cfgrib
import numpy as np
import yaml
from operators.omega_slope import omega_slope
from operators.time_operators import time_rate
from cfgrib import abc
import typing as T

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
    with open("./productDefinitionTemplate.yml", "r") as stream:
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


def mymessage_set(self, item: str, value: T.Any) -> None:
    arr = isinstance(value, (np.ndarray, T.Sequence)) and not isinstance(value, str)
    if arr:
        eccodes.codes_set_array(self.codes_id, item, value)
    else:
        try:
            eccodes.codes_set(self.codes_id, item, value)
        except eccodes.CodesInternalError:
            pass


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
cfgrib.messages.Message.message_set = mymessage_set


class ifs_data_loader:
    """Class for loading data from ifs and convert conventions to COSMO."""

    def __init__(self, field_mapping_file: str):
        with open(field_mapping_file) as f:
            self._field_map = yaml.safe_load(f)

    def open_ifs_to_cosmo(self, datafile: str, fields: list[str]):
        ds = {}

        read_keys = ["pv", "NV"]
        read_keys += [
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
    ds_out["SSR"] = time_rate(
        ds["SSR"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["SSR"].attrs = ds["SSR"].attrs
    ds_out["SSHF"] = time_rate(
        ds["SSHF"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )
    ds_out["SSHF"].attrs = ds["SSHF"].attrs
    ds_out["EWSS"] = time_rate(
        ds["EWSS"].isel(step=slice(istep - 1, istep + 1)), np.timedelta64(1, "s")
    )

    ds_out["EWSS"].attrs = ds["EWSS"].attrs

    ds_out["OMEGA_SLOPE"] = omega_slope(
        ds["PS"].isel(step=istep), ds["ETADOT"].isel(step=istep), ds["ak"], ds["bk"]
    ).isel(hybrid=slice(39, 61))

    return ds_out
