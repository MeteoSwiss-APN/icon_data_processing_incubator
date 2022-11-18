import os
import typing as T

import cfgrib
import cfgrib.xarray_to_grib
import eccodes
import numpy as np
import xarray as xr
import yaml
from cfgrib import abc


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


def get_da(field, dss):
    for ds in dss:
        if field in ds:
            return ds[field]


def load_data(fields, field_mapping, datafile):
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
    dss = cfgrib.open_datasets(
        datafile,
        backend_kwargs={"read_keys": read_keys},
        encode_cf=("time", "geography", "vertical"),
    )

    for f in fields:
        ds[f] = get_da(field_mapping[f]["ifs_name"], dss)
    ds["U"] = ds["U"].sel(hybrid=slice(40, 60))
    ds["V"] = ds["V"].sel(hybrid=slice(40, 60))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1, 60))
    ds["T"] = ds["T"].sel(hybrid=slice(40, 60))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40, 60))

    return ds


def write_to_grib(filename, ds):

    with open(filename, "ab") as f:
        n = 0

        for field in ds:
            # TODO undo
            if n > 5:
                continue

            # TODO need to support SDOR
            if field == "SDOR":
                continue
            # TODO remove those or check
            # WHy is NV set by Fieldextra for FIS ?
            ds[field].attrs["GRIB_centre"] = 78
            ds[field].attrs["GRIB_subCentre"] = 0
            ds[field].attrs["GRIB_scanningMode"] = 0
            # TODO set this from numpy layout
            ds[field].attrs["GRIB_jScansPositively"] = 0
            if ds[field].attrs["GRIB_units"] == "m**2 s**-2":
                # problem otherwise grib_set_values[3] lengthOfTimeRange (type=long) failed: Key/value not found
                ds[field].attrs["GRIB_units"] = "m"

            if ds[field].GRIB_edition == 1:
                # Somehow grib1 loads a perturbationNumber=0 which sets a 'number' coordinate.
                # That will force in cfgrib setting the productDefinitionTemplateNumber to 1
                # https://github.com/ecmwf/cfgrib/blob/27071067bcdd7505b1abbcb2cea282cf23b36598/cfgrib/xarray_to_grib.py#L123
                ds[field] = ds[field].drop_vars("number")
                ds[field].attrs["GRIB_edition"] = 2
                # Forecast [https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-3.shtml]
                ds[field].attrs["GRIB_typeOfGeneratingProcess"] = 2
                ds[field].attrs["GRIB_productDefinitionTemplateNumber"] = 0
            # TODO Why 153 ?
            # https://github.com/COSMO-ORG/eccodes-cosmo-resources/blob/master/definitions/grib2/localConcepts/edzw/modelName.def
            ds[field].attrs["GRIB_generatingProcessIdentifier"] = 153
            ds[field].attrs["GRIB_bitsPerValue"] = 16
            cfgrib.xarray_to_grib.canonical_dataarray_to_grib(
                ds[field],
                f,
                template_path="/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/samples/COSMO_GRIB2_default.tmpl",
            )
            n += 1


# similar to the subtract.accumulate but permute the order of the operans of the diff
# TODO implement as a ufunc
def cumdiff(A, axis):
    r = np.empty(np.shape(A))
    t = 0  # op = the ufunc being applied to A's  elements
    for i in range(np.shape(A)[axis]):
        t = np.take(A, i, axis) - t

        slices = []
        for dim in range(A.ndim):
            if dim == axis:
                slices.append(slice(i, i + 1))
            else:
                slices.append(slice(None))

        r[tuple(slices)] = np.expand_dims(t, axis=t.ndim)
    return r


def flexpart(ds, istep):
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

    ds_out["TOT_CON"] = (
        (ds["TOT_CON"].isel(step=istep) - ds["TOT_CON"].isel(step=istep - 1))
        * 0.333333
        * 1000
    )
    ds_out["TOT_CON"].attrs = ds["TOT_CON"].attrs
    ds_out["TOT_GSP"] = (
        (ds["TOT_GSP"].isel(step=istep) - ds["TOT_GSP"].isel(step=istep - 1))
        * 0.333333
        * 1000
    )
    ds_out["TOT_GSP"].attrs = ds["TOT_GSP"].attrs
    ds_out["SSR"] = (ds["SSR"].isel(step=istep) - ds["SSR"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["SSR"].attrs = ds["SSR"].attrs
    ds_out["SSHF"] = (ds["SSHF"].isel(step=istep) - ds["SSHF"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["SSHF"].attrs = ds["SSHF"].attrs
    ds_out["EWSS"] = (ds["EWSS"].isel(step=istep) - ds["EWSS"].isel(step=istep - 1)) / (
        3600 * 3
    )
    ds_out["EWSS"].attrs = ds["EWSS"].attrs

    surface_pressure_ref = 101325.0

    ak1 = ds["ak"][1:].assign_coords(
        {"hybrid": ds["ak"][{"hybrid": slice(0, -1)}].hybrid}
    )
    bk1 = ds["bk"][1:].assign_coords(
        {"hybrid": ds["bk"][{"hybrid": slice(0, -1)}].hybrid}
    )

    omega_slope = (
        2.0
        * ds["PS"].isel(step=istep)
        * ds["ETADOT"].isel(step=istep)
        * ((ak1 - ds["ak"][0:-1]) / ds["PS"].isel(step=istep) + bk1 - ds["bk"][0:-1])
        / ((ak1 - ds["ak"][0:-1]) / surface_pressure_ref + bk1 - ds["bk"][0:-1])
    )

    ds_out["OMEGA_SLOPE"] = omega_slope.reduce(cumdiff, dim="hybrid").isel(
        hybrid=slice(39, 61)
    )

    return ds_out


def run_flexpart():
    with open("field_mappings.yml") as f:
        field_map = yaml.safe_load(f)

    datadir = "/scratch/cosuna/flexpart-input/newdata/"
    datafile = datadir + "/efsf00000000"
    constants = ("FIS", "FR_LAND", "SDOR")
    inputf = (
        "ETADOT",
        "T",
        "QV",
        "U",
        "V",
        "PS",
        "U_10M",
        "V_10M",
        "T_2M",
        "TD_2M",
        "CLCT",
        "W_SNOW",
        "TOT_CON",
        "TOT_GSP",
        "SSR",
        "SSHF",
        "EWSS",
        "NSSS",
    )

    ds = load_data(constants + inputf, field_map, datafile)

    for h in range(3, 10, 3):
        datafile = datadir + f"/efsf00{h:02d}0000"
        newds = load_data(inputf, field_map, datafile)

        for field in newds:
            ds[field] = xr.concat([ds[field], newds[field]], dim="step")

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

    gpaths = os.environ["GRIB_DEFINITION_PATH"].split(":")
    gpaths = [path for path in gpaths if "cosmo-eccodes-definitions" not in path]
    os.environ["GRIB_DEFINITION_PATH"] = ":".join(gpaths)

    ds_out = {}
    for field in ("FIS", "FR_LAND", "SDOR"):
        ds_out[field] = ds[field]

    write_to_grib("flexpart_out.grib", ds_out)

    for i in range(1, 4):
        h = i * 3

        ds_out = flexpart(ds, i)
        write_to_grib("flexpart_out.grib", ds_out)


if __name__ == "__main__":
    run_flexpart()
