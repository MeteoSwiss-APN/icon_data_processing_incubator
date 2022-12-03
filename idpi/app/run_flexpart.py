import os
import pathlib

import cfgrib
import cfgrib.xarray_to_grib
import eccodes
import param_parser as pp
import xarray as xr
import yaml
from definitions import root_dir
from operators.flexpart import fflexpart, ifs_data_loader, load_flexpart_data


def write_to_grib(filename, ds, param_db):

    # "U",
    # "V",
    # "T",
    # "QV",
    # "PS",
    # "U_10M",
    # "V_10M",
    # "T_2M",
    # "TD_2M",
    # "CLCT",
    # "W_SNOW",

    with open((pathlib.Path(root_dir) / "share" / "field_mappings.yml").resolve()) as f:
        field_map = yaml.safe_load(f)

    with open(filename, "ab") as f:
        n = 0

        for field in ds:
            # TODO undo
            if n > 11:
                continue

            # TODO need to support SDOR
            if field == "SDOR":
                continue

            ds[field].attrs.pop("GRIB_cfName", None)

            if "paramId" in field_map[field]["cosmo"]:
                ds[field].attrs["GRIB_paramId"] = field_map[field]["cosmo"]["paramId"]

            paramId = ds[field].GRIB_paramId

            # TODO remove those or check
            # WHy is NV set by Fieldextra for FIS ?
            ds[field].attrs["GRIB_centre"] = 78
            ds[field].attrs["GRIB_subCentre"] = 0
            ds[field].attrs["GRIB_scanningMode"] = 0
            # TODO set this from numpy layout
            ds[field].attrs["GRIB_jScansPositively"] = 0
            print("U", field, ds[field].GRIB_units)
            if ds[field].attrs["GRIB_units"] == "m**2 s**-2":
                # problem otherwise grib_set_values[3] lengthOfTimeRange (type=long) failed: Key/value not found
                ds[field].attrs["GRIB_units"] = "m"
            print("param_", param_db[paramId], paramId)
            if "units" in param_db[paramId]:
                ds[field].attrs["GRIB_units"] = param_db[paramId]["units"]
            else:
                print("NO UNIT", field, ds[field].attrs["GRIB_units"])
                print(param_db[paramId])

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


def run_flexpart():
    gpaths = os.environ["GRIB_DEFINITION_PATH"].split(":")
    cosmo_gpath = [p for p in gpaths if "cosmoDefinitions" in p][0]
    eccodes_gpath = [p for p in gpaths if "cosmoDefinitions" not in p][0]
    eccodes.codes_set_definitions_path(eccodes_gpath)
    param_db = pp.param_db(cosmo_gpath + "/grib2")

    datadir = "/project/s83c/rz+/icon_data_processing_incubator/data/flexpart/"
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

    loader = ifs_data_loader(
        (pathlib.Path(root_dir) / "share" / "field_mappings.yml").resolve()
    )
    ds = load_flexpart_data(constants + inputf, loader, datafile)

    for h in range(3, 10, 3):
        datafile = datadir + f"/efsf00{h:02d}0000"
        newds = load_flexpart_data(inputf, loader, datafile)

        for field in newds:
            ds[field] = xr.concat([ds[field], newds[field]], dim="step")

    eccodes.codes_set_definitions_path(":".join([cosmo_gpath, eccodes_gpath]))

    ds_out = {}
    for field in ("FIS", "FR_LAND", "SDOR"):
        ds_out[field] = ds[field]

    write_to_grib("flexpart_out.grib", ds_out, param_db)

    for i in range(1, 4):
        h = i * 3

        ds_out = fflexpart(ds, i)
        print("i........ ", i)
        write_to_grib("flexpart_out.grib", ds_out, param_db)


if __name__ == "__main__":
    run_flexpart()
