import logging
import os
import pathlib

import cfgrib
import cfgrib.xarray_to_grib
import eccodes
import operators.flexpart as flx
import param_parser as pp
import xarray as xr
import yaml
from definitions import root_dir

logger = logging.getLogger(__name__)


def write_to_grib(filename, ds, param_db):

    with open((pathlib.Path(root_dir) / "share" / "field_mappings.yml").resolve()) as f:
        field_map = yaml.safe_load(f)

    with open(filename, "ab") as f:
        for field in ds:

            # cfgrib tries to extend_dims to cover all coordinates, but
            # Xarray complains that valid_time is a variable (not scalar) and can not extend dims
            ds[field] = ds[field].drop_vars("valid_time")

            # eccodes will ensure consistency between, for example GRIB_name, and the rules made in the eccodes
            # concept, name.def. It is difficult to ensure that in the internal metadata of the datasets,
            # and additionally not needed, since the paramId already fully identifies the parameter
            ds[field].attrs.pop("GRIB_cfName", None)
            ds[field].attrs.pop("GRIB_cfVarName", None)
            ds[field].attrs.pop("GRIB_name", None)
            ds[field].attrs.pop("GRIB_shortName", None)

            # if "cosmo" in field_map[field] and "paramId" in field_map[field]["cosmo"]:
            #     ds[field].attrs["GRIB_paramId"] = field_map[field]["cosmo"]["paramId"]

            paramId = ds[field].GRIB_paramId
            # ds[field].attrs["GRIB_centre"] = 78

            # ds[field].attrs["GRIB_subCentre"] = 0
            # TODO need to support SDOR
            # if field == "SDOR":
            ds[field].attrs["GRIB_centre"] = 'ecmf'
                # continue

            jScansPositively = (
                ds[field].coords["latitude"].values[-1]
                > ds[field].coords["latitude"].values[0]
            )
            iScansPositively = (
                ds[field].coords["longitude"].values[-1]
                > ds[field].coords["longitude"].values[0]
            )
            ds[field].attrs["GRIB_scanningMode"] = int(
                (not iScansPositively) * 1 + 2 * (jScansPositively)
            )
            ds[field].attrs["GRIB_jScansPositively"] = jScansPositively * 1

            if paramId in param_db and "units" in param_db[paramId]:
                ds[field].attrs["GRIB_units"] = param_db[paramId]["units"]
            else:
                logger.warning(
                    f"No units defined in eccodes definitions for field {field}. Using state units defined as: {ds[field].attrs['GRIB_units']}"
                )

            if ds[field].GRIB_edition == 1:
                ds[field].attrs["GRIB_edition"] = 2
                # Forecast [https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-3.shtml]
                ds[field].attrs["GRIB_typeOfGeneratingProcess"] = 2

            # TODO impose all compulsary keys
            if "GRIB_productDefinitionTemplateNumber" not in ds[field].attrs:
                ds[field].attrs["GRIB_productDefinitionTemplateNumber"] = 0

            # TODO do a proper grib1 to grib2 conversion
            if (
                paramId in param_db
                and "typeOfStatisticalProcessing" in param_db[paramId]["params"]
            ):
                ds[field].attrs["GRIB_productDefinitionTemplateNumber"] = 8

            if (
                "GRIB_typeOfStatisticalProcessing" in ds[field].attrs
                and ds[field].attrs["GRIB_productDefinitionTemplateNumber"] != 8
            ):
                raise RuntimeError(
                    "can not set typeOfStatisticalProcessing with a default productDefinitionTemplateNumber"
                    + ds[field].attrs["GRIB_productDefinitionTemplateNumer"]
                    + field
                )
            if (
                paramId in param_db
                and "typeOfStatisticalProcessing" in param_db[paramId]["params"]
                and ds[field].GRIB_productDefinitionTemplateNumber == 0
            ):
                raise RuntimeError(
                    "can not set typeOfStatisticalProcessing with a default productDefinitionTemplateNumber"
                )

            # TODO remove read only values
            for key in ["gridDefinitionDescription", "section4Length"]:
                ds[field].attrs.pop("GRIB_" + key, None)

            # TODO Why 153 ?
            # https://github.com/COSMO-ORG/eccodes-cosmo-resources/blob/master/definitions/grib2/localConcepts/edzw/modelName.def
            ds[field].attrs["GRIB_generatingProcessIdentifier"] = 153
            ds[field].attrs["GRIB_bitsPerValue"] = 16

            cfgrib.xarray_to_grib.canonical_dataarray_to_grib(
                ds[field],
                f,
                template_path="/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/samples/COSMO_GRIB2_default.tmpl",
            )


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
        "ASOB_S",
        "ASHFL_S",
        "EWSS",
        "NSSS",
    )

    loader = flx.ifs_data_loader(
        (pathlib.Path(root_dir) / "share" / "field_mappings.yml").resolve()
    )
    ds = flx.load_flexpart_data(constants + inputf, loader, datafile)

    for h in range(3, 10, 3):
        datafile = datadir + f"/efsf00{h:02d}0000"
        newds = flx.load_flexpart_data(inputf, loader, datafile)

        for field in newds:
            ds[field] = xr.concat([ds[field], newds[field]], dim="step")

    flx.append_pv(ds)

    eccodes.codes_set_definitions_path(cosmo_gpath)
    ds_out = {}
    for field in ("FIS", "FR_LAND", "SDOR"):
        ds_out[field] = ds[field]

    write_to_grib("flexpart_out.grib", ds_out, param_db)

    for i in range(1, 4):
        h = i * 3

        ds_out = flx.fflexpart(ds, i)
        print("i........ ", i)
        write_to_grib("flexpart_out.grib", ds_out, param_db)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_flexpart()
