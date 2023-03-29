import itertools
import logging
import os
import pathlib
import typing
from typing import Any

import cfgrib
import cfgrib.xarray_to_grib
import click
import eccodes
import numpy as np
import operators.flexpart as flx
import xarray as xr
from cfgrib import cfmessage, messages
from definitions import root_dir

logger = logging.getLogger(__name__)

# This is a copy of canonical_dataarray_to_grib with modifications marked as
# HACK CFGRIB


def canonical_dataarray_to_grib(
    data_var: xr.DataArray,
    file: typing.IO,
    default_grib_keys: dict[str, Any] = cfgrib.xarray_to_grib.DEFAULT_GRIB_KEYS,
    **kwargs,
):
    """Write a ``xr.DataArray`` in *canonical* form to a GRIB file."""
    grib_keys: dict[str, str] = {}

    # validate Dataset keys, DataArray names, and attr keys/values
    detected_keys, suggested_keys = cfgrib.xarray_to_grib.detect_grib_keys(
        data_var, default_grib_keys, grib_keys
    )
    merged_grib_keys = cfgrib.xarray_to_grib.merge_grib_keys(
        grib_keys, detected_keys, suggested_keys
    )
    merged_grib_keys["missingValue"] = messages.MISSING_VAUE_INDICATOR

    if "gridType" not in merged_grib_keys:
        raise ValueError("required grib_key 'gridType' not passed nor auto-detected")

    template_message = cfgrib.xarray_to_grib.make_template_message(
        merged_grib_keys, **kwargs
    )

    coords_names, data_var = cfgrib.xarray_to_grib.expand_dims(data_var)

    header_coords_values = [list(data_var.coords[name].values) for name in coords_names]

    for items in itertools.product(*header_coords_values):
        select = {n: v for n, v in zip(coords_names, items)}
        field_values = data_var.sel(**select).values.flat[:]

        # Missing values handling
        invalid_field_values = np.logical_not(np.isfinite(field_values))

        # There's no need to save a message full of missing values
        if invalid_field_values.all():
            continue

        missing_value = merged_grib_keys.get(
            "GRIB_missingValue", messages.MISSING_VAUE_INDICATOR
        )
        field_values[invalid_field_values] = missing_value

        message = cfmessage.CfMessage.from_message(template_message)
        for coord_name, coord_value in zip(coords_names, items):
            if coord_name in cfgrib.xarray_to_grib.ALL_TYPE_OF_LEVELS:
                coord_name = "level"
                # HACK CFGRIB
                # level is not a primary grib key. Eccodes will transform it into the equivalent pair (or more)
                # scaleFactorOfFirstFixedSurface, scaledValueOfFirstFixedSurface
                # By default, eccodes will set then a scaleFactorOfFirstFixedSurface of 2.
                # Flexpart can not deal with any other value than 0.
                message["scaleFactorOfFirstFixedSurface"] = 0
                message["scaledValueOfFirstFixedSurface"] = coord_value
            else:
                message[coord_name] = coord_value

        if invalid_field_values.any():
            message["bitmapPresent"] = 1
        message["missingValue"] = missing_value

        # OPTIMIZE: convert to list because Message.message_set doesn't support np.ndarray
        message["values"] = field_values.tolist()

        message.write(file)


def write_to_grib(filename: str, ds: xr.Dataset, sample_file: str):
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

            if ds[field].attrs["GRIB_longitudeOfFirstGridPointInDegrees"] > 180:
                # Setting longitudeOfFirstGridPointInDegrees has no effect, therefore we need to setlongitudeOfFirstGridPoint
                ds[field].attrs["GRIB_longitudeOfFirstGridPoint"] = (
                    ds[field].attrs["GRIB_longitudeOfFirstGridPointInDegrees"] * 1e6
                    - 360 * 1e6
                )
            if ds[field].attrs["GRIB_longitudeOfLastGridPointInDegrees"] > 180:
                # Setting longitudeOfLastGridPointInDegrees has no effect, therefore we need to setlongitudeOfLastGridPoint
                ds[field].attrs["GRIB_longitudeOfLastGridPoint"] = (
                    ds[field].attrs["GRIB_longitudeOfLastGridPointInDegrees"] * 1e6
                    - 360 * 1e6
                )

            ds[field].attrs["GRIB_localTablesVersion"] = 4

            ds[field].attrs["GRIB_subCentre"] = 0
            ds[field].attrs["GRIB_centre"] = "ecmf"

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

            if ds[field].GRIB_edition == 1:
                ds[field].attrs["GRIB_edition"] = 2
                # Forecast [https://www.nco.ncep.noaa.gov/pmb/docs/grib2/grib2_doc/grib2_table4-3.shtml]
                ds[field].attrs["GRIB_typeOfGeneratingProcess"] = 2

            # Neccessary for grib1 input data -> grib2
            if "GRIB_productDefinitionTemplateNumber" not in ds[field].attrs:
                ds[field].attrs["GRIB_productDefinitionTemplateNumber"] = 0

            if (
                "GRIB_typeOfStatisticalProcessing" in ds[field].attrs
                and ds[field].attrs["GRIB_productDefinitionTemplateNumber"] != 8
            ):
                raise RuntimeError(
                    "can not set typeOfStatisticalProcessing with a default productDefinitionTemplateNumber"
                    + ds[field].attrs["GRIB_productDefinitionTemplateNumer"]
                    + field
                )

            # TODO remove read only values
            for key in ["gridDefinitionDescription", "section4Length"]:
                ds[field].attrs.pop("GRIB_" + key, None)

            # TODO Why 153 ?
            # https://github.com/COSMO-ORG/eccodes-cosmo-resources/blob/master/definitions/grib2/localConcepts/edzw/modelName.def
            ds[field].attrs["GRIB_generatingProcessIdentifier"] = 153
            ds[field].attrs["GRIB_bitsPerValue"] = 16

            # cfgrib.xarray_to_grib.canonical_dataarray_to_grib(
            canonical_dataarray_to_grib(
                ds[field],
                f,
                template_path=sample_file,
            )


@click.command()
@click.option(
    "--data_dir",
    required=True,
    type=str,
    help="directory that contains input grib files",
)
@click.option("--data_prefix", type=str, default="efsf", help="grib data files prefix")
@click.option(
    "--rhour",
    type=int,
    required=True,
    help="hour of reference datetime",
)
@click.option("--rdate", required=True, type=str, help="date of reference datetime")
@click.option("--nsteps", required=True, type=int, help="number of lead times")
def run_flexpart(data_dir, data_prefix, rhour, rdate, nsteps):
    gpaths = os.environ["GRIB_DEFINITION_PATH"].split(":")
    cosmo_gpath = [p for p in gpaths if "eccodes-cosmo-resources" in p][0]
    eccodes_gpath = [p for p in gpaths if "eccodes-cosmo-resources" not in p][0]
    eccodes.codes_set_definitions_path(eccodes_gpath)
    sample_file = cosmo_gpath + "/../samples/COSMO_GRIB2_default.tmpl"

    datafile = data_dir + f"/{data_prefix}00{rhour:02d}0000"
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

    for h in range(1, nsteps):
        datafile = data_dir + f"/{data_prefix}00{h+rhour:02d}0000"
        newds = flx.load_flexpart_data(inputf, loader, datafile)

        for field in newds:
            ds[field] = xr.concat([ds[field], newds[field]], dim="step")

    # Hack to add pv to all fields (required by flexpart)
    for f in ds:
        for t in range(ds[f].coords["step"].size):
            ds[f].attrs["GRIB_PVPresent"] = 1
            ds[f].attrs["GRIB_pv"] = xr.concat(
                [ds["ak"].isel(step=t), ds["bk"].isel(step=t)], dim="hybrid_pv"
            ).data

    eccodes.codes_set_definitions_path(cosmo_gpath)
    ds_const = {}
    for field in ("FIS", "FR_LAND", "SDOR"):
        ds_const[field] = ds[field]

    for i in range(1, nsteps):
        ds_out = flx.fflexpart(ds, i)
        for a in ds_const:
            ds_out[a] = ds_const[a]

        print("i........ ", i)
        write_to_grib(f"dispf{rdate}{i:02d}", ds_out, sample_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_flexpart()
