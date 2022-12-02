# Standard library
import os
import shutil
import subprocess

# Third-party
import cfgrib
import cfgrib.xarray_to_grib
import eccodes
import jinja2
import numpy as np
import xarray as xr
import yaml
from operators.flexpart import fflexpart
import eccodes

def get_da(field, dss):
    for ds in dss:
        if field in ds:
            return ds[field]


def load_data(fields, field_mapping, datafile):
    ds = {}

    read_keys = ["pv", "NV"]
    dss = cfgrib.open_datasets(
        datafile,
        backend_kwargs={"read_keys": read_keys},
        encode_cf=("time", "geography", "vertical"),
    )

    for f in fields:
        ds[f] = get_da(field_mapping[f]["ifs"]["name"], dss)
    ds["U"] = ds["U"].sel(hybrid=slice(40, 60))
    ds["V"] = ds["V"].sel(hybrid=slice(40, 60))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1, 60))
    ds["T"] = ds["T"].sel(hybrid=slice(40, 60))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40, 60))

    return ds


def test_flexpart():
    os.environ[
        "GRIB_DEFINITION_PATH"
    ] = "/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/definitions/:/scratch/cosuna/spack-install/tsa/eccodes/2.19.0/gcc/viigacbsqxbbcid22hjvijrrcihebyeh/share/eccodes/definitions/"
    gpaths = os.environ["GRIB_DEFINITION_PATH"].split(":")
    eccodes_gpath = [p for p in gpaths if "cosmoDefinitions" not in p][0]
    eccodes.codes_set_definitions_path(eccodes_gpath)

    with open(
        "/scratch/cosuna/flexpart-input/icon_data_processing_incubator/idpi/test/field_mappings.yml"
    ) as f:
        field_map = yaml.safe_load(f)

    def path():
        return '/scratch/cosuna/spack-install/tsa/eccodes/2.19.0/gcc/viigacbsqxbbcid22hjvijrrcihebyeh/share/eccodes/definitions/'
    print(eccodes.codes_definition_path())

    os.environ['ECCODES_DEFINITION_PATH'] = '/scratch/cosuna/spack-install/tsa/eccodes/2.19.0/gcc/viigacbsqxbbcid22hjvijrrcihebyeh/share/eccodes/definitions/'
    eccodes.codes_definition_path = path

    print(eccodes.codes_definition_path())

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

    conf_files = {
        "inputi": datadir + "/efsf00<HH>0000",
        "inputc": datadir + "/efsf00000000",
        "output": "<HH>_flexpart.nc",
    }
    out_file = "00_flexpart.nc"
    prodfiles = ["fieldextra.diagnostic"]

    testdir = os.path.dirname(os.path.realpath(__file__))
    tmpdir = testdir + "/tmp"
    cwd = os.getcwd()

    executable = "/project/s83c/fieldextra/tsa/bin/fieldextra_gnu_opt_omp"

    # create the tmp dir
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)

    templateLoader = jinja2.FileSystemLoader(searchpath=testdir + "/fe_templates")
    templateEnv = jinja2.Environment(loader=templateLoader)
    template = templateEnv.get_template("./test_flexpart.nl")
    outputText = template.render(file=conf_files, ready_flags=tmpdir)

    with open(tmpdir + "/test_flexpart.nl", "w") as nl_file:
        nl_file.write(outputText)

    # remove output and product files
    for afile in [out_file] + prodfiles:
        if os.path.exists(cwd + "/" + afile):
            os.remove(cwd + "/" + afile)

    subprocess.run([executable, tmpdir + "/test_flexpart.nl "], check=True)

    fs_ds = xr.open_dataset("00_flexpart.nc")
    fs_ds_o = {}
    for f in ("FIS", "FR_LAND", "SDOR"):
        fs_ds_o[f] = fs_ds[f].isel(y_1=slice(None, None, -1))

    assert np.allclose(fs_ds_o["FIS"], ds["FIS"], rtol=3e-7, atol=5e-7, equal_nan=True)
    assert np.allclose(
        fs_ds_o["FR_LAND"], ds["FR_LAND"], rtol=3e-7, atol=5e-7, equal_nan=True
    )
    assert np.allclose(
        fs_ds_o["SDOR"], ds["SDOR"], rtol=3e-7, atol=5e-7, equal_nan=True
    )

    ds_out = {}
    for field in ("FIS", "FR_LAND", "SDOR"):
        ds_out[field] = ds[field]

    # Compute few steps of a 3 hourly data
    for i in range(1, 4):
        h = i * 3

        fs_ds = xr.open_dataset(f"{h:02d}_flexpart.nc")
        fs_ds_o = dict()

        # Invert the latitude order in FX netcdf
        for f in (
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
            "TOT_CON",
            "TOT_GSP",
            "SSR",
            "SSHF",
            "EWSS",
            "ETADOT",
        ):
            fs_ds_o[f] = fs_ds[f].isel(y_1=slice(None, None, -1))

        ds_out = fflexpart(ds, i)

        assert np.allclose(
            fs_ds_o["ETADOT"].transpose("y_1", "x_1", "z_1", "time").isel(time=0),
            ds_out["OMEGA_SLOPE"],
            rtol=3e-7,
            atol=5e-7,
            equal_nan=True,
        )
        assert np.allclose(
            fs_ds_o["U"], ds_out["U"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["V"], ds_out["V"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["T"], ds_out["T"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["QV"], ds_out["QV"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["PS"], ds_out["PS"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["U_10M"], ds_out["U_10M"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["V_10M"], ds_out["V_10M"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["T_2M"], ds_out["T_2M"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["TD_2M"], ds_out["TD_2M"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["CLCT"], ds_out["CLCT"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["W_SNOW"], ds_out["W_SNOW"], rtol=3e-7, atol=5e-7, equal_nan=True
        )

        assert np.allclose(
            fs_ds_o["TOT_CON"], ds_out["TOT_CON"], rtol=3e-6, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["TOT_GSP"], ds_out["TOT_GSP"], rtol=3e-6, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["SSR"], ds_out["SSR"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["SSHF"], ds_out["SSHF"], rtol=3e-7, atol=5e-7, equal_nan=True
        )
        assert np.allclose(
            fs_ds_o["EWSS"], ds_out["EWSS"], rtol=3e-7, atol=5e-7, equal_nan=True
        )


if __name__ == "__main__":
    test_flexpart()
