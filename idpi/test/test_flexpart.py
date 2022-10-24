import os
import shutil
import subprocess
import yaml
import jinja2
import numpy as np
import xarray as xr
import cfgrib

def get_da(field, dss):
    for ds in dss:
        if field in ds:
            return ds[field]

def load_data(fields, field_mapping, datafile):
    ds = {}

    dss = cfgrib.open_datasets(
        datafile,
        backend_kwargs={"read_keys": ["typeOfLevel", "gridType"]},
        encode_cf=("time", "geography", "vertical"),
    )

    for f in fields:
        ds[f] = get_da(field_mapping[f]['ifs_name'], dss)
    ds["U"] = ds["U"].sel(hybrid=slice(40,60))
    ds["V"] = ds["V"].sel(hybrid=slice(40,60))
    ds["ETADOT"] = ds["ETADOT"].sel(hybrid=slice(1,60))
    ds["T"] = ds["T"].sel(hybrid=slice(40,60))
    ds["QV"] = ds["QV"].sel(hybrid=slice(40,60))

    return ds

def write_to_grib(filename, ds):

    with open(filename, 'ab') as f:
        for field in ds: 
            ds[field].attrs['GRIB_centre'] = 78
            cfgrib.xarray_to_grib.canonical_dataarray_to_grib(ds[field], 
            f, template_path='/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/samples/COSMO_GRIB2_default.tmpl')

def test_brn():
    with open("field_mappings.yml") as f:
        field_map = yaml.safe_load(f)
    
    datadir = "/scratch/cosuna/flexpart-input/data/"
    datafile = datadir + "/efsf00000000"
    cdatafile = datadir + "/lfff00000000c.ch"

    ds = {}


    datafile = datadir + "/efsf00000000"


    constants = ('FIS', 'FR_LAND', 'SDOR')
    input = ("ETADOT", "T", "QV", "U", "V", "PS", "U_10M", "V_10M", "T_2M", "TD_2M", "CLCT", "W_SNOW", "TOT_CON", "TOT_GSP", "SSR",
    "SSHF", "EWSS", "NSSS")

    ds = load_data(constants+input, field_map, datafile)

    for field in ds:
        f = ds[field]
        ds[field] = f.reindex(latitude=list(reversed(f.latitude)))

    for h in range(3,10,3):
        datafile = datadir + f"/efsf00{h:02d}0000"
        newds = load_data(input,field_map, datafile)

        for field in newds:
            ds[field] = xr.concat([ds[field], newds[field]], dim="step")

    ds['TOT_CON'] = ds['TOT_CON']*1000
    ds['TOT_GSP'] = ds['TOT_GSP']*1000

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

    assert np.allclose(fs_ds["FIS"], ds["FIS"], rtol=3e-5, atol=5e-2, equal_nan=True)
    assert np.allclose(fs_ds["FR_LAND"], ds["FR_LAND"], rtol=3e-5, atol=5e-2, equal_nan=True)
    assert np.allclose(fs_ds["SDOR"], ds["SDOR"], rtol=3e-5, atol=5e-2, equal_nan=True)

    os.environ['GRIB_DEFINITION_PATH'] = "/project/g110/spack-install/tsa/cosmo-eccodes-definitions/2.19.0.7/gcc/zcuyy4uduizdpxfzqmxg6bc74p2skdfp/cosmoDefinitions/definitions/:/scratch/cosuna/spack-install/tsa/eccodes/2.19.0/gcc/viigacbsqxbbcid22hjvijrrcihebyeh/share/eccodes/definitions/"

    for i in range(1,4):
        h = i*3

        fs_ds = xr.open_dataset(f"{h:02d}_flexpart.nc")

        assert np.allclose(fs_ds["U"], ds["U"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["V"], ds["V"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["T"], ds["T"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["QV"], ds["QV"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["PS"], ds["PS"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["U_10M"], ds["U_10M"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["V_10M"], ds["V_10M"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["T_2M"], ds["T_2M"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["TD_2M"], ds["TD_2M"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["CLCT"], ds["CLCT"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["W_SNOW"], ds["W_SNOW"].isel(step=i), rtol=3e-5, atol=5e-2, equal_nan=True)

        ds_out = {}
        for field in ('U','V','T','QV','PS','U_10M','V_10M','T_2M','TD_2M','CLCT','W_SNOW'):
            ds_out[field] = ds[field].isel(step=i)
        ds_out["TOT_CON"] = (ds['TOT_CON'].isel(step=i) - ds['TOT_CON'].isel(step=i-1))*0.333333
        ds_out["TOT_GSP"] = (ds['TOT_GSP'].isel(step=i) - ds['TOT_GSP'].isel(step=i-1))*0.333333
        ds_out["SSR"] = (ds['SSR'].isel(step=i) - ds['SSR'].isel(step=i-1)) / (3600 *3)
        ds_out["SSHF"] = (ds['SSHF'].isel(step=i) - ds['SSHF'].isel(step=i-1)) / (3600 *3)
        ds_out["EWSS"] = (ds['EWSS'].isel(step=i) - ds['EWSS'].isel(step=i-1)) / (3600 *3)

        assert np.allclose(fs_ds["TOT_CON"], ds_out["TOT_CON"], rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["TOT_GSP"], ds_out["TOT_GSP"], rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["SSR"], ds_out["SSR"], rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["SSHF"], ds_out["SSHF"], rtol=3e-5, atol=5e-2, equal_nan=True)
        assert np.allclose(fs_ds["EWSS"], ds_out["EWSS"], rtol=3e-5, atol=5e-2, equal_nan=True)

        write_to_grib(f"flexpart_out.grib", ds_out)
    #assert np.allclose(fs_ds["NSSS"], ds["NSSS"].isel(step=1), rtol=3e-5, atol=5e-2, equal_nan=True)

    #assert np.allclose(fs_ds["ETADOT"], etadot, rtol=3e-5, atol=5e-2, equal_nan=True)

        #to_grib(TOT_CON.to_dataset(), f'out_{i:02d}.grib', grib_keys={'edition': 2}) 

if __name__ == "__main__":
    test_brn()
