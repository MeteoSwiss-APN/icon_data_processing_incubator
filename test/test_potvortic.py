import os
import shutil
import subprocess

from idpi import grib_decoder
import jinja2
import numpy as np
import idpi.operators.pot_vortic as pv
import xarray as xr


def test_pv():
    datadir = "/project/s83c/rz+/icon_data_processing_incubator/data/SWISS"
    datafile = datadir + "/lfff00000000.ch"
    cdatafile = datadir + "/lfff00000000c.ch"

    ds = {}
    grib_decoder.load_data(
        ds, ["U", "V", "W", "P", "T", "QV", "QC", "QI"], datafile, chunk_size=None
    )
    grib_decoder.load_data(ds, ["HHL", "HSURF"], cdatafile, chunk_size=None)

    ds = {
        k: v.rename(generalVerticalLayer="z") if "generalVerticalLayer" in v.dims else v
        for k, v in ds.items()
    }

    potv = pv.fpotvortic(
        ds["U"],
        ds["V"],
        ds["W"],
        ds["P"],
        ds["T"],
        ds["HHL"],
        ds["QV"],
        ds["QC"],
        ds["QI"],
    )

    conf_files = {
        "inputi": datadir + "/lfff<DDHH>0000.ch",
        "inputc": datadir + "/lfff00000000c.ch",
        "output": "<HH>_POT_VORTIC.nc",
    }
    out_file = "00_POT_VORTIC.nc"
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
    template = templateEnv.get_template("./test_POT_VORTIC.nl")
    outputText = template.render(file=conf_files, ready_flags=tmpdir)

    with open(tmpdir + "/test_POT_VORTIC.nl", "w") as nl_file:
        nl_file.write(outputText)

    # remove output and product files
    for afile in [out_file] + prodfiles:
        if os.path.exists(cwd + "/" + afile):
            os.remove(cwd + "/" + afile)

    subprocess.run([executable, tmpdir + "/test_POT_VORTIC.nl "], check=True)

    fs_ds = xr.open_dataset("00_POT_VORTIC.nc")
    brn_ref = fs_ds["BRN"].rename(
        {"x_1": "x", "y_1": "y", "z_1": "generalVerticalLayer"}
    )

    assert np.allclose(brn_ref, potv, rtol=3e-3, atol=5e-2, equal_nan=True)


if __name__ == "__main__":
    test_pv()
