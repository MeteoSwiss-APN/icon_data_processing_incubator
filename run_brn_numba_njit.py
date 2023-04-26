import os
import shutil
import subprocess

import grib_decoder
import jinja2
import numpy as np
import operators.brn_numba_njit as mbrn
import xarray as xr
import time
from numba import config, njit, threading_layer, prange
import earthkit.data
import statistics

def test_brn():
    datadir = "/scratch/e1000/meteoswiss/scratch/cosuna/data/COSMO1"
    datafile = datadir + "/lfff00000000"
    cdatafile = datadir + "/lfff00000000c"
   
    ds = {}
    grib_decoder.load_data(ds, ["P", "T", "QV", "U", "V"], datafile, chunk_size=None)
    grib_decoder.load_data(ds, ["HHL", "HSURF"], cdatafile, chunk_size=None)
   
    brn = mbrn.fbrn(
        ds["P"].data, ds["T"].data, ds["QV"].data, ds["U"].data, ds["V"].data, ds["HHL"].data, ds["HSURF"].data
    )
 
    times = []
    for c in range(10):
        start = time.time()
        brn = mbrn.fbrn(
            ds["P"].data, ds["T"].data, ds["QV"].data, ds["U"].data, ds["V"].data, ds["HHL"].data, ds["HSURF"].data        
        )
        stop = time.time()
        times.append(stop-start)

    print("time:", statistics.median(times))
    # print("Threading layer chosen: %s" % threading_layer())
    
if __name__ == "__main__":
    test_brn()
