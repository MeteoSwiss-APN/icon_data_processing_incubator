"""Decoder for grib data."""
# Third-party
import cfgrib  # type: ignore
import eccodes
import earthkit.data
import xarray as xr
import numpy as np


def load_data(outds, fields, datafiles):
    fs = earthkit.data.from_source("file", datafiles)
    # hhl will serve as a reference of the grid for the mass point
    hhl = fs.sel(param="HHL", level=1)
    for field in fields:
        fsel = fs.sel(param=field)
        typeOfLevels = np.array(fsel.metadata("typeOfLevel"))
        if not np.all(typeOfLevels == typeOfLevels[0]):
            raise RuntimeError(
                "not homogeneous type of level" + str(typeOfLevels), typeOfLevels[0]
            )
        typeOfLevel = typeOfLevels[0]
        dx = (
            int(hhl.metadata("longitudeOfLastGridPoint")[0])
            - int(hhl.metadata("longitudeOfFirstGridPoint")[0])
        ) / int(hhl.metadata("Nx")[0])
        dy = (
            int(hhl.metadata("latitudeOfLastGridPoint")[0])
            - int(hhl.metadata("latitudeOfFirstGridPoint")[0])
        ) / int(hhl.metadata("Ny")[0])

        dx_vs_ref = (
            fsel.sel(level=1).metadata("longitudeOfFirstGridPoint")[0]
            - hhl.metadata("longitudeOfFirstGridPoint")[0]
        )
        dy_vs_ref = (
            fsel.sel(level=1).metadata("latitudeOfFirstGridPoint")[0]
            - hhl.metadata("latitudeOfFirstGridPoint")[0]
        )

        # the distance of the origin to the reference (HHL) can not be larger
        # than the distance between two contiguous cells, since it should be
        # half the distance between cells.
        if dx_vs_ref > dx or dy_vs_ref > dy:
            raise RuntimeError("wrong coordinate reference")

        orig = dict()
        orig["x"] = 0.5 if dx_vs_ref > 0 else 0
        orig["y"] = 0.5 if dy_vs_ref > 0 else 0

        if typeOfLevel == "generalVertical":
            orig["z"] = -0.5
        elif typeOfLevel == "generalVerticalLayer":
            orig["z"] = 0
        else:
            raise RuntimeError("Unsupported type of level")

        outds[field] = xr.DataArray(
            data=fsel.order_by("level").to_numpy(),
            dims=("z", "y", "x"),
            attrs={"origin": orig, "name": field, "vcoord_type": "ml"},
        )

    if any(field not in outds for field in fields):
        raise RuntimeError("Not all fields found in datafile", fields)
