import xarray as xr

from . import mars, data_source, grib_decoder


def get_from_fdb(request: mars.Request) -> dict[str, xr.DataArray]:
    """Get model data from FDB.

    Parameters
    ----------
    request : mars.Request
        Request for data defined in the mars language.

    Raises
    ------
    ValueError
        if the request has a feature attribute.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dataset containing the requested data.

    """
    if request.feature is not None:
        raise ValueError("FDB does not support the feature attribute")
    source = data_source.DataSource()
    return grib_decoder.load(source, request)


def get_from_polytope(request: mars.Request) -> dict[str, xr.DataArray]:
    """Get model data from Polytope.

    Parameters
    ----------
    request : mars.Request
        Request for data defined in the mars language.

    Returns
    -------
    dict[str, xarray.DataArray]
        Dataset containing the requested data.

    """
    if request.feature is not None:
        collection = "mchgj"
    else:
        collection = "mch"
    source = data_source.DataSource(polytope_collection=collection)
    return grib_decoder.load(source, request)
