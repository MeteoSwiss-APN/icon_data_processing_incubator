"""Vertical interpolation operators."""

# Third-party
import numpy as np
import xarray as xr

# First-party
from idpi.operators.support_operators import init_field_with_vcoord


def interpolate_k2p(field, mode, p_field, p_tc_values, p_tc_units):
    """Interpolate a field from model (k) levels to pressure coordinates.

    Example for vertical interpolation to isosurfaces of a target field,
    which is strictly monotonically decreasing with height.



    Parameters
    ----------
    field : xarray.DataArray
        field to interpolate (only typeOfLevel="z" is supported)
    mode : str
        interpolation algorithm, one of {"linear_in_p", "linear_in_lnp", "nearest_sfc"}
    p_field : xarray.DataArray
        pressure field on k levels in Pa
        (only typeOfLevel="z" is supported)
    p_tc_values : list of float
        pressure target coordinate values
    p_tc_units : str
        pressure target coordinate units

    Returns
    -------
    field_on_tc : xarray.DataArray
        field on target (i.e., pressure) coordinates

    """
    # TODO: check missing value consistency with GRIB2 (currently comparisons are
    #       done with np.nan)
    #       check that p_field is the pressure field, given in Pa (can only be done
    #       if attributes are consequently set)
    #       check that field and p_field are compatible (have the same
    #       dimensions and sizes)
    #       print warn message if result contains missing values

    # Initializations
    # ... supported interpolation modes
    interpolation_modes = ("linear_in_p", "linear_in_lnp", "nearest_sfc")
    if mode not in interpolation_modes:
        raise RuntimeError("interpolate_k2p: unknown mode", mode)
    # ... supported tc units and corresponding conversion factors to Pa
    p_tc_unit_conversions = dict(Pa=1.0, hPa=100.0)
    if p_tc_units not in p_tc_unit_conversions.keys():
        raise RuntimeError(
            "interpolate_k2p: unsupported value of p_tc_units", p_tc_units
        )
    # ... supported range of pressure tc values (in Pa)
    p_tc_min = 1.0
    p_tc_max = 120000.0
    # ... supported vertical coordinate type for field and p_field
    supported_origin = 0

    # Define vertical target coordinates (tc)
    tc_values = p_tc_values.copy()
    tc_values.sort(reverse=False)
    tc_factor = p_tc_unit_conversions[p_tc_units]
    tc_values = np.array(tc_values) * tc_factor
    if min(tc_values) < p_tc_min or max(tc_values) > p_tc_max:
        raise RuntimeError(
            "interpolate_k2p: target coordinate value out of range "
            "(must be in interval [",
            p_tc_min,
            ", ",
            p_tc_max,
            "]Pa)",
        )

    # Check that typeOfLevel is supported and equal for both field and p_field
    if supported_origin != field.origin["z"]:
        raise RuntimeError(
            "interpolate_k2p: field to interpolate must be defined for origin=",
            supported_origin,
        )
    if supported_origin != p_field.origin["z"]:
        raise RuntimeError(
            "interpolate_k2p: pressure field must be defined for typeOfLevel=",
            supported_origin,
        )
    # Check that dimensions are the same for field and p_field
    if field.dims != p_field.dims or field.size != p_field.size:
        raise RuntimeError(
            "interpolate_k2p: field and p_field must have equal dimensions and size"
        )

    # Prepare output field field_on_tc on target coordinates
    field_on_tc = init_field_with_vcoord(field, tc_values, np.nan)

    # Interpolate
    # ... prepare interpolation
    pkm1 = p_field.copy()
    pkm1[{"z": slice(1, None)}] = p_field[{"z": slice(0, -1)}]
    pkm1[{"z": 0}] = np.nan
    fkm1 = field.copy()
    fkm1[{"z": slice(1, None)}] = field[{"z": slice(0, -1)}]
    fkm1[{"z": 0}] = np.nan

    # ... loop through tc values
    for tc_idx, p0 in enumerate(tc_values):
        # ... find the 3d field where pressure is > p0 on level k
        # and was <= p0 on level k-1
        p2 = p_field.where((p_field > p0) & (pkm1 <= p0))
        # ... extract the index k of the vertical layer at which p2 adopts its minimum
        #     (corresponds to search from top of atmosphere to bottom)
        # ... note that if the condition above is not fulfilled, minind will
        # be set to k_top
        minind = p2.fillna(p_tc_max).argmin(dim=["z"])
        # ... extract pressure and field at level k
        p2 = p2[{"z": minind["z"]}]
        f2 = field[{"z": minind["z"]}]
        # ... extract pressure and field at level k-1
        # ... note that f1 and p1 are both undefined, if minind equals k_top
        f1 = fkm1[{"z": minind["z"]}]
        p1 = pkm1[{"z": minind["z"]}]

        # ... compute the interpolation weights
        if mode == "linear_in_p":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            # be undefined
            ratio = (p0 - p1) / (p2 - p1)

        if mode == "linear_in_lnp":
            # ... note that p1 is undefined, if minind equals k_top, so ratio will
            #  be undefined
            ratio = (np.log(p0) - np.log(p1)) / (np.log(p2) - np.log(p1))

        if mode == "nearest_sfc":
            # ... note that by construction, p2 is always defined;
            #     this operation sets ratio to 0 if p1 (and by construction also f1)
            #     is undefined; therefore, the interpolation formula below works
            #     correctly also in this case
            ratio = xr.where(np.abs(p0 - p1) >= np.abs(p0 - p2), 1.0, 0.0)

        # ... interpolate and update field_on_tc
        field_on_tc[{"z": tc_idx}] = (1.0 - ratio) * f1 + ratio * f2

    return field_on_tc


def interpolate_k2theta(field, mode, th_field, th_tc_values, th_tc_units, h_field):
    """Interpolate a field from model levels to potential temperature coordinates.

       Example for vertical interpolation to isosurfaces of a target field
       that is no monotonic function of height.

    Parameters
    ----------
    field : xarray.DataArray
        field to interpolate (only typeOfLevel="generalVerticalLayer" is supported)
    mode : str
        interpolation algorithm, one of {"low_fold", "high_fold","undef_fold"}
    th_field : xarray.DataArray
        potential temperature theta on k levels in K
        (only typeOfLevel="generalVerticalLayer" is supported)
    th_tc_values : list of float
        target coordinate values
    th_tc_units : str
        target coordinate units
    h_field : xarray.DataArray
        height on k levels (only typeOfLevel="generalVerticalLayer" is supported)

    Returns
    -------
    field_on_tc : xarray.DataArray
        field on target (i.e., theta) coordinates

    """
    # TODO: check missing value consistency with GRIB2
    #       (currently comparisons are done with np.nan)
    #       check that th_field is the theta field, given in K
    #       (can only be done if attributes are consequently set)
    #       check that field, th_field, and h_field are compatible
    #       print warn message if result contains missing values

    # ATTENTION: the attribute "positive" is not set for generalVerticalLayer
    #            we know that for COSMO it would be defined as positive:"down";
    #            for the time being,
    #            we explicitly use the height field on model mid layer
    #            surfaces as auxiliary field

    # Parameters
    # ... supported folding modes
    folding_modes = ("low_fold", "high_fold", "undef_fold")
    if mode not in folding_modes:
        raise RuntimeError("interpolate_k2theta: unsupported mode", mode)

    # ... supported tc units and corresponding conversion factor to K
    # (i.e. to the same unit as theta); according to GRIB2
    #     isentropic surfaces are coded in K; fieldextra codes
    #     them in cK for NetCDF (to be checked)
    th_tc_unit_conversions = dict(K=1.0, cK=0.01)
    if th_tc_units not in th_tc_unit_conversions.keys():
        raise RuntimeError(
            "interpolate_k2theta: unsupported value of th_tc_units", th_tc_units
        )
    # ... supported range of tc values (in K)
    th_tc_min = 1.0
    th_tc_max = 1000.0
    # ... tc values outside range of meaningful values of height,
    # used in tc interval search (in m amsl)
    h_min = -1000.0
    h_max = 100000.0
    # ... supported vertical coordinate type for field and p_field
    supported_vc_type = "generalVerticalLayer"

    # Define vertical target coordinates
    tc = dict()
    tc_values = th_tc_values.copy()
    tc_values.sort(reverse=False)
    # Sorting cannot be exploited for optimizations, since theta is
    # not monotonous wrt to height tc values are stored in K
    tc["values"] = np.array(th_tc_values) * th_tc_unit_conversions[th_tc_units]
    if min(tc["values"]) < th_tc_min or max(tc["values"]) > th_tc_max:
        raise RuntimeError(
            "interpolate_k2theta: target coordinate value "
            "out of range (must be in interval [",
            th_tc_min,
            ", ",
            th_tc_max,
            "]K)",
        )
    tc["attrs"] = {
        "units": "K",
        "positive": "up",
        "standard_name": "air_potential_temperature",
        "long_name": "potential temperature",
    }
    tc["typeOfLevel"] = "theta"
    tc["NV"] = 0

    # Check that typeOfLevel is supported and equal for field, th_field, and h_field
    if supported_vc_type not in field.dims:
        raise RuntimeError(
            "interpolate_k2theta: field to interpolate must "
            "be defined for typeOfLevel=",
            supported_vc_type,
        )
    if supported_vc_type not in th_field.dims:
        raise RuntimeError(
            "interpolate_k2theta: theta field must be defined for typeOfLevel=",
            supported_vc_type,
        )
    if supported_vc_type not in h_field.dims:
        raise RuntimeError(
            "interpolate_k2theta: height field must be defined for typeOfLevel=",
            supported_vc_type,
        )

    # Prepare output field field_on_tc on target coordinates
    field_on_tc = init_field_with_vcoord(field, tc, np.nan)

    # Interpolate
    # ... prepare interpolation
    thkm1 = th_field.copy()
    thkm1[{"generalVerticalLayer": slice(1, None)}] = th_field[
        {"generalVerticalLayer": slice(0, -1)}
    ].assign_coords(
        {
            "generalVerticalLayer": th_field[
                {"generalVerticalLayer": slice(1, None)}
            ].generalVerticalLayer
        }
    )
    thkm1[{"generalVerticalLayer": 0}] = np.nan

    fkm1 = field.copy()
    fkm1[{"generalVerticalLayer": slice(1, None)}] = field[
        {"generalVerticalLayer": slice(0, -1)}
    ].assign_coords(
        {
            "generalVerticalLayer": field[
                {"generalVerticalLayer": slice(1, None)}
            ].generalVerticalLayer
        }
    )
    fkm1[{"generalVerticalLayer": 0}] = np.nan

    # ... loop through tc values
    for tc_idx, th0 in enumerate(tc["values"]):
        folding_coord_exception = xr.full_like(
            h_field[{"generalVerticalLayer": 0}], False
        )
        # ... find the height field where theta is >= th0 on level k and was <= th0
        #     on level k-1 or where theta is <= th0 on level k
        #     and was >= th0 on level k-1
        h = h_field.where(
            ((th_field >= th0) & (thkm1 <= th0)) | ((th_field <= th0) & (thkm1 >= th0))
        )
        if mode == "undef_fold":
            # ... find condition where more than one interval is found, which
            # contains the target coordinate value
            folding_coord_exception = xr.where(h.notnull(), 1.0, 0.0).sum(
                dim=["generalVerticalLayer"]
            )
            folding_coord_exception = folding_coord_exception.where(
                folding_coord_exception > 1.0
            ).notnull()
        if mode in ("low_fold", "undef_fold"):
            # ... extract the index k of the smallest height at which
            # the condition is fulfilled
            tcind = h.fillna(h_max).argmin(dim=["generalVerticalLayer"])
        if mode == "high_fold":
            # ... extract the index k of the largest height at which the condition
            # is fulfilled
            tcind = h.fillna(h_min).argmax(dim=["generalVerticalLayer"])

        # ... extract theta and field at level k
        th2 = th_field[{"generalVerticalLayer": tcind["generalVerticalLayer"]}]
        f2 = field[{"generalVerticalLayer": tcind["generalVerticalLayer"]}]
        # ... extract theta and field at level k-1
        f1 = fkm1[{"generalVerticalLayer": tcind["generalVerticalLayer"]}]
        th1 = thkm1[{"generalVerticalLayer": tcind["generalVerticalLayer"]}]

        # ... compute the interpolation weights
        ratio = xr.where(np.abs(th2 - th1) > 0, (th0 - th1) / (th2 - th1), 0.0)

        # ... interpolate and update field_on_tc
        field_on_tc[{tc["typeOfLevel"]: tc_idx}] = xr.where(
            folding_coord_exception, np.nan, (1.0 - ratio) * f1 + ratio * f2
        )

    return field_on_tc
