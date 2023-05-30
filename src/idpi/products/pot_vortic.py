# Standard library
import sys
import logging

# Third-party
import numpy as np

# First-party
from idpi.operators.destagger import destagger
from idpi.operators.pot_vortic import fpotvortic
from idpi.operators.rho import f_rho_tot
from idpi.operators.theta import ftheta
from idpi.operators.total_diff import TotalDiff
from idpi.operators.vertical_interpolation import interpolate_k2p
from idpi.operators.vertical_interpolation import interpolate_k2theta
from idpi.operators.vertical_reduction import integrate_k

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def _compute_pot_vortic(U, V, W, T, P, QV, QC, QI, HHL, theta):
    logger.info("Computing total density")
    rho_tot = f_rho_tot(T, P, QV, QC, QI)

    logger.info("Computing terrain following grid deformation factors")
    dlon = HHL.attrs["GRIB_iDirectionIncrementInDegrees"]
    dlat = HHL.attrs["GRIB_jDirectionIncrementInDegrees"]
    deg2rad = np.pi / 180
    total_diff = TotalDiff(dlon * deg2rad, dlat * deg2rad, HHL)

    logger.info("Computing potential vorticity")
    return fpotvortic(U, V, W, theta, rho_tot, total_diff)


def _compute_mean(pot_vortic, hhl, hfl, pressure):
    logger.info("Computing mean potential vorticity between 700 and 900 hPa")
    h700, h900 = interpolate_k2p(hfl, "linear_in_lnp", pressure, [700, 900], "hPa")
    return integrate_k(pot_vortic, "normed_integral", "z2z", hhl, (h900, h700))


def _compute_at_theta(theta, hfl, **fields):
    logger.info(f"Interpolating {tuple(fields.keys())} at isotherms")
    theta_values = [310, 315, 320, 325, 330, 335]
    return {
        key: interpolate_k2theta(field, "low_fold", theta, theta_values, "K", hfl)
        for key, field in fields.items()
    }


def product2(U, V, W, T, P, QV, QC, QI, HHL):
    logger.info("Computing potential temperature")
    theta = ftheta(P, T)

    pot_vortic = _compute_pot_vortic(U, V, W, T, P, QV, QC, QI, HHL, theta)

    hfl = destagger(HHL, "generalVertical")
    output_mean = _compute_mean(pot_vortic, HHL, hfl, P)
    output_at_theta = _compute_at_theta(
        theta,
        hfl,
        p=P,
        u=destagger(U, "x"),
        v=destagger(V, "y"),
        pot_vortic=pot_vortic,
    )

    return output_mean, output_at_theta
