import numpy as np
from idpi.constants import pc_rdv, pc_o_rdv, pc_b1, pc_b2w, pc_b3, pc_b4w


def pv_sw(t):
    """Pressure of water vapor at equilibrium over liquid water.
    Temperature t must be expressed in Kelvin
    Result is in Pascal"""

    return pc_b1 * np.exp(pc_b2w * (t - pc_b3) / (t - pc_b4w))


def qv_pvp(pv, p):
    """Specific water vapor content (from perfect gas law and approximating q~w).
    Dimensionless"""

    return pc_rdv * pv / np.maximum((p - pc_o_rdv * pv), 1.0)


def pv_qp(qv, p):
    """Partial pressure of water vapor (from perfect gas law and approximating q~w). Same unit as p"""

    return qv * p / (pc_rdv + pc_o_rdv * qv)
