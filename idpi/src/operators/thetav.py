#!/usr/bin/python

import constants as const


def fthetav(p, t, qv):
    return (const.p0 / p) ** const.pc_rdocp * t * (1.0 + (const.pc_rvd_o * qv / (1.0 - qv)))
