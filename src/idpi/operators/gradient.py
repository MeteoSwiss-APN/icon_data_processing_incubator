"""Algorithm for computation of height of zero degree isotherm.

This is done without extrapolation below model orography.
"""
# Third-party
import numpy as np

# First-party
from idpi.iarray import Iarray


def gradient(iarray: Iarray, dim: str):
    axis = iarray.dims.index(dim)
    return np.gradient(iarray, axis=axis)
