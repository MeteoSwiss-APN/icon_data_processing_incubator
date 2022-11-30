"""algorithm for computing omega_slope."""
import numpy as np
import operators.constants as cnt

# similar to the subtract.accumulate but permute the order of the operans of the diff
# TODO implement as a ufunc
def cumdiff(A, axis):
    r = np.empty(np.shape(A))
    t = 0  # op = the ufunc being applied to A's  elements
    for i in range(np.shape(A)[axis]):
        t = np.take(A, i, axis) - t

        slices = []
        for dim in range(A.ndim):
            if dim == axis:
                slices.append(slice(i, i + 1))
            else:
                slices.append(slice(None))

        r[tuple(slices)] = np.expand_dims(t, axis=t.ndim)
    return r


def omega_slope(ps, etadot, ak, bk):
    """Compute omega slope."""
 
    ak1 = ak[1:].assign_coords(
        {"hybrid": ak[{"hybrid": slice(0, -1)}].hybrid}
    )
    bk1 = bk[1:].assign_coords(
        {"hybrid": bk[{"hybrid": slice(0, -1)}].hybrid}
    )

    return (
        2.0
        * ps
        * etadot
        * ((ak1 - ak[0:-1]) / ps + bk1 - bk[0:-1])
        / ((ak1 - ak[0:-1]) / cnt.surface_pressure_ref() + bk1 - bk[0:-1])
    ).reduce(cumdiff, dim="hybrid")