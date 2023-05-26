from idpi.iarray import Iarray, Off
from idpi.operators.gradient import gradient
import numpy as np
import unittest


class IarrayCtr(unittest.TestCase):
    def test_iarray(self):
        iarr = Iarray(dims=["x"], coords={"x": 0, "y": 0}, data=np.random.randn(2))
        self.assertTrue(iarr.dims == ["x"])

    def test_access(self):
        # TODO this should fail
        iarr = Iarray(dims=["x"], coords={"x": 0, "y": 0}, data=np.array([2, 3, 4, 5]))
        self.assertTrue(np.array_equal(iarr.isel({"x": slice(1, None)}), [3, 4, 5]))

        # TODO test do not fail if array shape is != that dims

    def test_access_2d(self):
        iarr = Iarray(
            dims=["x", "y"],
            coords={"x": 0, "y": 0},
            data=np.array([[2, 3, 4, 5], [5, 8, 5, 1]]),
        )
        iarr[{"y": slice(1, None)}] = [[30, 40, 50], [50, 60, 70]]
        self.assertTrue(
            np.array_equal(iarr, np.array([[2, 30, 40, 50], [5, 50, 60, 70]]))
        )

    def test_set(self):
        iarr = Iarray(dims=["x"], coords={"x": 0, "y": 0}, data=np.array([2, 3, 4, 5]))
        iarr[dict(x=slice(1, -1))] = [8, 9]
        self.assertTrue(np.array_equal(iarr, [2, 8, 9, 5]))

    def test_gradient(self):
        iarr = Iarray(dims=["x"], coords={"x": 0, "y": 0}, data=np.array([2, 3, 4, 5]))
        self.assertTrue(
            np.array_equal(gradient(iarr, dim="x"), np.gradient(iarr, axis=0))
        )

    def test_gradient2d(self):
        iarr = Iarray(
            dims=["y", "x"],
            coords={"x": 0, "y": 0},
            data=np.array([[2, 3, 4, 5], [5, 8, 5, 1]]),
        )
        self.assertTrue(
            np.array_equal(gradient(iarr, dim="x"), np.gradient(iarr, axis=1))
        )

        self.assertTrue(
            np.array_equal(gradient(iarr, dim="y"), np.gradient(iarr, axis=0))
        )

    def test_invalid_coords(self):
        with self.assertRaises(ValueError):
            Iarray(
                dims=["x"],
                data=np.random.randn(2),
            )

        with self.assertRaises(ValueError):
            Iarray(
                dims=["x"],
                coords={"x": 1},
                data=np.random.randn(2),
            )
        with self.assertRaises(ValueError):
            Iarray(
                dims=["y", "x"],
                coords={"x": 0},
                data=np.array([[2, 3, 4, 5], [5, 8, 5, 1]]),
            )

        with self.assertRaises(ValueError):
            Iarray(
                dims=["z", "x"],
                coords={"x": 0},
                data=np.array([[2, 3, 4, 5], [5, 8, 5, 1]]),
            )

    def test_coords(self):
        iarr = Iarray(
            dims=["z", "x"],
            coords={"x": 0, "z": -0.5},
            data=np.array([[2, 3, 4, 5], [5, 8, 5, 1]]),
        )
        self.assertTrue(iarr.coords["z"] == -0.5)
