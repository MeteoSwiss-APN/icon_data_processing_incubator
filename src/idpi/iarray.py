from collections.abc import Hashable, Sequence
import numpy as np
from dataclasses import dataclass
from functools import partial
from typing import Literal


@dataclass
class Off:
    value: int = 0


class CoordValues:
    def __init__(self, base: Literal["pressure", "theta", "z"], values: list):
        self.base = base
        self.values = values


class Iarray(np.ndarray):
    def __new__(cls, *, dims: Sequence[Hashable], data, coords={}):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        # add the new attribute to the created instance
        if dims and len(dims) != len(data.shape):
            raise ValueError("size of dims and data do not match")

        obj.dims = dims

        for dim in ("x", "y", "z"):
            if dim in dims and dim not in coords:
                raise ValueError(
                    "coordinates should provide info for all present dims: ", dim
                )

        valid_coords = {"x": [0, 0.5], "y": [0, 0.5], "z": [-0.5, 0]}
        for dim in ("x", "y"):
            if dim in dims:
                if coords["x"] not in valid_coords[dim]:
                    raise ValueError("wrong coordinate for ", dim)

        if "z" in dims:
            if (
                not isinstance(coords["z"], CoordValues)
                and coords["z"] not in valid_coords["z"]
            ):
                raise ValueError("wrong coordinate for z")

        obj.coords = coords
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.dims = getattr(obj, "dims", None)
        self.coords = getattr(obj, "coords", None)

    # def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
    #     for i in inputs:
    #         print("KK", type(i))
    #     print("JJJ", ufunc, method)
    #     print("JJJ", *inputs)

    def isel(self, keys):
        slices = tuple(keys[dim] if dim in keys else slice(None) for dim in self.dims)
        return super().__getitem__(slices)

    def __setitem__(self, keys, value):
        if isinstance(keys, dict):
            slices = tuple(
                keys[dim] if dim in keys else slice(None) for dim in self.dims
            )
            return super().__setitem__(slices, value)

        return super().__setitem__(keys, value)

    def __getitem__(self, keys):
        if isinstance(keys, dict):
            slices = tuple(
                keys[dim] if dim in keys else slice(None) for dim in self.dims
            )
            return super().__getitem__(slices)

        return super().__getitem__(keys)
