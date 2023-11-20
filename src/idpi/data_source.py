"""Data source helper class."""

# Standard library
import dataclasses as dc
import sys
import typing
from contextlib import contextmanager, nullcontext
from functools import singledispatchmethod
from pathlib import Path

# Third-party
import earthkit.data as ekd  # type: ignore
import eccodes  # type: ignore

# Local
from . import config, mars

GRIB_DEF = {
    mars.Model.COSMO_1E: "cosmo",
    mars.Model.COSMO_2E: "cosmo",
}


@contextmanager
def cosmo_grib_defs():
    """Enable COSMO GRIB definitions."""
    root_dir = Path(sys.prefix) / "share"
    paths = (
        root_dir / "eccodes-cosmo-resources/definitions",
        root_dir / "eccodes/definitions",
    )
    for path in paths:
        if not path.exists():
            raise RuntimeError(f"{path} does not exist")
    defs_path = ":".join(map(str, paths))
    restore = eccodes.codes_definition_path()
    eccodes.codes_set_definitions_path(defs_path)
    try:
        yield
    finally:
        eccodes.codes_set_definitions_path(restore)


def grib_def_ctx(grib_def: str):
    if grib_def == "cosmo":
        return cosmo_grib_defs()
    return nullcontext()


@dc.dataclass
class DataSource:
    datafiles: list[str] | None = None
    request_template: dict[str, typing.Any] = dc.field(default_factory=dict)

    @singledispatchmethod
    def query(self, request):
        raise NotImplementedError(f"request of type {type(request)} not supported.")

    @query.register
    def _(self, request: dict):
        # The presence of the yield keyword makes this def a generator.
        # As a result, the context manager will remain active until the
        # exhaustion of the data source iterator.
        req_kwargs = self.request_template | request
        req = mars.Request(**req_kwargs)

        grib_def = config.get("data_scope", GRIB_DEF[req.model])
        with grib_def_ctx(grib_def):
            if self.datafiles:
                fs = ekd.from_source("file", self.datafiles)
                source = fs.sel(req_kwargs)
                # ideally, the sel would be done with the mars request but
                # fdb and file sources currently disagree on the type of the
                # date and time fields.
                # see: https://github.com/ecmwf/earthkit-data/issues/253
            else:
                source = ekd.from_source("fdb", req.to_fdb())
            yield from source

    @query.register
    def _(self, request: str):
        yield from self.query({"param": request})

    @query.register
    def _(self, request: tuple):
        param, levtype = request
        yield from self.query({"param": param, "levtype": levtype})
