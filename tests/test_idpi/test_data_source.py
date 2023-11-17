from contextlib import nullcontext
from unittest.mock import patch, call

import pytest

from idpi import config, data_source, mars


@pytest.fixture
def mock_from_source():
    with patch.object(data_source.ekd, "from_source") as mock:
        yield mock


@pytest.fixture
def mock_grib_def_ctx():
    with patch.object(data_source, "grib_def_ctx") as mock:
        mock.return_value = nullcontext()
        yield mock


def test_query_files(mock_from_source, mock_grib_def_ctx):
    datafiles = ["foo"]
    param = "bar"

    ds = data_source.DataSource(datafiles)
    for _ in ds.query(param):
        pass

    assert mock_grib_def_ctx.mock_calls == [call("cosmo")]
    assert mock_from_source.mock_calls == [
        call("file", datafiles),
        call().sel({"param": param}),
        call().sel().__iter__(),
    ]


def test_query_files_tuple(mock_from_source, mock_grib_def_ctx):
    datafiles = ["foo"]
    request = param, levtype = ("bar", "ml")

    ds = data_source.DataSource(datafiles)
    for _ in ds.query(request):
        pass

    assert mock_grib_def_ctx.mock_calls == [call("cosmo")]
    assert mock_from_source.mock_calls == [
        call("file", datafiles),
        call().sel({"param": param, "levtype": levtype}),
        call().sel().__iter__(),
    ]


def test_query_files_ifs(mock_from_source, mock_grib_def_ctx):
    datafiles = ["foo"]
    param = "bar"

    with config.set_values(data_scope="ifs"):
        ds = data_source.DataSource(datafiles)
        for _ in ds.query(param):
            pass

    assert mock_grib_def_ctx.mock_calls == [call("ifs")]
    assert mock_from_source.mock_calls == [
        call("file", datafiles),
        call().sel({"param": param}),
        call().sel().__iter__(),
    ]


def test_query_fdb(mock_from_source, mock_grib_def_ctx):
    datafiles = []
    param = "U"
    template = {"date": "20200101", "time": "0000"}

    ds = data_source.DataSource(datafiles, template)
    for _ in ds.query(param):
        pass

    assert mock_grib_def_ctx.mock_calls == [call("cosmo")]
    assert mock_from_source.mock_calls == [
        call("fdb", mars.Request(param, **template).to_fdb()),
        call().__iter__(),
    ]
