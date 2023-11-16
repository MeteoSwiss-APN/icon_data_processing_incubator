from unittest.mock import patch, call

from idpi import data_source, mars


@patch.object(data_source.ekd, "from_source")
def test_query_files(mock_from_source):
    datafiles = ["foo"]
    param = "bar"

    ds = data_source.DataSource(datafiles)
    for _ in ds.query(param):
        pass

    assert mock_from_source.mock_calls == [
        call("file", datafiles),
        call().sel({"param": param}),
        call().sel().__iter__(),
    ]


@patch.object(data_source.ekd, "from_source")
def test_query_fdb(mock_from_source):
    datafiles = []
    param = "U"
    template = {"date": "20200101", "time": "0000"}

    ds = data_source.DataSource(datafiles, template)
    for _ in ds.query(param):
        pass

    assert mock_from_source.mock_calls == [
        call("fdb", mars.Request(param, **template).to_fdb()),
        call().__iter__(),
    ]
