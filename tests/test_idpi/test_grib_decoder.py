import pytest
from unittest.mock import Mock, patch
from idpi.grib_decoder import GribReader

earthkit = Mock()


@patch("idpi.grib_decoder.earthkit.data.from_source")
def test_ref_param_not_found(mock_from_source):
    with pytest.raises(ValueError):
        reader = GribReader([])


@patch("idpi.grib_decoder.GribReader.load_grid_reference")
@patch("idpi.grib_decoder.earthkit.data.from_source")
def test_ref_param_not_found(mock_load_grid_reference, mock_from_source):
    mock_from_source.sel.return_value = []

    with pytest.raises(ValueError):
        reader = GribReader([])
        reader.load_dataset(["U", "V"])
