import flexpart_cli
from click.testing import CliRunner
import pytest

class _TestCLI:
    """Base class to test the command line interface."""

    def call(self, args=None, *, exit_=0):
        runner = CliRunner()
        result = runner.invoke(flexpart_cli.run_flexpart, args)
        assert result.exit_code == exit_
        return result

@pytest.mark.ifs
class TestCmd(_TestCLI):
    """Test CLI with some commands."""

    def test_help(self):
        result = self.call(["--help"])
        assert result.output.startswith("Usage: ")
        assert "Show this message and exit." in result.output

    def test_run(self):
        self.call(
            [
                "--data_dir",
                "/project/s83c/rz+/icon_data_processing_incubator/data/ifs-flexpart-europe/23030600",
                "--rhour",
                "0",
                "--rdate",
                "20230306",
                "--nsteps",
                "3",
            ]
        )
