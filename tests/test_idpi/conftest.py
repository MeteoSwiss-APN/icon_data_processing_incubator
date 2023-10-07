"""Test configuration."""
# Standard library
import subprocess
from pathlib import Path

# Third-party
import pytest
import xarray as xr
from jinja2 import Environment, FileSystemLoader


@pytest.fixture
def data_dir():
    """Base data dir."""
    return Path(
        "/project/s83c/rz+/icon_data_processing_incubator/datasets/32_39x45_51/"
    )


@pytest.fixture
def fieldextra_executable():
    """Fieldextra executable."""
    return "/project/s83c/fieldextra/tsa/bin/fieldextra_gnu_opt_omp"


@pytest.fixture
def template_env():
    """Jinja input namelist template environment."""
    test_dir = Path(__file__).parent
    loader = FileSystemLoader(test_dir / "fieldextra_templates")
    return Environment(loader=loader, keep_trailing_newline=True)


@pytest.fixture
def fieldextra(tmp_path, data_dir, template_env, fieldextra_executable):
    """Run fieldextra on a given field."""

    def f(
        product: str,
        conf_files: dict[str, str] | None = None,
        load_output: str | list[str] = "00_outfile.nc",
        **ctx,
    ):
        if not conf_files:
            conf_files = {
                "inputi": data_dir / "COSMO-1E/1h/ml_sl/000/lfff00000000",
                "inputc": data_dir / "COSMO-1E/1h/const/000/lfff00000000c",
                "output": "<HH>_outfile.nc",
            }

        template = template_env.get_template(f"test_{product}.nl")
        nl_path = tmp_path / f"test_{product}.nl"
        nl_path.write_text(template.render(file=conf_files, **ctx))

        subprocess.run([fieldextra_executable, str(nl_path)], check=True, cwd=tmp_path)

        if isinstance(load_output, str):
            return xr.open_dataset(tmp_path / load_output)
        return [xr.open_dataset(tmp_path / foutput) for foutput in load_output]

    return f
