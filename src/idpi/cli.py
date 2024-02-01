"""Command line interface of idpi."""
from pathlib import Path

# Third-party
import click

# Local
from . import __version__
from . import grib_decoder
from .operators import regrid


def print_version(ctx, _, value: bool) -> None:
    """Print the version number and exit."""
    if value:
        click.echo(__version__)
        ctx.exit(0)


@click.group()
@click.option(
    "--version",
    "-V",
    help="Print version and exit.",
    is_flag=True,
    expose_value=False,
    callback=print_version,
)
def main() -> None:
    """Console script for test_cli_project."""
    print("CLI for IDPI")


RESAMPLING = {
    "nearest": regrid.Resampling.nearest,
    "bilinear": regrid.Resampling.bilinear,
    "cubic": regrid.Resampling.cubic,
}


@main.command("regrid")
@click.option("--crs", type=click.Choice(["geolatlon"]), default="geolatlon")
@click.option("--resampling", type=click.Choice(list(RESAMPLING.keys())), default="nearest")
@click.argument("infile")
@click.argument("outfile")
@click.argument("params")
def regrid_cmd(crs: str, resampling: str, infile: Path, outfile: Path, params: str):
    resampling_arg = RESAMPLING[resampling]
    crs_str = regrid.CRS_ALIASES.get(crs, crs)
    
    reader = grib_decoder.GribReader.from_files([infile])
    ds = reader.load_fieldnames(params.split(","))

    with outfile.open("wb") as fout:
        for field in ds.values():
            src = regrid.RegularGrid.from_field(field)
            dst = src.to_crs(crs_str)

            field_out = regrid.regrid(field, dst, resampling_arg)
            grib_decoder.save(field_out, fout)
