"""Command line interface of idpi."""
# Standard library
from pathlib import Path

# Third-party
import click

# Local
from . import __version__, grib_decoder
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
@click.option(
    "--crs",
    type=click.Choice(["geolatlon"]),
    default="geolatlon",
    help="Coordinate reference system",
)
@click.option(
    "--resampling",
    type=click.Choice(list(RESAMPLING.keys())),
    default="nearest",
    help="Resampling method",
)
@click.argument(
    "infile",
    type=click.Path(exists=True, path_type=Path),
)
@click.argument(
    "outfile",
    type=click.Path(writable=True, path_type=Path),
)
@click.argument("params")
def regrid_cmd(crs: str, resampling: str, infile: Path, outfile: Path, params: str):
    """Regrid the given PARAMS found in INFILE and write to OUTFILE."""
    resampling_arg = RESAMPLING[resampling]
    crs_str = regrid.CRS_ALIASES.get(crs, crs)

    if outfile.exists():
        click.confirm(f"OUTFILE {outfile} exists. Overwrite?")

    reader = grib_decoder.GribReader.from_files([infile], ref_param="HHL")
    ds = reader.load_fieldnames(params.split(","))

    with outfile.open("wb") as fout:
        for name, field in ds.items():
            src = regrid.RegularGrid.from_field(field)
            dst = src.to_crs(crs_str)

            click.echo(f"Regriding field {name} to {dst}")
            field_out = regrid.regrid(field, dst, resampling_arg)

            click.echo(f"Writing grib fields to {outfile}")
            grib_decoder.save(field_out, fout)
