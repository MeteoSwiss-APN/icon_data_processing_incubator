"""Command line interface of idpi."""
# Standard library
from importlib.resources import files
from pathlib import Path

# Third-party
import click
import yaml

# Local
from . import __version__, grib_decoder
from .operators import destagger, gis, regrid


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


def _load_mapping():
    mapping_path = files("idpi.data").joinpath("field_mappings.yml")
    return yaml.safe_load(mapping_path.open())


def handle_vector_fields(ds):
    mapping = _load_mapping()
    names = set(ds)
    pairs = []
    while names:
        name = names.pop()
        item = mapping[name]["cosmo"]
        if u := item.get("uComponent"):
            if u not in names:
                raise click.Abort(f"The u-component {u} must be part of PARAMS")
            names.remove(u)
            pairs.append((u, name))
        elif v := item.get("vComponent"):
            if v not in names:
                raise click.Abort(f"The v-component {v} must be part of PARAMS")
            names.remove(v)
            pairs.append((name, v))

    for u_name, v_name in pairs:
        click.echo(f"Rotating vector field components {u_name}, {v_name} to geolatlon")
        u, v = ds[u_name], ds[v_name]
        if u.origin["x"] != 0.0:
            u = destagger.destagger(u, "x")
        if v.origin["y"] != 0.0:
            v = destagger.destagger(v, "y")
        u_g, v_g = gis.vref_rot2geolatlon(u, v)
        ds[u_name] = u_g
        ds[v_name] = v_g


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

    handle_vector_fields(ds)

    with outfile.open("wb") as fout:
        for name, field in ds.items():
            src = regrid.RegularGrid.from_field(field)
            dst = src.to_crs(crs_str)

            click.echo(f"Regriding field {name} to {dst}")
            field_out = regrid.regrid(field, dst, resampling_arg)

            click.echo(f"Writing grib fields to {outfile}")
            grib_decoder.save(field_out, fout)
