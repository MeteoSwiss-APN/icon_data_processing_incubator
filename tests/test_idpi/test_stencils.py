# Third-party
import numpy as np
from numpy.testing import assert_allclose

# First-party
from idpi.operators import stencils
from idpi.operators.theta import ftheta
from idpi.operators.destagger import destagger
from idpi import grib_decoder


def test_padded_field(data_dir, grib_defs):
    datafile = data_dir / "lfff00000000.ch"

    ds = {}
    grib_decoder.load_data(ds, ["P", "T"], datafile, chunk_size=None)

    theta = ftheta(ds["P"], ds["T"])

    t = stencils.PaddedField(theta.rename(generalVerticalLayer="z"))

    tp = np.pad(theta, 1, mode="edge")
    tp[:, :, 0] = tp[:, :, -1] = np.nan
    tp[:, 0, :] = tp[:, -1, :] = np.nan
    dt_dx = 0.5 * (tp[1:-1, 1:-1, 2:] - tp[1:-1, 1:-1, :-2])
    dt_dy = 0.5 * (tp[1:-1, 2:, 1:-1] - tp[1:-1, :-2, 1:-1])
    dt_dz = 0.5 * (tp[2:, 1:-1, 1:-1] - tp[:-2, 1:-1, 1:-1])
    dt_dz[0, :, :] *= 2
    dt_dz[-1, :, :] *= 2

    assert_allclose(t.dx(), dt_dx)
    assert_allclose(t.dy(), dt_dy)
    assert_allclose(t.dz(), dt_dz)


def test_staggered_field(data_dir, grib_defs):
    datafile = data_dir / "lfff00000000.ch"

    ds = {}
    grib_decoder.load_data(ds, ["W"], datafile, chunk_size=None)

    wf = destagger(ds["W"], "generalVertical").rename(generalVerticalLayer="z")
    wp = stencils.PaddedField(wf)
    ws = stencils.StaggeredField(ds["W"].rename(generalVertical="z"), wp, "z")

    wn = ds["W"].to_numpy()
    dw_dx = wp.dx()
    dw_dy = wp.dy()
    dw_dz = wn[1:] - wn[:-1]

    assert_allclose(ws.dx(), dw_dx)
    assert_allclose(ws.dy(), dw_dy)
    assert_allclose(ws.dz(), dw_dz)


def test_total_diff(data_dir, grib_defs):
    cdatafile = data_dir / "lfff00000000c.ch"

    ds = {}
    grib_decoder.load_data(ds, ["HHL"], cdatafile, chunk_size=None)

    deg2rad = np.pi / 180

    hhl = ds["HHL"].values
    dlon = ds["HHL"].attrs["GRIB_iDirectionIncrementInDegrees"] * deg2rad
    dlat = ds["HHL"].attrs["GRIB_jDirectionIncrementInDegrees"] * deg2rad

    inv_dlon = 1 / dlon
    inv_dlat = 1 / dlat
    hhlp = np.pad(hhl, ((0, 0), (1, 1), (1, 1)), constant_values=np.nan)

    sqrtg_r_s = 1 / (hhl[:-1] - hhl[1:])
    dzeta_dlam = (
        0.25
        * inv_dlon
        * sqrtg_r_s
        * (
            (hhlp[:-1, 1:-1, 2:] - hhlp[:-1, 1:-1, :-2])
            + (hhlp[1:, 1:-1, 2:] - hhlp[1:, 1:-1, :-2])
        )
    )
    dzeta_dphi = (
        0.25
        * inv_dlat
        * sqrtg_r_s
        * (
            (hhlp[:-1, 2:, 1:-1] - hhlp[:-1, :-2, 1:-1])
            + (hhlp[1:, 2:, 1:-1] - hhlp[1:, :-2, 1:-1])
        )
    )

    total_diff = stencils.TotalDiff(dlon, dlat, ds["HHL"])

    assert_allclose(total_diff.sqrtg_r_s.values, sqrtg_r_s)
    assert_allclose(total_diff.dzeta_dlam.values, dzeta_dlam, rtol=1e-6)
    assert_allclose(total_diff.dzeta_dphi.values, dzeta_dphi, rtol=1e-6)

    datafile = data_dir / "lfff00000000.ch"

    grib_decoder.load_data(ds, ["P", "T"], datafile, chunk_size=None)
    theta = ftheta(ds["P"], ds["T"])

    tp = np.pad(theta, 1, mode="edge")
    dt_dx = 0.5 * (tp[1:-1, 1:-1, 2:] - tp[1:-1, 1:-1, :-2])
    dt_dy = 0.5 * (tp[1:-1, 2:, 1:-1] - tp[1:-1, :-2, 1:-1])
    dt_dz = 0.5 * (tp[2:, 1:-1, 1:-1] - tp[:-2, 1:-1, 1:-1])
    dt_dz[0] *= 2
    dt_dz[-1] *= 2

    dt_dlam = dt_dx / dlon + dzeta_dlam * dt_dz
    dt_dphi = dt_dy / dlat + dzeta_dphi * dt_dz
    dt_dzeta = dt_dz * sqrtg_r_s

    t = stencils.PaddedField(theta.rename(generalVerticalLayer="z"))
    assert_allclose(dt_dlam, total_diff.d_dlam(t), rtol=1e-5, atol=1e-2)
    assert_allclose(dt_dphi, total_diff.d_dphi(t), rtol=1e-5, atol=1e-2)
    assert_allclose(dt_dzeta, total_diff.d_dzeta(t), rtol=1e-5, atol=1e-3)
