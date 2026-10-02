"""Default y-limits round outward to tick-aligned values (plots.nice_limits / nice_log_limits)."""

import math

import numpy as np
import pytest
from matplotlib.figure import Figure

from dfit_tool import plots
from dfit_tool.model import compute_all
from tests.helpers import make_testdata, injection_state


@pytest.mark.parametrize("lo, hi, expected", [
    (0.0, 6430.0, (0.0, 7000.0)),
    (5770.0, 6430.0, (5700.0, 6500.0)),
    (0.0, 5.5, (0.0, 6.0)),
    (0.0, 330.0, (0.0, 350.0)),
    (0.0, 30.0, (0.0, 30.0)),        # already round: unchanged
    (-12.3, 8.7, (-12.5, 10.0)),
    (810.0, 5200.0, (500.0, 5500.0)),  # step stays 500, not 1000 (which would floor at 0)
])
def test_nice_limits_rounds_outward(lo, hi, expected):
    assert plots.nice_limits(lo, hi) == pytest.approx(expected)


@pytest.mark.parametrize("lo, hi", [(float("nan"), 1.0), (1.0, float("inf")), (5.0, 5.0), (5.0, 1.0)])
def test_nice_limits_passes_through_unusable_input(lo, hi):
    out = plots.nice_limits(lo, hi)
    assert (out[0] == lo or (math.isnan(out[0]) and math.isnan(lo))) and out[1] == hi


def test_nice_log_limits_rounds_to_decades():
    assert plots.nice_log_limits(0.037, 4.2) == pytest.approx((0.01, 10.0))
    assert plots.nice_log_limits(1.0, 100.0) == pytest.approx((1.0, 100.0))


def _is_round(v: float) -> bool:
    """True when v is a multiple of {1, 2, 2.5, 5} x 10^k for some k with few significant digits."""
    if v == 0:
        return True
    return abs(v - float(f"{v:.2g}")) < 1e-9 * abs(v)


def test_renderer_default_ylims_are_round():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    for step in ("overview", "injection", "isip", "gfunction", "tangent"):
        ax = Figure().add_subplot(111)
        d = plots.RENDERERS[step](ax, td, state, res)
        for lim in (d.ylim, d.y2lim):
            if lim is not None:
                assert _is_round(lim[0]) and _is_round(lim[1]), (step, lim)


def test_isip_ylim_fits_default_xlim_window_not_whole_plotted_clamp():
    """ISIP plots -5..15 min but opens on -1..3 min. Injection pressure before -1 min (here a
    12000 psi excursion at -3 min) must not set the default pressure ceiling."""
    td = make_testdata()
    state = injection_state(td)
    td.df.loc[state.shutin_idx - 180, td.columns[0]] = 12000.0  # -3 min, outside default view
    res = compute_all(state, td)
    ax = Figure().add_subplot(111)
    d = plots.render_isip(ax, td, state, res)
    t_min = (td.t_s - res.t_shutin_s) / 60.0
    in_view = (t_min >= d.xlim[0]) & (t_min <= d.xlim[1])
    p_view = res.bhp_all[in_view]
    assert d.ylim[1] < 12000.0
    assert d.ylim[0] <= np.nanmin(p_view) and d.ylim[1] >= np.nanmax(p_view)


@pytest.mark.parametrize("step", ["injection", "isip", "tangent", "porepressure"])
def test_pressure_steps_return_round_ylim_from_plotted_data(step):
    """These steps used to return ylim=None, leaving the pressure axis on matplotlib's
    unrounded autoscale (e.g. 2153..11958)."""
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    ax = Figure().add_subplot(111)
    d = plots.RENDERERS[step](ax, td, state, res)
    assert d.ylim is not None
    assert _is_round(d.ylim[0]) and _is_round(d.ylim[1]), d.ylim
    if step == "isip":
        return  # fits the default view window only; see the dedicated ISIP test above
    p = np.asarray(ax.get_lines()[0].get_ydata(), dtype=float)
    p = p[np.isfinite(p)]
    assert d.ylim[0] <= p.min() and d.ylim[1] >= p.max()
