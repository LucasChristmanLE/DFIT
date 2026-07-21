"""Renderer view contract: render_* never sets Axes limits; it returns a ViewDefaults that the
caller (ui.py) applies. Also covers the pure view-resolution helper ui.py uses in refresh()."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from dfit_tool.model import compute_all
from dfit_tool import plots
from dfit_tool.plots import ViewDefaults
from dfit_tool.ui import ViewState, _resolve_view
from tests.helpers import make_testdata, overview_state


def _render(renderer, td, state, res):
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = renderer(ax, td, state, res)
    return fig, ax, defaults


def test_all_renderers_return_view_defaults():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    for name, renderer in plots.RENDERERS.items():
        _, _, defaults = _render(renderer, td, state, res)
        assert isinstance(defaults, ViewDefaults), f"{name} did not return a ViewDefaults"


def test_gfunction_leaves_twin_axes_unclipped_but_returns_percentile_y2lim():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_gfunction, td, state, res)
    dg = res.diagnostics
    finite = np.isfinite(dg.dPdG)
    hi = np.percentile(dg.dPdG[finite], 95)
    expected_clip = (0, max(hi * 1.5, 1.0))

    assert defaults.y2lim == pytest.approx(expected_clip)

    twin = next(a for a in fig.axes if a is not ax)
    actual = twin.get_ylim()
    # The renderer no longer clips the twin axes itself -- the full (unclipped) data stays
    # visible, including whatever sits above the percentile clip that only the returned
    # ViewDefaults carries.
    assert actual != pytest.approx(expected_clip)
    assert actual[1] >= float(np.nanmax(dg.dPdG[finite]))


def test_porepressure_does_not_force_axes_xlim_to_zero():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_porepressure, td, state, res)
    dg = res.diagnostics
    x = dg.t ** -0.5
    xmax = float(np.nanmax(x))

    assert defaults.xlim == pytest.approx((0.0, xmax))
    # Autoscaled (not forced to start at 0 by the renderer).
    assert ax.get_xlim()[0] != 0.0


def test_overview_returns_injection_window_instead_of_setting_it():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_overview, td, state, res)

    t_h = td.t_s / 3600.0
    span_h = max(t_h[state.shutin_idx] - t_h[state.start_idx], 0.25)
    expected = (t_h[state.start_idx] - 0.5 * span_h, t_h[state.shutin_idx] + 2.0 * span_h)

    assert defaults.xlim == pytest.approx(expected)
    # The Axes itself is left autoscaled to the full file extent (a few hundred seconds here),
    # not the injection-window default -- whose 0.25 h floor dwarfs this short synthetic file.
    actual = ax.get_xlim()
    assert actual != pytest.approx(expected)
    assert actual[1] < float(t_h[-1]) + 0.05


def test_resolve_view_first_visit_seeds_from_defaults_falling_back_to_full_extent():
    defaults = ViewDefaults(xlim=(1.0, 2.0), ylim=None, y2lim=None)
    view = _resolve_view(None, defaults, full_x=(0.0, 10.0), full_y=(0.0, 5.0), full_y2=(0.0, 3.0))
    assert view.xlim == (1.0, 2.0)  # renderer had an opinion
    assert view.ylim == (0.0, 5.0)  # renderer left it None -> full autoscaled extent
    assert view.y2lim == (0.0, 3.0)  # same for the twin axes


def test_resolve_view_revisit_reuses_stored_view_unchanged():
    stored = ViewState(xlim=(3.0, 4.0), ylim=(1.0, 2.0), y2lim=(0.0, 1.0))
    defaults = ViewDefaults(xlim=(100.0, 200.0))
    view = _resolve_view(stored, defaults, full_x=(0.0, 10.0), full_y=(0.0, 5.0), full_y2=None)
    assert view is stored
