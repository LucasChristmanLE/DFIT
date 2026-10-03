"""render_overview manual mask/keep bands and dropout-marker thinning."""

import numpy as np
from matplotlib.figure import Figure

from dfit_tool import colors as C, picks, plots
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata


def _setup():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    return td, st, res


def _spans(ax, gid):
    return [p for p in ax.patches if p.get_gid() == gid]


def test_overview_draws_manual_spans_at_hour_positions():
    td, st, res = _setup()
    st.mask_intervals = [(3600.0, 7200.0)]
    st.keep_intervals = [(10800.0, 14400.0), (18000.0, 21600.0)]
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=False)
    m = _spans(ax, "manual_mask")
    k = _spans(ax, "manual_keep")
    assert len(m) == 1 and len(k) == 2
    assert m[0].get_x() == 1.0 and m[0].get_x() + m[0].get_width() == 2.0
    assert sorted(p.get_x() for p in k) == [3.0, 5.0]
    assert m[0].get_facecolor()[:3] != k[0].get_facecolor()[:3]


def test_manual_spans_do_not_change_view_defaults():
    td, st, res = _setup()
    ax0 = Figure().add_subplot(111)
    vd0 = plots.render_overview(ax0, td, st, res)
    st.mask_intervals = [(-5000.0, 1e6)]
    st.keep_intervals = []
    ax1 = Figure().add_subplot(111)
    vd1 = plots.render_overview(ax1, td, st, compute_all(st, td))
    assert vd0 == vd1


def test_dropout_markers_thinned_to_cap():
    ax = Figure().add_subplot(111)
    x = np.arange(100000, dtype=float)
    plots._plot_dropout_markers(ax, x, x * 2.0)
    line = next(l for l in ax.get_lines() if l.get_gid() == "dropout_masked")
    assert 10000 <= len(line.get_xdata()) <= 20000


def test_dropout_markers_small_not_thinned():
    ax = Figure().add_subplot(111)
    x = np.arange(500, dtype=float)
    plots._plot_dropout_markers(ax, x, x)
    line = next(l for l in ax.get_lines() if l.get_gid() == "dropout_masked")
    assert len(line.get_xdata()) == 500


def test_color_roles_exist():
    assert C.MANUAL_MASK == C.DROPOUT
    assert C.MANUAL_KEEP not in (C.WINDOW, C.EXCLUDED)
    assert 0 < C.MANUAL_SPAN_ALPHA < 0.5
