"""Hover readout: ``plots.hover_series``/``hover_text`` (what each step exposes and the text shown)
and ``picks.HoverReadoutController`` (nearest-sample snapping, hide rules, cleanup). Controllers
are driven with real synthesized ``MouseEvent``s on an Agg canvas, like the other controller tests.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import picks, plots
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata


# --------------------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------------------
def _full_case():
    """A state with window, ISIP, G-function, and pore-pressure picks in place so every step has
    something to hover (mirrors tests/test_stiffness.py)."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    dg = res.diagnostics
    picks.commit_min_dpdg_point(st, float(dg.G[dg.G.size // 4]))
    picks.commit_contact_point(st, float(dg.G[3 * dg.G.size // 4]))
    picks.commit_closure_point(st, float(dg.G[dg.G.size // 2]))
    res = compute_all(st, td)
    picks.seed_pp(st, res)
    st.postclosure_scenario = "PC-A linear"
    st.pp_axis = picks.suggest_pp_axis(st.postclosure_scenario) or st.pp_axis
    res = compute_all(st, td)
    assert res.stiffness_S is not None
    return td, st, res


def _axes(with_twin=False, log=False):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    if log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if with_twin:
        ax.twinx()
    return fig, ax, canvas


def _motion(canvas, ax, x, y, button=None, buttons=None):
    px, py = ax.transData.transform((x, y))
    return MouseEvent("motion_notify_event", canvas, px, py, button=button, buttons=buttons)


def _make(ax, canvas, x, y, text_fn=None):
    ax.plot(x, y)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(float(np.min(y)) - 1.0, float(np.max(y)) + 1.0)
    canvas.draw()
    return picks.HoverReadoutController(canvas, ax, x, y, text_fn or (lambda i: f"i={i}"))


# --------------------------------------------------------------------------------------------------
# hover_series
# --------------------------------------------------------------------------------------------------
def test_overview_and_injection_series_use_hours_and_bhp():
    td, st, res = _full_case()
    for step in ("overview", "injection"):
        s = plots.hover_series(step, td, st, res)
        assert s is not None
        assert s.value_label in ("BHP", "Surface")
        k = len(s.x) // 2
        assert np.isclose(s.t_file_s[k], s.x[k] * 3600.0)
        np.testing.assert_allclose(s.y[k], res.bhp_all[np.argmin(np.abs(td.t_s - s.t_file_s[k]))])
        assert s.t_shutin_s == res.t_shutin_s


def test_overview_series_includes_surface_overlay():
    td, st, res = _full_case()
    res.pressure_is_bhp = True
    res.p_surface_all = np.asarray(res.bhp_all, float) - 3000.0
    res.surface_dropout_mask = np.zeros(td.n, dtype=bool)
    res.surface_dropout_mask[5] = True  # masked surface samples are not drawn, so not hoverable
    s = plots.hover_series("overview", td, st, res)
    labels = np.asarray(s.value_label)
    n_bhp = int(np.isfinite(res.bhp_all).sum())
    assert (labels == "BHP").sum() == n_bhp
    surf = labels == "Surface"
    assert surf.sum() == int(np.isfinite(res.p_surface_all).sum()) - 1
    k = np.flatnonzero(surf)[10]
    i_raw = int(np.argmin(np.abs(td.t_s - s.t_file_s[k])))
    assert s.y[k] == res.p_surface_all[i_raw]
    assert plots.hover_text(s, int(k)).startswith(f"Surface {s.y[k]:.0f} psi\n")
    assert plots.hover_text(s, 0).startswith("BHP ")
    # The overlay is drawn on Overview only.
    assert plots.hover_series("injection", td, st, res).value_label == "BHP"


def test_isip_series_x_is_minutes_from_shutin():
    td, st, res = _full_case()
    s = plots.hover_series("isip", td, st, res)
    k = len(s.x) // 2
    assert np.isclose(s.x[k], (s.t_file_s[k] - res.t_shutin_s) / 60.0)


def test_gfunction_and_tangent_series_align_with_diagnostics():
    td, st, res = _full_case()
    dg, rs = res.diagnostics, res.resampled
    for step in ("gfunction", "tangent"):
        s = plots.hover_series(step, td, st, res)
        ok = np.isfinite(dg.G) & np.isfinite(rs.p)
        k = 3
        kk = np.flatnonzero(ok)[k]
        assert s.x[k] == dg.G[kk]
        assert s.y[k] == rs.p[kk]
        assert s.t_file_s[k] == res.t_shutin_s + rs.dt[kk]
        assert s.value_label is not None


def test_loglog_series_is_time_only():
    td, st, res = _full_case()
    dg = res.diagnostics
    s = plots.hover_series("loglog", td, st, res)
    assert s.value_label is None
    good = (dg.t > 0) & (dg.dp > 0)
    np.testing.assert_array_equal(s.x, dg.t[good])
    np.testing.assert_array_equal(s.y, dg.dp[good])
    np.testing.assert_allclose(s.t_file_s, res.t_shutin_s + dg.t[good])


def test_porepressure_series_uses_plotted_exponent():
    td, st, res = _full_case()
    dg = res.diagnostics
    expo = -0.5 if st.pp_axis == "tm12" else -1.0
    s = plots.hover_series("porepressure", td, st, res)
    assert s.value_label is not None
    k = 2
    assert s.x[k] == dg.t[np.flatnonzero(np.isfinite(dg.t ** expo) & np.isfinite(dg.p))[k]] ** expo


def test_stiffness_series_is_time_only_and_aligned():
    td, st, res = _full_case()
    s = plots.hover_series("stiffness", td, st, res)
    assert s.value_label is None
    S = res.stiffness_S
    ok = np.isfinite(S) & (S > 0) & np.isfinite(res.stiffness_p_eff[1:])
    np.testing.assert_array_equal(s.y, S[ok])
    np.testing.assert_array_equal(s.x, res.stiffness_p_eff[1:][ok])
    np.testing.assert_allclose(s.t_file_s, res.t_shutin_s + res.stiffness_dt[1:][ok])


def test_stiffness_dt_aligned_with_p_eff():
    td, st, res = _full_case()
    assert res.stiffness_dt is not None
    assert len(res.stiffness_dt) == len(res.stiffness_p_eff)


def test_series_none_without_diagnostics_or_data():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    res.diagnostics = None
    res.stiffness_S = None
    for step in ("gfunction", "tangent", "loglog", "porepressure", "stiffness"):
        assert plots.hover_series(step, td, st, res) is None
    res.bhp_all = None
    assert plots.hover_series("overview", td, st, res) is None


# --------------------------------------------------------------------------------------------------
# hover_text
# --------------------------------------------------------------------------------------------------
def _series(label="BHP", shutin=3600.0):
    return plots.HoverSeries(x=np.array([0.0, 1.0]), y=np.array([5234.4, 100.0]),
                             t_file_s=np.array([44424.0, 7200.0]), t_shutin_s=shutin,
                             value_label=label)


def test_hover_text_pressure_step():
    assert plots.hover_text(_series(), 0) == (
        "BHP 5234 psi\n12.34 h from file start\n11.34 h from shut-in")


def test_hover_text_negative_before_shutin():
    assert plots.hover_text(_series(), 1).splitlines()[2] == "1.00 h from shut-in"
    s = _series(shutin=14400.0)
    assert plots.hover_text(s, 1).splitlines()[2] == "-2.00 h from shut-in"


def test_hover_text_time_only_step():
    assert plots.hover_text(_series(label=None), 0) == (
        "12.34 h from file start\n11.34 h from shut-in")


def test_hover_text_no_shutin():
    assert plots.hover_text(_series(shutin=None), 0) == "BHP 5234 psi\n12.34 h from file start"


# --------------------------------------------------------------------------------------------------
# HoverReadoutController
# --------------------------------------------------------------------------------------------------
def _line():
    x = np.linspace(0.0, 20.0, 41)
    return x, np.sin(x) * 5.0


def test_motion_over_sample_shows_text_for_that_index():
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    assert ctrl._annot.get_visible()
    assert ctrl._annot.get_text() == "i=17"
    assert ctrl._marker.get_visible()


def test_motion_beyond_tolerance_hides():
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    px, py = ax.transData.transform((x[17], y[17]))
    ctrl._on_motion(MouseEvent("motion_notify_event", canvas, px, py + 60))
    assert not ctrl._annot.get_visible()


def test_button_held_hides():
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17], button=1))
    assert not ctrl._annot.get_visible()
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    assert ctrl._annot.get_visible()


def test_works_with_twin_axes_present():
    x, y = _line()
    fig, ax, canvas = _axes(with_twin=True)
    ctrl = _make(ax, canvas, x, y)
    ev = _motion(canvas, ax, x[9], y[9])
    assert ev.inaxes is not ax  # the twin owns inaxes over the shared region
    ctrl._on_motion(ev)
    assert ctrl._annot.get_text() == "i=9"


def test_works_on_log_log_axes():
    x = np.geomspace(1.0, 1000.0, 30)
    y = x ** -0.5 * 100.0
    fig, ax, canvas = _axes(log=True)
    ctrl = _make(ax, canvas, x, y)
    ax.set_xlim(0.8, 1200.0)
    ax.set_ylim(1.0, 200.0)
    canvas.draw()
    ctrl._on_motion(_motion(canvas, ax, x[12], y[12]))
    assert ctrl._annot.get_text() == "i=12"


def test_unsorted_x_snaps_to_original_index():
    x = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, 3.0, 2.0))
    assert ctrl._annot.get_text() == "i=2"


def test_full_redraw_repaints_readout_on_same_sample():
    # A full draw skips the animated artists, so the next motion over the same sample must
    # blit them again rather than treat the readout as already shown.
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    blits = []
    canvas.blit = lambda bbox=None: blits.append(bbox)
    canvas.draw()
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    assert blits
    assert ctrl._annot.get_visible()


def test_figure_leave_hides():
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    ctrl._on_motion(_motion(canvas, ax, x[17], y[17]))
    ctrl._on_leave(None)
    assert not ctrl._annot.get_visible()


def test_disconnect_removes_artists_and_callbacks():
    x, y = _line()
    fig, ax, canvas = _axes()
    ctrl = _make(ax, canvas, x, y)
    annot, marker = ctrl._annot, ctrl._marker
    ctrl.disconnect()
    assert annot not in ax.get_children()
    assert marker not in ax.get_children()
    assert ctrl._cids == []


def test_large_array_motion_resolves():
    n = 1_000_000
    x = np.linspace(0.0, 1000.0, n)
    y = np.sin(x / 50.0) * 100.0
    fig, ax, canvas = _axes()
    ax.set_xlim(0.0, 1000.0)
    ax.set_ylim(-110.0, 110.0)
    canvas.draw()
    ctrl = picks.HoverReadoutController(canvas, ax, x, y, lambda i: f"i={i}")
    k = 500_000
    ctrl._on_motion(_motion(canvas, ax, x[k], y[k]))
    assert ctrl._annot.get_visible()
    assert abs(int(ctrl._annot.get_text()[2:]) - k) < 3000  # ~1 px wide, many samples per pixel
