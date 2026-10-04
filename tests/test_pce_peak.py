"""PC-E ("no trend"): pore pressure from a -1/2 log-log line extrapolated from the
postclosure t*dP/dt peak, P = Pp + m*t^(-1/2) anchored at the peak, so
m = 2*D_pk*sqrt(t_pk) and Pp = P_pk - 2*D_pk. Replaces the pp_window fit under PC-E.
"""

from __future__ import annotations

import math
import os
import types

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import interpret, picks, plots, store
from dfit_tool.model import PickState, _decode, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata

PCE = "PC-E no trend"


# --------------------------------------------------------------------------------------------------
# interpret
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("i", [0, 7, 19])
def test_pore_pressure_from_peak_recovers_linear_flow_tail(i):
    t = np.geomspace(1e3, 1e6, 20)
    p = 3000.0 + 400.0 * t ** -0.5
    tdpdt = 200.0 * t ** -0.5  # -t*dP/dt for that tail, positive-up
    pp, m = interpret.pore_pressure_from_peak(t, p, tdpdt, i)
    assert pp == pytest.approx(3000.0)
    assert m == pytest.approx(400.0)


@pytest.mark.parametrize("d", [0.0, -5.0, float("nan")])
def test_pore_pressure_from_peak_none_without_positive_derivative(d):
    t = np.array([10.0, 100.0])
    assert interpret.pore_pressure_from_peak(t, np.array([1.0, 2.0]), np.array([d, d]), 1) is None


def _late_peak_series():
    """log10 t*dP/dt with an early spike, a valley, and an equally tall peak 2 samples from
    the end."""
    v = np.array([2.6, 2.4, 2.0, 1.7, 1.6, 1.7, 1.9, 2.1, 2.3, 2.5, 2.6, 2.5, 2.4])
    t = np.geomspace(1.0, 1e5, v.size)
    return t, v


def test_loglog_peak_min_after_finds_late_peak():
    _, v = _late_peak_series()
    assert interpret._loglog_peak(v) != 10
    assert interpret._loglog_peak(v, min_after=1) == 10


def test_suggest_pce_peak_prefers_late_peak_over_early_spike():
    t, v = _late_peak_series()
    assert interpret.suggest_pce_peak_index(t, 10.0 ** v) == 10


def test_suggest_pce_peak_falls_back_to_late_max():
    t = np.geomspace(1.0, 1e5, 12)
    y = np.linspace(10.0, 50.0, 12)  # still rising: no interior peak
    assert interpret.suggest_pce_peak_index(t, y) == 11


# --------------------------------------------------------------------------------------------------
# compute_all
# --------------------------------------------------------------------------------------------------
def _pce_state():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    picks.seed_pp(st, res)
    st.postclosure_scenario = PCE
    st.pp_axis = "tm12"
    dg = res.diagnostics
    i = int(np.nanargmax(np.where(dg.t > 0, dg.tdpdt, -np.inf)))
    st.pce_peak_t = float(dg.t[i])
    return td, st, i


def test_compute_all_pce_uses_peak_extrapolation():
    td, st, i = _pce_state()
    res = compute_all(st, td)
    dg = res.diagnostics
    assert res.pore_pressure == pytest.approx(dg.p[i] - 2.0 * dg.tdpdt[i])
    assert res.pore_pressure_slope == pytest.approx(2.0 * dg.tdpdt[i] * math.sqrt(dg.t[i]))
    assert res.pce_peak_t == pytest.approx(dg.t[i])


def test_compute_all_pce_ignores_pp_window():
    td, st, _ = _pce_state()
    a = compute_all(st, td).pore_pressure
    st.pp_window = (1.0, 2.0)
    assert compute_all(st, td).pore_pressure == pytest.approx(a)


def test_compute_all_pce_without_pick_is_blank():
    td, st, _ = _pce_state()
    st.pce_peak_t = None
    assert compute_all(st, td).pore_pressure is None


def test_compute_all_pca_still_uses_window_fit():
    td, st, _ = _pce_state()
    st.postclosure_scenario = "PC-A linear"
    res = compute_all(st, td)
    assert res.pore_pressure is not None
    assert res.pce_peak_t is None


def test_compute_all_pce_peak_past_last_sample_blank_and_warns():
    td, st, _ = _pce_state()
    st.pce_peak_t = float(compute_all(st, td).diagnostics.t[-1]) * 2.0
    st.pp_window = None
    res = compute_all(st, td)
    assert res.pore_pressure is None
    assert res.pce_peak_t is None
    assert any("PC-E peak" in w for w in res.warnings)


def test_compute_all_pce_tail_trim_before_peak_warns_not_snaps():
    td, st, _ = _pce_state()
    st.pp_window = None
    st.tail_trim_dt = 0.5 * st.pce_peak_t
    res = compute_all(st, td)
    assert res.pore_pressure is None
    assert any("PC-E peak" in w for w in res.warnings)
    assert not any("pore-pressure window" in w for w in res.warnings)


# --------------------------------------------------------------------------------------------------
# picks
# --------------------------------------------------------------------------------------------------
def test_seed_pce_peak_sets_pick_only_under_pce():
    td, st, _ = _pce_state()
    st.pce_peak_t = None
    picks.seed_pce_peak(st, compute_all(st, td))
    assert st.pce_peak_t is not None
    st2 = injection_state(td)
    st2.postclosure_scenario = "PC-A linear"
    picks.seed_pce_peak(st2, compute_all(st2, td))
    assert st2.pce_peak_t is None


def test_seed_pp_no_op_under_pce():
    td = make_testdata()
    st = injection_state(td)
    st.postclosure_scenario = PCE
    picks.seed_pp(st, compute_all(st, td))
    assert st.pp_window is None


def test_decode_old_save_has_no_peak():
    assert _decode({"postclosure_scenario": PCE}).pce_peak_t is None


# --------------------------------------------------------------------------------------------------
# renderers
# --------------------------------------------------------------------------------------------------
def _line(ax, gid):
    return next(ln for ln in ax.get_lines() if ln.get_gid() == gid)


def test_render_loglog_pce_draws_peak_and_half_slope_line():
    td, st, i = _pce_state()
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_loglog(ax, td, st, res)
    dg = res.diagnostics
    pk = _line(ax, "pce_peak")
    assert pk.get_xdata()[0] == pytest.approx(dg.t[i])
    ln = _line(ax, "pce_halfslope")
    x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
    assert x[0] == pytest.approx(dg.t[i]) and y[0] == pytest.approx(dg.tdpdt[i])
    slope = (math.log10(y[-1]) - math.log10(y[0])) / (math.log10(x[-1]) - math.log10(x[0]))
    assert slope == pytest.approx(-0.5)
    assert not ax.patches
    assert "From Peak" in ax.get_title()


def test_render_porepressure_pce_draws_extrapolation_to_intercept():
    td, st, i = _pce_state()
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_porepressure(ax, td, st, res)
    ln = _line(ax, "pce_extrapolation")
    x, y = np.asarray(ln.get_xdata()), np.asarray(ln.get_ydata())
    dg = res.diagnostics
    assert x[0] == pytest.approx(dg.t[i] ** -0.5) and y[0] == pytest.approx(dg.p[i])
    assert x[-1] == 0.0 and y[-1] == pytest.approx(res.pore_pressure)
    assert not ax.patches
    assert "From Peak" in ax.get_title()


def test_summary_pce_rows_describe_peak_not_window():
    from dfit_tool import summary
    td, st, _ = _pce_state()
    st.step_status = {"loglog": "done", "porepressure": "done"}
    res = compute_all(st, td)
    post = next(s for s in summary.summary_sections(st, res) if s.title == "Postclosure")
    items = [r[0] for r in post.rows]
    assert "window (min)" not in items
    assert ["method", "-1/2 from peak"] in post.rows
    assert ["peak (min)", f"{res.pce_peak_t / 60.0:.2f}"] in post.rows


# --------------------------------------------------------------------------------------------------
# log row
# --------------------------------------------------------------------------------------------------
def _row(tmp_path, st, td):
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                               st, td, res)


def test_log_row_pce_peak_min(tmp_path):
    td, st, _ = _pce_state()
    assert store.LOG_COLUMNS[-1] == "pce_peak_min"
    row = _row(tmp_path, st, td)
    assert row["pce_peak_min"] == pytest.approx(compute_all(st, td).pce_peak_t / 60.0)
    st.postclosure_scenario = "PC-A linear"
    assert _row(tmp_path, st, td)["pce_peak_min"] is None


# --------------------------------------------------------------------------------------------------
# ui wiring
# --------------------------------------------------------------------------------------------------
def _stub(td, st, step):
    res = compute_all(st, td)
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS[step](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td, stub.res, stub.state, stub.step = td, res, st, step
    stub._controllers = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.refresh = lambda: None
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


def test_loglog_pce_wires_draggable_peak_and_commit_moves_it():
    td, st, i = _pce_state()
    stub = _stub(td, st, "loglog")
    DfitApp._attach_controllers(stub)
    ctrls = [c for c in stub._controllers if isinstance(c, picks.DraggablePointController)]
    assert len(ctrls) == 1
    assert not any(isinstance(c, picks.SpanController) for c in stub._controllers)
    new_t = float(stub.res.diagnostics.t[i - 3])
    ctrls[0].commit_fn(new_t)
    assert st.pce_peak_t == new_t


def _goto_stub(td, st):
    stub = types.SimpleNamespace()
    stub.td, stub.state, stub.step = td, st, "overview"
    stub._seed_step = lambda key: None
    stub.refresh = lambda: None
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.gate_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub._ensure_pce_peak = types.MethodType(DfitApp._ensure_pce_peak, stub)
    stub._goto = types.MethodType(DfitApp._goto, stub)
    return stub


def test_goto_any_step_seeds_missing_pce_peak():
    td, st, _ = _pce_state()
    st.pce_peak_t = None
    st.step_status = {k: "visited" for k in ("overview", "injection", "isip", "gfunction",
                                             "tangent", "loglog", "porepressure", "stiffness")}
    stub = _goto_stub(td, st)
    stub._goto("stiffness")
    assert st.pce_peak_t is not None


def test_ensure_pce_peak_reseeds_invalid_pick():
    td, st, _ = _pce_state()
    good = st.pce_peak_t
    st.pce_peak_t = float(compute_all(st, td).diagnostics.t[-1]) * 2.0
    stub = _goto_stub(td, st)
    stub._ensure_pce_peak()
    assert st.pce_peak_t == pytest.approx(good)


def test_render_loglog_pce_extension_does_not_widen_datalim():
    td, st, _ = _pce_state()
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_loglog(ax, td, st, res)
    assert ax.dataLim.x1 <= float(res.diagnostics.t[-1]) * 1.0001


def test_summary_pce_gates_on_porepressure_like_chart():
    from dfit_tool import summary
    td, st, _ = _pce_state()
    st.step_status = {"loglog": "done"}
    res = compute_all(st, td)
    post = next(s for s in summary.summary_sections(st, res) if s.title == "Postclosure")
    assert ["pore pressure (psi)", summary.DASH] in post.rows


def test_porepressure_pce_wires_no_span():
    td, st, _ = _pce_state()
    stub = _stub(td, st, "porepressure")
    DfitApp._attach_controllers(stub)
    assert not any(isinstance(c, picks.SpanController) for c in stub._controllers)
