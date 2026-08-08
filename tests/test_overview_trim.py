"""The tail-trim tool, always on at the Overview step (no more "Show trim tool" toggle --
CLAUDE.md TODO). ``PickState.tail_trim_dt`` semantics (shut-in-relative seconds) and
``model.compute_all``'s masking are unchanged -- see tests/test_tail_trim.py, which covers those
and is untouched by this move. This file covers ``plots.render_overview``'s ``interactive``
kwarg and default-cut resolution, and ``ui.DfitApp``'s overview controller wiring (the
``x_hours``-coordinate commit closure).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import picks, plots, resample
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, injection_state, pre_crash_trim_dt


def _gid(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


def _seeded_with_crash(trim_dt=None):
    """A PickState with the injection window seeded and a monotone zero-crash in the
    post-shut-in tail (helpers.make_testdata's zero_crash_at), optionally trimmed. Mirrors
    test_render_constructions.py's (now-removed) gfunction fixture of the same name."""
    td = make_testdata(n=1200, zero_crash_at=0.5)
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    if trim_dt is not None:
        st.tail_trim_dt = trim_dt
        res = compute_all(st, td)
    return td, st, res


# --------------------------------------------------------------------------------------------------
# render_overview: tail_excluded (informational, gated on a trim/guard cut) vs. tail_trim
# (the draggable line, gated on interactive)
# --------------------------------------------------------------------------------------------------
def test_render_overview_trim_set_interactive_false_draws_excluded_not_trim_line():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res2, interactive=False)

    t_trim_h = (last_normal_dt + res2.t_shutin_s) / 3600.0
    # The main pressure trace (no gid) is split at the trim time -- it never reaches past it.
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    assert main.get_xdata().max() <= t_trim_h

    # The excluded tail is the real raw trace's remainder, not a hidden overlay: it starts right
    # at the trim boundary and runs all the way to the record's last raw sample.
    tail = _gid(ax, "tail_excluded")
    tail_x = tail.get_xdata()
    assert len(tail_x)
    last_raw_h = float(td.t_s[-1]) / 3600.0
    assert t_trim_h < tail_x.min() < t_trim_h + 0.01
    assert tail_x.max() == pytest.approx(last_raw_h)
    assert not any(l.get_gid() == "tail_trim" for l in ax.get_lines())


def test_render_overview_trim_set_interactive_true_draws_trim_line_at_trim_x():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res2, interactive=True)

    vline = _gid(ax, "tail_trim")
    assert vline.get_xdata()[0] == pytest.approx((last_normal_dt + res2.t_shutin_s) / 3600.0)
    # The excluded-tail preview is informational and still drawn regardless of interactive.
    assert _gid(ax, "tail_excluded") is not None


def test_render_overview_no_trim_no_guard_vline_at_last_raw_sample_no_excluded_tail():
    """With nothing to cut the line parks at the END OF THE DATA -- the last raw sample, not
    resampled_full.dt[-1]. The resampler keeps a point only per 30-psi drop, so its last kept
    point can sit well short of the record's end; a line parked there would read as a cut that
    isn't in effect, with ungrayed data to its right."""
    td, st, res = _seeded_with_crash()  # no trim set, no guard fired
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    last_raw_h = float(td.t_s[-1]) / 3600.0
    vline = _gid(ax, "tail_trim")
    assert vline.get_xdata()[0] == pytest.approx(last_raw_h)
    assert not any(l.get_gid() == "tail_excluded" for l in ax.get_lines())
    # With no trim, the main trace is unsplit and reaches the full record's last raw sample.
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    assert main.get_xdata().max() == pytest.approx(last_raw_h)


def test_render_overview_parked_line_ignores_a_short_resampled_tail():
    """The case the raw-sample rule exists for: resampled_full ends well short of the record (the
    resampler keeps a point only per 30-psi drop, so a slow falloff's last kept point can lag the
    end by hours). With no cut in effect the line must still park at the raw end, not there."""
    td, st, res = _seeded_with_crash()
    short_dt = np.array([0.0, 30.0, 60.0])          # last kept point ~1 min after shut-in
    res.resampled_full = resample.Resampled(dt=short_dt, p=np.array([5000.0, 4900.0, 4800.0]),
                                            n_raw=res.resampled_full.n_raw)
    assert st.tail_trim_dt is None and res.resampled_full.guard_dt is None

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    last_raw_h = float(td.t_s[-1]) / 3600.0
    last_resampled_h = (float(short_dt[-1]) + res.t_shutin_s) / 3600.0
    assert last_resampled_h < last_raw_h                       # the two genuinely differ here
    assert _gid(ax, "tail_trim").get_xdata()[0] == pytest.approx(last_raw_h)
    assert not any(l.get_gid() == "tail_excluded" for l in ax.get_lines())


def test_render_overview_guard_fired_no_trim_line_at_guard_dt_and_tail_grayed():
    """A guard fire (no trim pick set -- picks.seed_tail_trim never sets one for "rise_guard")
    still moves both the draggable line and the gray-out on Overview -- previously only the
    G-function plot's own guard_excluded preview showed this."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert st.tail_trim_dt is None

    guard_dt = 55.0
    res.resampled_full = resample.Resampled(dt=np.array([0.0, 30.0, 50.0]),
                                            p=np.array([5000.0, 4800.0, 4700.0]),
                                            n_raw=res.resampled_full.n_raw,
                                            guard_dt=guard_dt)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    vline = _gid(ax, "tail_trim")
    expected_x = (guard_dt + res.t_shutin_s) / 3600.0
    assert vline.get_xdata()[0] == pytest.approx(expected_x)
    assert _gid(ax, "tail_excluded") is not None
    assert st.tail_trim_dt is None  # the guard cut is display-only, never a pick


def test_render_overview_stale_trim_past_guard_uses_guard_dt_not_trim_dt():
    """F2: both tail_trim_dt and guard_dt are shut-in-relative, so dragging shut-in later (F1's
    resync only fires from ui.py's controller commit, not from a bare compute_all call, so a
    caller that hand-builds ``res`` like this one can still leave the two out of sync) can leave
    a stale tail_trim_dt sitting PAST guard_dt. Taking tail_trim_dt at face value (the old
    "no min() needed" logic) would render the guard-excluded region as kept -- the opposite of
    what this feature exists to do. The cut must be the EARLIER of the two, so the line sits at
    guard_dt and the guard-excluded tail stays grayed."""
    td, st, res = _seeded_with_crash()
    guard_dt = 40.0
    res.resampled_full = resample.Resampled(dt=np.array([0.0, 20.0, 40.0]),
                                            p=np.array([5000.0, 4800.0, 4700.0]),
                                            n_raw=res.resampled_full.n_raw,
                                            guard_dt=guard_dt)
    st.tail_trim_dt = guard_dt + 3600.0  # stale: an hour past the guard boundary

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    vline = _gid(ax, "tail_trim")
    expected_x = (guard_dt + res.t_shutin_s) / 3600.0
    assert vline.get_xdata()[0] == pytest.approx(expected_x)
    assert _gid(ax, "tail_excluded") is not None


def test_render_overview_no_resampled_full_no_gids_no_crash():
    td = make_testdata()
    st = PickState(pressure_col="PRESSURE")  # no start/shutin picked -> no t_shutin_s/resampled_full
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    assert all(l.get_gid() not in ("tail_excluded", "tail_trim") for l in ax.get_lines())


# --------------------------------------------------------------------------------------------------
# export: render_step_figure always renders with interactive=False, so an exported PNG never
# carries the draggable line
# --------------------------------------------------------------------------------------------------
def test_render_step_figure_overview_never_draws_trim_line_but_draws_excluded_tail():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)

    fig = plots.render_step_figure("overview", td, st, res2)
    ax = fig.axes[0]
    assert not any(l.get_gid() == "tail_trim" for l in ax.get_lines())
    assert any(l.get_gid() == "tail_excluded" for l in ax.get_lines())


def test_render_step_figure_overview_png_yaxis_starts_at_zero():
    """F7: the pinned y-min 0 (CLAUDE.md TODO) must survive into the actual exported artifact,
    not just the live canvas -- test_view_state.py's test_refresh_unions_overview_full_y_down_to_zero
    only covers the live y-slider's valmin, which is a different code path
    (ui.refresh/_build_sliders) from the one that produces the PNG (render_step_figure)."""
    td, st, res = _seeded_with_crash()

    fig = plots.render_step_figure("overview", td, st, res)

    ax = fig.axes[0]
    assert ax.get_ylim()[0] == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------------
# ui.py controller wiring (end to end against the real _attach_controllers closure)
# --------------------------------------------------------------------------------------------------
def _stub(td, st, res, step):
    """Same duck-typed stand-in convention as test_render_constructions.py's ``_stub``."""
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS[step](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td = td
    stub.res = res
    stub.state = st
    stub.step = step
    stub._controllers = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.refresh = lambda: None
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


def test_overview_wiring_attaches_drag_and_hover():
    td, st, res = _seeded_with_crash()
    stub = _stub(td, st, res, "overview")
    DfitApp._attach_controllers(stub)
    assert len(stub._controllers) == 2
    ctrl, hover_ctrl = stub._controllers
    assert isinstance(ctrl, picks.DragLineController)
    assert isinstance(hover_ctrl, picks.HoverCursorController)


def test_overview_wiring_no_resampled_full_no_controllers():
    td = make_testdata()
    st = PickState(pressure_col="PRESSURE")  # no start/shutin -> no t_shutin_s/resampled_full
    res = compute_all(st, td)
    stub = _stub(td, st, res, "overview")
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []


# --------------------------------------------------------------------------------------------------
# commit closure (moved from the old gfunction wiring tests, adapted to x_hours coordinates:
# x_hours = (dt + t_shutin_s) / 3600)
# --------------------------------------------------------------------------------------------------
def _trim_commit(td, st, res):
    stub = _stub(td, st, res, "overview")
    DfitApp._attach_controllers(stub)
    trim_ctrl = next(c for c in stub._controllers if isinstance(c, picks.DragLineController))
    return stub, trim_ctrl.handlers["tail_trim"]


def test_overview_trim_commit_release_at_last_sample_clears():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    x_hours = (float(res.resampled_full.dt[-1]) + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt is None


def test_overview_trim_commit_release_mid_record_snaps_to_nearest_sample():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    mid = len(res.resampled_full.dt) // 2
    x_hours = (float(res.resampled_full.dt[mid]) + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[mid]))


def test_overview_trim_commit_release_before_shutin_clamps_to_index_2():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    x_hours = (res.t_shutin_s - 3600.0) / 3600.0  # 1h before shut-in -> negative dt_target
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[2]))


def test_overview_trim_commit_recovery_two_point_resample_never_indexerrors():
    """Regression: with an exactly-2-point resampled_full, clamping to index 2 BEFORE the clear
    check must never index past the end of dt_full. A mid-plot drag on a record this short can
    only ever clear the trim -- pinned here by pre-setting a real trim and asserting it's
    actually cleared, not just that nothing raised."""
    td = make_testdata()
    dt_full = np.array([0.0, 10.0])
    res = DerivedResults(diagnostics=None, resampled=None,
                         resampled_full=resample.Resampled(dt=dt_full, p=np.array([500.0, 300.0]),
                                                            n_raw=2),
                         t_shutin_s=1000.0)
    st = PickState(tail_trim_dt=5.0)
    stub, commit = _trim_commit(td, st, res)

    commit((res.t_shutin_s + 5.0) / 3600.0)  # a mid-plot x -- must not raise, and must clear

    assert st.tail_trim_dt is None
