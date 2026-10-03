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
import pandas as pd
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import interpret, picks, plots, resample
from dfit_tool.io_load import TestData as IoTestData
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


def test_render_overview_override_past_guard_uses_trim_dt_not_guard_dt():
    """Sibling of the stale-trim test above, same past-guard setup, but with
    state.tail_guard_override = True -- a deliberate drag past the guard, tracked by the new
    flag -- so interpret.resolve_tail_cut_dt now respects tail_trim_dt unclamped instead of
    falling back to guard_dt."""
    td, st, res = _seeded_with_crash()
    guard_dt = 40.0
    res.resampled_full = resample.Resampled(dt=np.array([0.0, 20.0, 40.0]),
                                            p=np.array([5000.0, 4800.0, 4700.0]),
                                            n_raw=res.resampled_full.n_raw,
                                            guard_dt=guard_dt)
    st.tail_trim_dt = guard_dt + 3600.0  # same "stale-looking" position as the test above
    st.tail_guard_override = True        # ...but now flagged as a deliberate override

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    vline = _gid(ax, "tail_trim")
    expected_x = (st.tail_trim_dt + res.t_shutin_s) / 3600.0
    assert vline.get_xdata()[0] == pytest.approx(expected_x)


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


def _seeded_with_guard_fire_and_more_data_after():
    """A record whose rise guard fires partway through the post-shut-in decline, with a further
    decline AFTER the rise -- unlike test_rise_guard.py's own fixture, this leaves real data (and
    real resampled points, once model.compute_all's stop_at_guard=False keeps consuming past the
    guard) past guard_dt to drag onto, exercising the actual bug being fixed: dragging the
    Overview trim line later than a fired guard must find real samples to snap to."""
    start_idx, shutin_idx = 50, 100
    decline1_len = 300
    rise_len = 80
    decline2_len = 300
    n = shutin_idx + decline1_len + rise_len + decline2_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + decline1_len] = np.linspace(5000.0, 3000.0, decline1_len)
    rise_idx = shutin_idx + decline1_len
    pressure[rise_idx:rise_idx + rise_len] = pressure[rise_idx - 1] + 100.0
    decline2_start = rise_idx + rise_len
    pressure[decline2_start:] = np.linspace(pressure[decline2_start - 1], 1000.0, decline2_len)

    df = pd.DataFrame({"PRESSURE": pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col="PRESSURE", rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)
    return td, st, res


def test_overview_wiring_drag_past_guard_sets_override_and_admits_data():
    """End-to-end through the real _attach_controllers closure: dragging the tail-trim line past
    a fired guard's boundary must (1) actually land the trim there -- not clamp/clear, the
    original bug -- (2) flag it as a deliberate override, and (3) make that previously-excluded
    data show up in a follow-up compute_all's res.resampled/res.diagnostics."""
    td, st, res = _seeded_with_guard_fire_and_more_data_after()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None  # sanity: the guard actually fired

    baseline = compute_all(st, td)  # no trim/override yet -- clamped at guard_dt
    baseline_n = len(baseline.resampled.dt)
    assert baseline.resampled.dt[-1] <= guard_dt

    stub, commit = _trim_commit(td, st, res)
    # Drag to a resampled sample comfortably past the guard, in the decline-after-rise segment --
    # but not the very last sample, which the commit closure treats as "release at the end", i.e.
    # clear the trim (see test_overview_trim_commit_release_at_last_sample_clears above).
    past_guard = res.resampled_full.dt[res.resampled_full.dt > guard_dt]
    assert len(past_guard) >= 2  # sanity: there's real data past the guard to pick a mid-point of
    target_dt = float(past_guard[len(past_guard) // 2])
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)

    assert st.tail_trim_dt is not None
    assert st.tail_trim_dt == pytest.approx(target_dt)
    assert st.tail_guard_override is True

    after = compute_all(st, td)
    assert after.resampled.dt[-1] > guard_dt
    assert len(after.resampled.dt) > baseline_n


def _seeded_with_guard_fire_rise_never_comes_back():
    """The actual reported bug shape, pinned precisely this time: the rise fires the guard and
    then never moves again for the rest of the record (a stuck sensor, not a rebound and not a
    continuing ramp). A prior round's fixture of the same intent (``..._rise_stays_elevated_then_
    declines``) was found by review to actually decline again afterward, so it wasn't really
    testing this shape at all.

    Under the bidirectional (+-step) keep rule, resample_pressure_increment's resampled_full now
    generally keeps the excursion's OWN first sample too (a >= step move off the last pre-guard
    kept point) -- so a fixture meant to test "genuinely nothing past guard_dt to land on" has to
    make sure even that one boundary sample isn't keepable. ``resample_step`` (unlike the fixed,
    guard-detecting ``rise_tol``) is caller-configurable and doesn't affect guard detection at
    all, so setting it well above the fixed +100 psi rise here (300 vs 100) guarantees nothing --
    not even the excursion's own jump -- ever moves far enough to get kept: resampled_full keeps
    ZERO points at or past guard_dt, and stop_at_guard=False/True produce identical resampled_full
    past this point. The override this test proves out therefore has to work purely off the raw
    record (ui.py's commit_trim raw-sample snap), with nothing already-resampled past the guard to
    land on."""
    start_idx, shutin_idx = 50, 100
    decline_len = 300
    rise_len = 300
    n = shutin_idx + decline_len + rise_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0

    pressure = np.full(n, 1000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + decline_len] = np.linspace(5000.0, 3000.0, decline_len)
    rise_idx = shutin_idx + decline_len
    # Jumps once, then holds perfectly flat forever -- never moves again.
    pressure[rise_idx:] = pressure[rise_idx - 1] + 100.0

    df = pd.DataFrame({"PRESSURE": pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col="PRESSURE", rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx, resample_step=300.0)
    res = compute_all(st, td)
    return td, st, res


def test_overview_wiring_drag_past_guard_works_when_rise_never_comes_back():
    """The hardest, most realistic shape, and the one that actually pins the original reported
    bug: a rise that fires the guard and never comes back down. resampled_full has genuinely
    nothing new past guard_dt, so the override must work off the raw record alone -- proving the
    revert's fix (ui.py's raw-sample snap in commit_trim) rather than relying on the resampler
    having already found a point to land on."""
    td, st, res = _seeded_with_guard_fire_rise_never_comes_back()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None  # sanity: the guard actually fired
    # Confirms this really is the "never comes back" shape: resampled_full has nothing new
    # at or past the guard at all.
    assert not np.any(res.resampled_full.dt >= guard_dt)
    assert any("Tail guard stopped resampling" in w for w in res.warnings)

    stub, commit = _trim_commit(td, st, res)
    raw_dt_post = td.t_s[td.t_s >= res.t_shutin_s] - res.t_shutin_s
    # Drag all the way to the raw record's last sample -- the extreme case, where the commit
    # closure's non-override branch would clear the trim to None, but the override branch must
    # not (clearing would fall back through resolve_tail_cut_dt to guard_dt again).
    target_dt = float(raw_dt_post[-1])
    assert target_dt > guard_dt  # sanity: actually past the guard
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)

    assert st.tail_trim_dt is not None
    assert st.tail_trim_dt == pytest.approx(target_dt)
    assert st.tail_guard_override is True

    after = compute_all(st, td)
    # The guard-stopped warning is suppressed once the override moves the resolved cutoff past
    # the guard -- here (the "never comes back" shape) that's because nothing new was actually
    # admitted, so it's replaced by a distinct honest warning instead (Finding 1, see the
    # dedicated tests right below); the resolved cutoff sits at the dragged raw time regardless.
    assert not any("Tail guard stopped resampling" in w for w in after.warnings)
    resolved_cutoff = interpret.resolve_tail_cut_dt(st.tail_trim_dt, after.resampled_full.guard_dt,
                                                     st.tail_guard_override)
    assert resolved_cutoff == pytest.approx(target_dt)

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, after, interactive=True)
    vline = _gid(ax, "tail_trim")
    expected_x = (target_dt + after.t_shutin_s) / 3600.0
    assert vline.get_xdata()[0] == pytest.approx(expected_x)


# --------------------------------------------------------------------------------------------------
# Finding 1 (this round): the override's messaging must be honest about whether it actually
# admitted any new data, not just about whether the cutoff moved -- see model.py's
# admitted_new_data/override_extends_past_guard.
# --------------------------------------------------------------------------------------------------
def test_overview_wiring_drag_past_guard_rise_never_comes_back_admits_nothing_and_warns_honestly():
    """The regression the review caught: the earlier round's test above only checked that the
    original guard-stopped warning was absent, never whether the override actually changed
    anything. It doesn't -- res.resampled/res.diagnostics stay numerically identical to the
    no-override default -- and compute_all must say so via a distinct, honest warning rather
    than just falling silent (which reads as "the override worked")."""
    td, st, res = _seeded_with_guard_fire_rise_never_comes_back()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None
    baseline = compute_all(st, td)  # no trim/override yet -- the guard-clamped default

    stub, commit = _trim_commit(td, st, res)
    raw_dt_post = td.t_s[td.t_s >= res.t_shutin_s] - res.t_shutin_s
    target_dt = float(raw_dt_post[-1])  # drag all the way to the raw record's last sample
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_guard_override is True

    after = compute_all(st, td)
    # No data was actually admitted: point counts and values are unchanged from the default.
    assert len(after.resampled.dt) == len(baseline.resampled.dt)
    np.testing.assert_array_equal(after.resampled.dt, baseline.resampled.dt)
    np.testing.assert_array_equal(after.resampled.p, baseline.resampled.p)
    assert len(after.diagnostics.G) == len(baseline.diagnostics.G)

    assert not any("Tail guard stopped resampling" in w for w in after.warnings)  # superseded
    assert not any(w.startswith("Tail trimmed") for w in after.warnings)  # would confusingly pair
    assert any("Tail-guard override to" in w for w in after.warnings)


def test_overview_wiring_drag_past_guard_with_more_data_after_admits_real_data():
    """Sibling of the test above, using the OTHER guard-fire fixture (a genuine further decline
    resumes past the guard, so there's real data to admit): the override actually changes
    res.resampled/res.diagnostics, the ordinary "Tail trimmed" warning is present reporting the
    later real cutoff, and the "admits nothing" warning above must NOT appear -- this is the
    case it exists to distinguish from."""
    td, st, res = _seeded_with_guard_fire_and_more_data_after()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None
    baseline = compute_all(st, td)  # no trim/override yet -- the guard-clamped default

    stub, commit = _trim_commit(td, st, res)
    past_guard = res.resampled_full.dt[res.resampled_full.dt > guard_dt]
    assert len(past_guard) >= 2  # sanity: real data past the guard to pick a mid-point of
    target_dt = float(past_guard[len(past_guard) // 2])
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)

    after = compute_all(st, td)
    assert len(after.resampled.dt) > len(baseline.resampled.dt)
    assert after.resampled.dt[-1] > guard_dt

    assert not any("Tail guard stopped resampling" in w for w in after.warnings)
    assert not any("Tail-guard override to" in w for w in after.warnings)
    assert any(w.startswith("Tail trimmed") for w in after.warnings)


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


def test_overview_trim_commit_at_or_before_guard_snaps_before_guard_without_override():
    """A drag target at/before guard_dt snaps to a raw sample strictly before the guard and never
    sets tail_guard_override."""
    td, st, res = _seeded_with_guard_fire_and_more_data_after()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None
    stub, commit = _trim_commit(td, st, res)
    pre_guard = res.resampled_full.dt[res.resampled_full.dt < guard_dt]
    assert len(pre_guard) >= 2  # sanity: real pre-guard resampled points to snap to
    target_dt = float(pre_guard[len(pre_guard) // 2])
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(target_dt)
    assert st.tail_guard_override is False


def test_overview_trim_commit_past_guard_always_uses_raw_snap_and_never_none():
    """A drag target past guard_dt always uses the raw-sample snap (ui.py's override branch) and
    always sets an explicit trim -- even at the raw record's very last sample, which the
    non-override branch would otherwise clear to None."""
    td, st, res = _seeded_with_guard_fire_rise_never_comes_back()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None
    stub, commit = _trim_commit(td, st, res)
    raw_dt_post = td.t_s[td.t_s >= res.t_shutin_s] - res.t_shutin_s
    target_dt = float(raw_dt_post[-1])  # the raw record's last sample
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt is not None
    assert st.tail_trim_dt == pytest.approx(target_dt)
    assert st.tail_guard_override is True


def test_overview_trim_commit_release_mid_record_snaps_to_nearest_sample():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    mid = len(res.resampled_full.dt) // 2
    x_hours = (float(res.resampled_full.dt[mid]) + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[mid]))


def test_overview_trim_commit_snaps_to_raw_samples_between_sparse_kept_points():
    """Regression (AEF 05-61-34-5649B): on a slow falloff the 30-psi kept points are hours apart,
    so snapping a drag to them left only two places to land. A drag between two kept points must
    land on the nearest raw sample."""
    td = make_testdata(n=1200, zero_crash_at=None)
    p = td.df["PRESSURE"].to_numpy(copy=True)
    p[300 + 300:300 + 700] = p[300 + 300]   # flat from dt=300 to dt=700
    p[300 + 700:] = 50.0                    # crash at dt=700
    td.df["PRESSURE"] = p
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    assert not np.any((res.resampled_full.dt > 300.0) & (res.resampled_full.dt < 700.0))

    stub, commit = _trim_commit(td, st, res)
    commit((500.4 + res.t_shutin_s) / 3600.0)

    assert st.tail_trim_dt == pytest.approx(500.0)
    assert st.tail_guard_override is False


def test_overview_trim_commit_release_before_shutin_clamps_to_index_2():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    x_hours = (res.t_shutin_s - 3600.0) / 3600.0  # 1h before shut-in -> negative dt_target
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[2]))


def test_overview_trim_commit_else_branch_never_crosses_into_override_territory():
    """Finding 2 (this round): the non-override branch (drag target at/before guard_dt, or no
    guard at all) must restrict its snap candidates to samples at/before the guard -- dt_full
    itself can carry points PAST guard_dt now (stop_at_guard=False), so an unmasked nearest
    search can silently pick one of those whenever a sparse pre-guard decline sits next to a
    denser post-guard one, setting tail_guard_override even though the analyst never dragged
    past the guard. Here the drag target (450) sits well before guard_dt (500), but the global
    nearest sample under the OLD (unmasked) code would have been one of the dense post-guard
    points (500.1+, diff ~50) rather than the sparse pre-guard ones (200, diff 250)."""
    td, st, res = _seeded_with_crash()
    guard_dt = 500.0
    dt_full = np.array([0.0, 100.0, 200.0,                      # sparse pre-guard
                         500.1, 500.2, 500.3, 500.4, 500.5])     # dense, right past the guard
    res.resampled_full = resample.Resampled(dt=dt_full, p=np.linspace(5000.0, 4000.0, len(dt_full)),
                                            n_raw=res.resampled_full.n_raw, guard_dt=guard_dt)

    stub, commit = _trim_commit(td, st, res)
    target_dt = 450.0
    assert target_dt <= guard_dt  # sanity: exercises the non-override ("else") branch
    x_hours = (target_dt + res.t_shutin_s) / 3600.0
    commit(x_hours)

    # Fixed: never crosses into guard-override territory -- either clears, or lands at/before
    # the guard, and tail_guard_override always stays False.
    assert st.tail_guard_override is False
    assert st.tail_trim_dt is None or st.tail_trim_dt <= guard_dt


def _stub_capturing_hint(td, st, res, step):
    """Same as ``_stub`` above, except ``hint_lbl.config`` calls are recorded (in order) into the
    returned list, so a test can inspect the actual hint text set during wiring -- ``_stub``'s
    own no-op lambda discards it."""
    calls: list[str] = []
    stub = _stub(td, st, res, step)
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: calls.append(kw.get("text")))
    return stub, calls


def test_overview_hint_mentions_guard_override_when_guard_fired():
    """Finding 3 (docs/UI-text only, no behavior change): once a guard has fired for this
    record, releasing at the right edge doesn't just "clear the trim" (the pre-existing wording)
    -- it overrides the guard too, admitting everything past it. The hint must say so."""
    td, st, res = _seeded_with_guard_fire_and_more_data_after()
    assert res.resampled_full.guard_dt is not None  # sanity: the guard actually fired

    stub, calls = _stub_capturing_hint(td, st, res, "overview")
    DfitApp._attach_controllers(stub)

    assert calls, "hint_lbl.config was never called"
    assert "override" in calls[-1].lower() and "guard" in calls[-1].lower()


def test_overview_hint_plain_when_no_guard_fired():
    """No guard in this record -- the plain "clear the trim" wording is accurate on its own and
    must not gain an override clause that doesn't apply."""
    td, st, res = _seeded_with_crash()
    assert res.resampled_full.guard_dt is None  # sanity: no guard fired

    stub, calls = _stub_capturing_hint(td, st, res, "overview")
    DfitApp._attach_controllers(stub)

    assert calls, "hint_lbl.config was never called"
    assert "override" not in calls[-1].lower()


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
