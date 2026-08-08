"""Manual-only tail trim (CLAUDE.md TODO #3): PickState.tail_trim_dt masks the resampled record
before diagnostics; model.compute_all keeps the untrimmed arrays around (resampled_full/G_full)
so the renderer/controller can always recover from a pathological trim."""

from __future__ import annotations

import types

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import picks, plots, resample
from dfit_tool.interpret import MIN_SURFACE_PRESSURE_PSI, suggest_tail_trim_dt
from dfit_tool.model import PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, injection_state, pre_crash_trim_dt


def _seeded_with_crash(zero_crash_at: float = 0.5):
    """A test whose post-shut-in record crashes monotonically to 0 psi partway through --
    the motivating bug: the tail guard in resample.resample_pressure_increment only stops on a
    late *rise*, so a monotone crash sails through untouched."""
    td = make_testdata(n=1200, zero_crash_at=zero_crash_at)
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    return td, st, res


def test_tail_trim_dt_defaults_to_none():
    assert PickState().tail_trim_dt is None


def test_no_trim_resampled_equals_resampled_full_and_g_equals_g_full():
    td, st, res = _seeded_with_crash()
    assert st.tail_trim_dt is None
    assert np.array_equal(res.resampled.dt, res.resampled_full.dt)
    assert np.array_equal(res.resampled.p, res.resampled_full.p)
    assert np.array_equal(res.diagnostics.G, res.G_full)


def test_untrimmed_crash_reaches_zero_in_resampled_p():
    """Documents the bug this feature exists to fix: with no trim, the crash is not excluded."""
    td, st, res = _seeded_with_crash()
    assert float(np.nanmin(res.resampled.p)) == pytest.approx(0.0, abs=1e-6)


def test_trim_before_crash_excludes_it_but_resampled_full_still_has_it():
    td, st, res = _seeded_with_crash()
    # The crash starts partway through the post-shut-in record; trim just before it drops below
    # 1000 psi, derived from the data rather than a hardcoded index.
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)

    assert res2.resampled.dt.max() == pytest.approx(last_normal_dt)
    assert float(np.nanmin(res2.resampled.p)) > 100.0  # crash excluded
    assert float(np.nanmin(res2.resampled_full.p)) == pytest.approx(0.0, abs=1e-6)  # still there


def test_trim_at_last_point_is_equivalent_to_no_trim():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = float(res.resampled_full.dt[-1])
    res2 = compute_all(st, td)

    assert np.array_equal(res2.resampled.dt, res.resampled_full.dt)
    assert np.array_equal(res2.resampled.p, res.resampled_full.p)


def test_too_tight_trim_leaves_no_diagnostics_and_warns():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = 0.0  # keeps only the dt=0 reference point
    res2 = compute_all(st, td)

    assert res2.diagnostics is None
    assert any("Tail trim leaves only" in w for w in res2.warnings)


def test_too_tight_trim_warning_is_inserted_first():
    """warn_lbl now stacks every warning, but the escape instruction for an otherwise-blank plot
    must still land topmost, ahead of other warnings queued before it in compute_all."""
    td, st, res = _seeded_with_crash()
    st.pressure_is_bhp = False
    st.density_ppg = None  # also trips "Surface pressure selected but density/TVD not set"
    st.tail_trim_dt = 0.0
    res2 = compute_all(st, td)

    assert res2.warnings[0].startswith("Tail trim leaves only")


def test_stale_contact_pick_beyond_trim_warns():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    edge = res2.diagnostics.G[-1]

    st.contact_G = float(edge) + 1.0
    res3 = compute_all(st, td)

    assert any("beyond the tail trim" in w for w in res3.warnings)
    assert any("contact" in w for w in res3.warnings)


def test_stale_pick_warning_absent_when_no_trim_set():
    """Gated on state.tail_trim_dt being set -- a reloaded save against a shorter *source* (no
    trim) must not be misreported as "beyond the tail trim"."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert st.tail_trim_dt is None

    st.contact_G = float(res.diagnostics.G[-1]) + 5.0  # would be "stale" if a trim were set
    res2 = compute_all(st, td)

    assert not any("beyond the tail trim" in w for w in res2.warnings)


def test_pp_window_beyond_trim_warns():
    """The pore-pressure fit masks its window against the diagnostics' post-shut-in time array
    (dg.t), not G -- a finite pp_window upper bound beyond the trimmed tail must still surface
    as a stale pick, even though it's a different axis than contact/min-dP/dG/closure."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)
    edge_t = res2.diagnostics.t[-1]

    st.pp_window = (10.0, edge_t + 100.0)
    res3 = compute_all(st, td)

    assert any("pore-pressure window" in w for w in res3.warnings)
    assert any("beyond the tail trim" in w for w in res3.warnings)


def test_pp_window_open_ended_is_never_stale():
    """A pp_window with an infinite upper bound ("to the end of the data") naturally shrinks
    with the trim instead of going stale -- never flagged, as long as the *lower* bound is
    still inside the trimmed data."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    st.pp_window = (10.0, float("inf"))
    res2 = compute_all(st, td)

    assert not any("pore-pressure window" in w for w in res2.warnings)


def test_pp_window_open_ended_but_lower_bound_beyond_trim_warns():
    """handle_pp_span sets t_hi = inf for the natural "from here to the end" gesture -- an
    open-ended upper bound is exempt on its own, but if the *lower* bound also lies beyond the
    trimmed tail, the fit still empties (masks nothing) with no other warning."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)
    edge_t = res2.diagnostics.t[-1]

    st.pp_window = (edge_t + 50.0, float("inf"))
    res3 = compute_all(st, td)

    assert any("pore-pressure window" in w for w in res3.warnings)
    assert any("beyond the tail trim" in w for w in res3.warnings)


def test_pp_window_lower_bound_in_last_inter_sample_gap_warns():
    """The stale check is outcome-based (mask count), not bound-vs-edge: a lower bound that
    lands strictly between the last two surviving samples is still "<= edge" but leaves the
    same <2-sample mask the pp block itself requires to fit at all -- must still warn."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)
    t = res2.diagnostics.t
    lo_in_last_gap = (t[-2] + t[-1]) / 2.0
    assert lo_in_last_gap < t[-1]  # sanity: strictly inside the data, not beyond the edge

    st.pp_window = (lo_in_last_gap, float("inf"))
    res3 = compute_all(st, td)

    assert any("pore-pressure window" in w for w in res3.warnings)
    assert any("beyond the tail trim" in w for w in res3.warnings)


def test_apply_closure_scenario_leaves_tail_trim_untouched():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)
    trim = st.tail_trim_dt

    st.closure_scenario = "C-A clear"
    picks.apply_closure_scenario(st, res2)

    assert st.tail_trim_dt == pytest.approx(trim)


# --------------------------------------------------------------------------------------------------
# interpret.suggest_tail_trim_dt -- the default cut for the always-on Overview trim line
# --------------------------------------------------------------------------------------------------
def test_suggest_tail_trim_dt_crash_only():
    dt = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    p = np.array([5000.0, 4000.0, 200.0, 50.0, 10.0])
    assert suggest_tail_trim_dt(dt, p, None) == (pytest.approx(30.0), "low_pressure")


def test_suggest_tail_trim_dt_guard_only():
    dt = np.array([0.0, 10.0, 20.0])
    p = np.array([5000.0, 4900.0, 4800.0])  # never crashes
    assert suggest_tail_trim_dt(dt, p, guard_dt=15.0) == (pytest.approx(15.0), "rise_guard")


def test_suggest_tail_trim_dt_both_earliest_wins_low_pressure():
    dt = np.array([0.0, 10.0, 20.0, 30.0])
    p = np.array([5000.0, 4000.0, 50.0, 40.0])  # crashes at dt=20
    assert suggest_tail_trim_dt(dt, p, guard_dt=25.0) == (pytest.approx(20.0), "low_pressure")


def test_suggest_tail_trim_dt_both_earliest_wins_rise_guard():
    dt = np.array([0.0, 10.0, 20.0, 30.0])
    p = np.array([5000.0, 4000.0, 3000.0, 50.0])  # crashes at dt=30, guard fires first
    assert suggest_tail_trim_dt(dt, p, guard_dt=15.0) == (pytest.approx(15.0), "rise_guard")


def test_suggest_tail_trim_dt_tie_favors_rise_guard():
    dt = np.array([0.0, 10.0, 20.0])
    p = np.array([5000.0, 4000.0, 50.0])  # crash lands at the same dt as the guard
    assert suggest_tail_trim_dt(dt, p, guard_dt=20.0) == (pytest.approx(20.0), "rise_guard")


def test_suggest_tail_trim_dt_neither():
    dt = np.array([0.0, 10.0, 20.0])
    p = np.array([5000.0, 4900.0, 4800.0])
    assert suggest_tail_trim_dt(dt, p, None) == (None, "")


def test_suggest_tail_trim_dt_bhp_channel_ignores_crash():
    """A caller passes p_surface_post=None when the mapped channel is already BHP -- a
    sub-100-psi test is meaningless on a bottomhole gauge, so the crash candidate is skipped
    entirely rather than evaluated against a BHP array."""
    dt = np.array([0.0, 10.0, 20.0])
    assert suggest_tail_trim_dt(dt, None, None) == (None, "")
    assert suggest_tail_trim_dt(dt, None, 15.0) == (pytest.approx(15.0), "rise_guard")


def test_suggest_tail_trim_dt_nan_pressure_not_treated_as_below_floor():
    """F7: NaN < floor_psi is False in numpy (NaN comparisons are always False), so a dropout
    sample should already be excluded by plain "<" -- but pin it explicitly, since a naive
    rewrite (e.g. via np.nan_to_num or a sign flip) could easily make a NaN register as a
    crash. The real sub-floor sample two steps later must be the one that's found."""
    dt = np.array([0.0, 10.0, 20.0, 30.0])
    p = np.array([5000.0, np.nan, 4000.0, 50.0])
    assert suggest_tail_trim_dt(dt, p, None) == (pytest.approx(30.0), "low_pressure")


# --------------------------------------------------------------------------------------------------
# picks.seed_tail_trim -- the Overview step's park-and-apply seeder (not in SEEDERS; ui._seed_step
# calls it directly after re-deriving res from the just-seeded injection window)
# --------------------------------------------------------------------------------------------------
def test_seed_tail_trim_non_destructive_over_existing_trim():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = 12345.0
    st.tail_trim_reason = "low_pressure"

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt == pytest.approx(12345.0)
    assert st.tail_trim_reason == "low_pressure"


def test_seed_tail_trim_sets_reason_and_excludes_the_crash():
    """The strict property: every raw sample the trim keeps is at or above the 100-psi floor. An
    "at or before the cut" snap would leave the crash's own sample in (it is usually the LAST
    point the resampler kept, since the flat tail after it never drops another 30 psi) -- see
    picks.seed_tail_trim's side="left" comment."""
    td, st, res = _seeded_with_crash()
    assert st.tail_trim_dt is None
    crash_floor = float(np.nanmin(res.resampled_full.p))  # the crash reaches ~0 psi untrimmed
    assert crash_floor < MIN_SURFACE_PRESSURE_PSI

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt is not None
    assert st.tail_trim_reason == "low_pressure"
    # No KEPT raw post-shut-in sample may sit below the floor -- the whole point of the trim.
    kept = (td.t_s >= res.t_shutin_s) & (td.t_s <= res.t_shutin_s + st.tail_trim_dt)
    surf_kept = td.pressure_surface(st.channel_config())[kept]
    assert float(np.nanmin(surf_kept)) >= MIN_SURFACE_PRESSURE_PSI
    # ... and the trimmed resampled record no longer reaches the crash floor either.
    res2 = compute_all(st, td)
    assert float(np.nanmin(res2.resampled.p)) > crash_floor + 50.0


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_seed_tail_trim_bails_for_short_resampled_full(n):
    """F7/F4: dt_full lengths 0-3 can never satisfy the >=3-kept-points bail check (idx < 2 or
    idx >= len(dt_full) - 1 always holds for these lengths -- there's no room for a trim that
    both keeps >=3 points and excludes anything), regardless of where the crash lands, so all
    four must leave the pick unset rather than raise or clamp into an out-of-range index."""
    td, st, res = _seeded_with_crash()  # a real sub-100-psi crash exists in the raw record
    dt_full = np.arange(n, dtype=float) * 5.0
    p_full = 5000.0 - 100.0 * np.arange(n, dtype=float)
    res.resampled_full = resample.Resampled(dt=dt_full, p=p_full, n_raw=res.resampled_full.n_raw)

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""


def test_seed_tail_trim_trims_when_the_crash_is_the_last_kept_point():
    """Regression: the ordinary crashed record. The resampler keeps a point at every >=30 psi
    drop, so the crash cliff is kept -- and it is usually the LAST point kept, because the flat
    ~0 psi tail after it never drops another 30 psi. An "at or before the cut" snap lands on
    dt_full[-1], trips the past-the-end bail, and sets no trim at all, making the seeder a
    near-no-op exactly where it matters most."""
    td, st, res = _seeded_with_crash(zero_crash_at=None)  # base record, no real crash
    dt_full = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
    res.resampled_full = resample.Resampled(
        dt=dt_full, p=np.array([5000.0, 4900.0, 4800.0, 4700.0, 4600.0]),
        n_raw=res.resampled_full.n_raw)
    # Raw samples are 1 s apart, so this puts the crash's dt exactly at dt_full[-1].
    td.df.loc[st.shutin_idx + 20, "PRESSURE"] = 5.0

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt == pytest.approx(15.0)  # dt_full[-2], NOT no-trim
    assert st.tail_trim_reason == "low_pressure"


def test_seed_tail_trim_sets_no_pick_for_guard_only_record():
    """A guard fire alone (no sub-100-psi crash) must set no pick at all -- the resampler already
    excluded that data, so only the rendered line position/gray-out need to reflect it
    (plots.render_overview), not a stored trim."""
    td, st, res = _seeded_with_crash(zero_crash_at=None)  # helpers.make_testdata's own default:
                                                           # declines toward ~3500 psi, never below
                                                           # the 100-psi floor
    rf = res.resampled_full
    res.resampled_full = resample.Resampled(dt=rf.dt, p=rf.p, n_raw=rf.n_raw, guard_dt=55.0)

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""


def test_seed_tail_trim_bails_when_too_few_points_precede_the_crash():
    """A crash landing before dt_full's index 2 must set NO trim, not clamp up to index 2.

    Clamping (the old behavior) can pick idx=2 even though dt_full[2] sits AT OR PAST the crash
    -- here the crash is at dt=1s and dt_full[2]=10s, so a clamped trim of 10.0 would keep the
    50-psi crashed sample at dt=1s inside the "trimmed" record, and the "Tail auto-trimmed"
    message would then name a point where pressure never actually fell below the floor. Setting
    no trim is strictly better than setting a wrong one: the separate low-surface-pressure
    warning (which scans raw samples, not kept ones) still covers this record, so nothing goes
    silent -- it just isn't mislabeled as fixed."""
    td, st, res = _seeded_with_crash(zero_crash_at=None)  # base record, no real crash
    shutin_idx = st.shutin_idx
    td.df.loc[shutin_idx + 1, "PRESSURE"] = 50.0  # first post-shut-in sample below the floor,
                                                   # 1 s after shut-in -- well before dt_full[2]
    res.resampled_full = resample.Resampled(dt=np.array([0.0, 5.0, 10.0, 15.0, 20.0]),
                                            p=np.array([5000.0, 4900.0, 4800.0, 4700.0, 4600.0]),
                                            n_raw=res.resampled_full.n_raw)

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""


# --------------------------------------------------------------------------------------------------
# picks.commit_tail_trim -- a manual drag/clear always overrides an auto-attributed reason
# --------------------------------------------------------------------------------------------------
def test_commit_tail_trim_clears_reason():
    st = PickState(tail_trim_dt=100.0, tail_trim_reason="low_pressure")
    picks.commit_tail_trim(st, 50.0)
    assert st.tail_trim_reason == ""

    st.tail_trim_reason = "low_pressure"
    picks.commit_tail_trim(st, None)
    assert st.tail_trim_reason == ""


# --------------------------------------------------------------------------------------------------
# picks.resync_auto_tail_trim -- F1: re-derives an auto ("low_pressure") trim when shut-in moves
# --------------------------------------------------------------------------------------------------
def test_resync_auto_tail_trim_leaves_manual_trim_alone():
    """reason == "" covers both a manual drag and no trim at all -- resync must never touch it,
    since only the analyst's own drag/clear may set or clear a manual pick."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = 42.0
    st.tail_trim_reason = ""

    picks.resync_auto_tail_trim(st, td, res)

    assert st.tail_trim_dt == pytest.approx(42.0)
    assert st.tail_trim_reason == ""


def test_resync_auto_tail_trim_rederives_after_shutin_moves():
    """The motivating bug: tail_trim_dt is shut-in-relative, so nudging shut-in later moves the
    stale cut later in absolute time too, re-admitting whatever crash it was supposed to
    exclude. Resync must re-run seed_tail_trim against the NEW window so the crash stays
    excluded from the diagnostics no matter where shut-in ends up."""
    td, st, res = _seeded_with_crash()
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_reason == "low_pressure"
    stale_dt = st.tail_trim_dt

    st.shutin_idx += 20  # nudge shut-in later, mirroring the CLAUDE.md-cited regression
    res2 = compute_all(st, td)  # computed with the STALE trim still in state -- see docstring

    picks.resync_auto_tail_trim(st, td, res2)

    assert st.tail_trim_reason == "low_pressure"
    assert st.tail_trim_dt != pytest.approx(stale_dt)  # re-derived, not left stale
    res3 = compute_all(st, td)
    assert float(np.nanmin(res3.resampled.p)) > 100.0  # crash still excluded post-resync


def test_resync_auto_tail_trim_clears_when_new_window_has_no_crash():
    td, st, res = _seeded_with_crash(zero_crash_at=None)  # base record, no real crash
    st.tail_trim_dt = 123.0
    st.tail_trim_reason = "low_pressure"

    picks.resync_auto_tail_trim(st, td, res)

    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""


def test_injection_shutin_drag_resyncs_stale_auto_trim():
    """F1 end to end: driving the real ui.DfitApp._attach_controllers Injection-step wiring
    (the duck-typed-stub convention, per test_overview_trim.py's ``_stub``) rather than calling
    picks.resync_auto_tail_trim directly -- this is the path the mutation-testing note in F5
    warns can silently rot if only the headless helper is covered. Dragging the shut-in line
    later must not re-admit the crash the auto trim excluded."""
    td, st, res = _seeded_with_crash()
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_reason == "low_pressure"
    res = compute_all(st, td)

    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS["injection"](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td = td
    stub.res = res
    stub.state = st
    stub.step = "injection"
    stub._controllers = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.refresh = lambda: None
    DfitApp._attach_controllers(stub)

    drag_ctrl = next(c for c in stub._controllers if isinstance(c, picks.DragLineController))
    new_shutin_idx = st.shutin_idx + 20  # nudge later, same regression CLAUDE.md cites
    x_hours = float(td.t_s[new_shutin_idx]) / 3600.0
    drag_ctrl.handlers["shutin"](x_hours)

    assert st.shutin_idx == new_shutin_idx
    res2 = compute_all(st, td)
    assert float(np.nanmin(res2.resampled.p)) > 100.0  # crash stays excluded after the drag


def test_injection_start_drag_does_not_resync_tail_trim():
    """start_idx deliberately does NOT trigger a resync (per F1's design): tail_trim_dt and
    guard_dt both live in dt-from-shut-in space, and resample.resample_pressure_increment takes
    only (dt, p) built from shut-in onward, so a start-only change cannot move either -- a resync
    here would just be wasted work, not a correctness fix."""
    td, st, res = _seeded_with_crash()
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_reason == "low_pressure"
    stale_dt = st.tail_trim_dt
    res = compute_all(st, td)

    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS["injection"](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td = td
    stub.res = res
    stub.state = st
    stub.step = "injection"
    stub._controllers = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.refresh = lambda: None
    DfitApp._attach_controllers(stub)

    drag_ctrl = next(c for c in stub._controllers if isinstance(c, picks.DragLineController))
    new_start_idx = st.start_idx + 5
    x_hours = float(td.t_s[new_start_idx]) / 3600.0
    drag_ctrl.handlers["start"](x_hours)

    assert st.start_idx == new_start_idx
    assert st.tail_trim_dt == pytest.approx(stale_dt)  # untouched by a start-only drag


# --------------------------------------------------------------------------------------------------
# model.compute_all -- the explanatory warning an active trim always emits (never silent)
# --------------------------------------------------------------------------------------------------
def test_compute_all_emits_explanatory_line_for_manual_trim():
    """F7: assert the actual excluded-sample count, not just that some substring is present
    somewhere in the warnings, and assert the line's POSITION (index 0, per insert(0)) rather
    than any(...) -- a regression that appended instead of inserting, or miscounted, would
    otherwise slip past a substring-only/any(...) check."""
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)  # tail_trim_reason left at its default ""
    post = td.t_s >= res.t_shutin_s
    dt_post = td.t_s[post] - res.t_shutin_s
    expected_n_excluded = int(np.sum(dt_post > st.tail_trim_dt))
    res2 = compute_all(st, td)

    assert res2.warnings[0] == (
        f"Tail trimmed {st.tail_trim_dt/60:.0f} min after shut-in "
        f"({expected_n_excluded} raw samples excluded)")


def test_compute_all_emits_explanatory_line_for_low_pressure_trim():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    st.tail_trim_reason = "low_pressure"
    res2 = compute_all(st, td)

    assert any(w.startswith("Tail auto-trimmed") for w in res2.warnings)
