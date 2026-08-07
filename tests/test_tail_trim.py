"""Manual-only tail trim (CLAUDE.md TODO #3): PickState.tail_trim_dt masks the resampled record
before diagnostics; model.compute_all keeps the untrimmed arrays around (resampled_full/G_full)
so the renderer/controller can always recover from a pathological trim."""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import picks
from dfit_tool.model import PickState, compute_all
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


def test_reset_gfunction_picks_leaves_tail_trim_untouched():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)
    trim = st.tail_trim_dt

    st.closure_scenario = "C-A clear"
    picks.reset_gfunction_picks(st, res2)

    assert st.tail_trim_dt == pytest.approx(trim)
