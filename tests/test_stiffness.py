"""Relative system stiffness vs effective pressure (URTeC-2019-123 A.8/A.9): the 8th workflow
step, "stiffness". The h-function is a time-convolution leakoff integral (needs a pore-pressure
estimate); relative stiffness S = -dP_eff/dh; the upturn where S rises off its minimum marks
fracture-wall contact and gives a fourth, comparison-only Shmin estimate
(Shmin(stiffness) = picked pressure - 75 psi, interpret.shmin_compliance). Skipped end to end
under PC-F, exactly like pore pressure -- see model.stiffness_skipped.

interpret.h_function is checked against McClure's reference O(n^2) loop, transcribed verbatim
here (his dt in minutes / pressure in MPa cancel out of this relative quantity, so this
transcription keeps psi/seconds, matching h_function's own units -- see its docstring).
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from matplotlib.figure import Figure

from dfit_tool import interpret, model, picks, plots, store, ui
from dfit_tool.model import (
    PickState, compute_all, infer_step_status, porepressure_skipped, stiffness_skipped,
)
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, injection_state


# --------------------------------------------------------------------------------------------------
# shared fixtures
# --------------------------------------------------------------------------------------------------
def _state_with_pore_pressure(closure_scenario: str = "C-A clear"):
    """A state with the injection window, apparent-ISIP, min-dP/dG + contact + closure picks, and
    a pore-pressure estimate all in place -- everything model.compute_all's stiffness block
    gates on. Mirrors test_gradients.py's _full_state / test_pcf_skip.py's _state_with_pp_window."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.closure_scenario = closure_scenario
    res = compute_all(st, td)
    dg = res.diagnostics
    min_G, contact_G = float(dg.G[dg.G.size // 4]), float(dg.G[3 * dg.G.size // 4])
    picks.commit_min_dpdg_point(st, min_G)
    picks.commit_contact_point(st, contact_G)
    closure_G = float(dg.G[dg.G.size // 2])
    picks.commit_closure_point(st, closure_G)
    res = compute_all(st, td)

    picks.seed_pp(st, res)
    st.postclosure_scenario = "PC-A linear"
    st.pp_axis = picks.suggest_pp_axis(st.postclosure_scenario) or st.pp_axis
    res = compute_all(st, td)
    assert res.pore_pressure is not None
    assert res.te_s is not None
    return td, st, res


def _mcclure_h_function_oracle(dt_s, p_eff, pore_pressure_psi, te_s):
    """McClure's reference O(n^2) loop (URTeC-2019-123 A.8), transcribed verbatim except for
    units: his ``dt`` is minutes and pressures are divided by 145.04 (psi -> MPa) before use --
    both cancel out of this relative quantity, so this oracle (like interpret.h_function) works
    directly in psi/seconds instead."""
    n = len(dt_s)
    h = np.zeros(n)
    for i in range(n):
        h[i] = (p_eff[0] - pore_pressure_psi) * math.sqrt(dt_s[i] + te_s / 2.0)
        for j in range(i):
            h[i] += (p_eff[j + 1] - p_eff[j]) * math.sqrt(dt_s[i] - dt_s[j])
    return h


# --------------------------------------------------------------------------------------------------
# interpret.h_function vs the McClure oracle loop
# --------------------------------------------------------------------------------------------------
def test_h_function_matches_mcclure_oracle_loop():
    dt_s = np.array([0.0, 30.0, 90.0, 180.0, 300.0])
    p_eff = np.array([5000.0, 4800.0, 4500.0, 4300.0, 4200.0])
    pore_pressure = 3000.0
    te_s = 600.0

    expected = _mcclure_h_function_oracle(dt_s, p_eff, pore_pressure, te_s)
    got = interpret.h_function(dt_s, p_eff, pore_pressure, te_s)

    np.testing.assert_allclose(got, expected)


def test_h_function_single_point_is_just_the_offset_term():
    """n=1: no j<i terms at all, isolating the (p_eff[0]-Pres)*sqrt(dt+te/2) offset term. The
    hand-checked expected value hard-codes 600/2 (not 600 or some other divisor), so this is
    what actually pins the te/2 factor specifically."""
    dt_s = np.array([0.0])
    p_eff = np.array([5000.0])
    h = interpret.h_function(dt_s, p_eff, 3000.0, 600.0)
    assert h[0] == pytest.approx((5000.0 - 3000.0) * math.sqrt(0.0 + 600.0 / 2.0))


def test_h_function_te_over_two_offset_shifts_every_term():
    """Doubling te shifts the sqrt(dt + te/2) offset term alone; checked against the oracle at
    two different te values to confirm h_function tracks te changes the same way the oracle
    does. The oracle itself already bakes in te/2, so matching it here doesn't independently
    pin that divisor -- see test_h_function_single_point_is_just_the_offset_term for that."""
    dt_s = np.array([0.0, 60.0, 150.0])
    p_eff = np.array([4000.0, 3900.0, 3750.0])
    pore_pressure = 2500.0
    for te_s in (120.0, 240.0):
        expected = _mcclure_h_function_oracle(dt_s, p_eff, pore_pressure, te_s)
        got = interpret.h_function(dt_s, p_eff, pore_pressure, te_s)
        np.testing.assert_allclose(got, expected)


# --------------------------------------------------------------------------------------------------
# interpret.relative_stiffness
# --------------------------------------------------------------------------------------------------
def test_relative_stiffness_matches_mcclure_oracle_sign_and_length():
    dt_s = np.array([0.0, 30.0, 90.0, 180.0, 300.0])
    p_eff = np.array([5000.0, 4800.0, 4500.0, 4300.0, 4200.0])
    h = interpret.h_function(dt_s, p_eff, 3000.0, 600.0)

    S = interpret.relative_stiffness(p_eff, h)

    expected = -np.diff(p_eff) / np.diff(h)
    np.testing.assert_allclose(S, expected)
    assert len(S) == len(p_eff) - 1


def test_relative_stiffness_dh_zero_is_nan_and_raises_no_warnings():
    import warnings

    p_eff = np.array([100.0, 100.0, 90.0])  # dp[0] == 0 too
    h = np.array([1.0, 1.0, 2.0])            # dh[0] == 0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        S = interpret.relative_stiffness(p_eff, h)

    assert math.isnan(S[0])
    assert not math.isnan(S[1])


# --------------------------------------------------------------------------------------------------
# interpret.suggest_stiffness_upturn_index
# --------------------------------------------------------------------------------------------------
def test_suggest_stiffness_upturn_index_finds_the_clear_rise():
    # min at idx 2 (value 8.0); first sample right of it >= 110% (8.8) is idx 3 (9.0).
    S = np.array([50.0, 20.0, 8.0, 9.0, 15.0, 40.0])
    assert interpret.suggest_stiffness_upturn_index(S) == 3


def test_suggest_stiffness_upturn_index_falls_back_to_min_when_no_rise():
    # Monotonic decline -- the min is the last sample, and nothing to its right ever rises.
    S = np.array([50.0, 20.0, 10.0, 9.0, 8.5, 8.2])
    assert interpret.suggest_stiffness_upturn_index(S) == 5


def test_suggest_stiffness_upturn_index_none_when_no_finite_positive_sample():
    S = np.array([np.nan, -1.0, -5.0, 0.0])
    assert interpret.suggest_stiffness_upturn_index(S) is None


def test_suggest_stiffness_upturn_index_ignores_non_finite_and_negative_candidates():
    S = np.array([np.nan, -3.0, 20.0, 8.0, 9.0, 40.0])
    # positive-finite candidates are [20, 8, 9, 40] at indices [2,3,4,5]; min is 8.0 at idx 3.
    assert interpret.suggest_stiffness_upturn_index(S) == 4


def test_suggest_stiffness_upturn_index_masks_out_early_noise_below_g_min():
    # Global min (2.0 at idx 1) sits below G=1.0 (early noise); the masked min among G>=1.0
    # candidates (idx 2..5) is 8.0 at idx 3, which then rises >=10% at idx 4 (9.0 >= 8.8).
    S = np.array([50.0, 2.0, 20.0, 8.0, 9.0, 40.0])
    G = np.array([0.1, 0.5, 1.2, 1.4, 1.6, 1.8])
    assert interpret.suggest_stiffness_upturn_index(S, G) == 4


def test_suggest_stiffness_upturn_index_falls_back_when_g_mask_is_empty():
    # Entire record sits below G=1.0 -- falls back to the unmasked candidate set, same result
    # as the no-G case: min at idx 2 (8.0), rises >=10% at idx 3 (9.0 >= 8.8).
    S = np.array([50.0, 20.0, 8.0, 9.0, 15.0, 40.0])
    G = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert interpret.suggest_stiffness_upturn_index(S, G) == 3


def test_suggest_stiffness_upturn_index_no_g_given_is_unmasked():
    S = np.array([50.0, 2.0, 20.0, 8.0, 9.0, 40.0])
    assert interpret.suggest_stiffness_upturn_index(S) == interpret.suggest_stiffness_upturn_index(
        S, G=None
    )


# --------------------------------------------------------------------------------------------------
# model.compute_all: p_eff construction (line before the min-dP/dG pick, actual pressure after)
# --------------------------------------------------------------------------------------------------
def test_p_eff_uses_tangent_line_before_min_and_actual_pressure_at_and_after():
    td, st, res = _state_with_pore_pressure()
    dg = res.diagnostics
    i_min = int(np.nanargmin(np.abs(dg.G - st.min_dpdg_G)))
    anchor_x, anchor_y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, i_min, half=4)
    expected_line = anchor_y + slope * (dg.G - anchor_x)

    p_eff = res.stiffness_p_eff
    assert p_eff is not None
    np.testing.assert_allclose(p_eff[:i_min], expected_line[:i_min])
    np.testing.assert_allclose(p_eff[i_min:], res.resampled.p[i_min:])


def test_stiffness_S_is_one_shorter_than_p_eff_and_aligned_with_p_eff_1():
    td, st, res = _state_with_pore_pressure()
    assert len(res.stiffness_S) == len(res.stiffness_p_eff) - 1


# --------------------------------------------------------------------------------------------------
# model.compute_all: STIFFNESS_MAX_POINTS cap -- bidirectional (keeps-rises-too) resampling can
# leave far more resampled points than the O(n^2) h_function construction can afford; above the
# cap, compute_all decimates to an even subset (plus index 0/i_min/last) before building
# p_eff/h/S, and the new stiffness_G array carries the matching G-time subset. The fixture's
# default resampled grid is 36 points (see the module docstring's synthetic shape), so patching
# the cap down to 10 is what actually exercises the cap branch here without needing a huge
# synthetic record (that case is covered separately by a scratch timing/memory check, not a
# test -- see the task report).
# --------------------------------------------------------------------------------------------------
def test_stiffness_cap_decimates_arrays_and_warns(monkeypatch):
    monkeypatch.setattr(model, "STIFFNESS_MAX_POINTS", 10)
    td, st, res = _state_with_pore_pressure()
    dg = res.diagnostics
    n = len(dg.G)
    assert n > 10  # sanity: the fixture's full resampled grid exceeds the patched cap

    assert res.stiffness_p_eff is not None
    assert res.stiffness_G is not None
    # <= cap + 2: the evenly spaced subset already includes indices 0 and n-1 (np.linspace's own
    # endpoints), so only the min-dP/dG pick's own index (i_min) can ever add beyond the cap --
    # but np.unique also collapses any of the three that coincide with an evenly spaced point, so
    # this is a loose upper bound, not an exact count.
    assert len(res.stiffness_p_eff) <= 12
    assert len(res.stiffness_G) == len(res.stiffness_p_eff)
    assert len(res.stiffness_S) == len(res.stiffness_p_eff) - 1
    assert np.all(np.diff(res.stiffness_G) > 0)  # strictly increasing -- a real G-time subset

    i_min = int(np.nanargmin(np.abs(dg.G - st.min_dpdg_G)))
    assert dg.G[0] in res.stiffness_G
    assert dg.G[i_min] in res.stiffness_G
    assert dg.G[-1] in res.stiffness_G

    assert any("decimated to limit memory" in w for w in res.warnings)
    expected = f"Stiffness plot uses {len(res.stiffness_p_eff)} of {n} resampled points"
    assert any(expected in w for w in res.warnings)


def test_stiffness_uncapped_case_unchanged():
    """Below the (default 2000) cap, nothing changes: no warning, and stiffness_G is exactly
    the full diagnostics G-time grid (the fixture's 36-point default resampled grid is nowhere
    near the cap)."""
    td, st, res = _state_with_pore_pressure()
    assert not any("decimated to limit memory" in w for w in res.warnings)
    np.testing.assert_array_equal(res.stiffness_G, res.diagnostics.G)
    assert len(res.stiffness_p_eff) == len(res.diagnostics.G)


def test_seed_stiffness_pick_lands_in_capped_arrays(monkeypatch):
    """seed_stiffness must key off stiffness_G (not diagnostics.G) so the seeded pick is always
    one of the (possibly decimated) stiffness_p_eff samples."""
    monkeypatch.setattr(model, "STIFFNESS_MAX_POINTS", 10)
    td, st, res = _state_with_pore_pressure()
    assert st.stiffness_pick_P is None

    picks.seed_stiffness(st, res)

    assert st.stiffness_pick_P is not None
    assert st.stiffness_pick_P in res.stiffness_p_eff[1:]


# --------------------------------------------------------------------------------------------------
# model.compute_all gates: every prerequisite must be present, and >= 4 resampled points.
# --------------------------------------------------------------------------------------------------
def test_stiffness_arrays_none_without_min_dpdg_pick():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    dg = res.diagnostics
    closure_G = float(dg.G[dg.G.size // 2])
    picks.commit_closure_point(st, closure_G)
    picks.seed_pp(st, res)
    st.postclosure_scenario = "PC-A linear"
    res = compute_all(st, td)

    assert st.min_dpdg_G is None
    assert res.pore_pressure is not None  # sanity: everything else is ready
    assert res.stiffness_p_eff is None
    assert res.stiffness_S is None
    assert res.shmin_stiffness is None


def test_stiffness_arrays_none_without_pore_pressure():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    dg = res.diagnostics
    min_G, contact_G = float(dg.G[dg.G.size // 4]), float(dg.G[3 * dg.G.size // 4])
    picks.commit_min_dpdg_point(st, min_G)
    picks.commit_contact_point(st, contact_G)
    res = compute_all(st, td)  # no pp_window set -> no pore pressure

    assert res.pore_pressure is None
    assert res.stiffness_p_eff is None
    assert res.stiffness_S is None


def test_stiffness_arrays_none_under_pcf():
    td, st, res = _state_with_pore_pressure()
    assert res.stiffness_S is not None  # sanity: ready before PC-F

    st.postclosure_scenario = "PC-F no peak"
    res2 = compute_all(st, td)

    assert res2.pore_pressure is None
    assert res2.stiffness_p_eff is None
    assert res2.stiffness_S is None
    assert res2.shmin_stiffness is None


def test_stiffness_arrays_none_with_fewer_than_four_resampled_points():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    st.tail_trim_dt = float(res.resampled_full.dt[2])  # exactly 3 kept points
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    assert len(res.resampled.p) == 3
    dg = res.diagnostics
    picks.commit_min_dpdg_point(st, float(dg.G[0]))
    picks.commit_contact_point(st, float(dg.G[-1]))
    st.pp_window = (float(dg.t[0]), float(dg.t[-1]))
    st.postclosure_scenario = "PC-A linear"
    res = compute_all(st, td)

    assert res.pore_pressure is not None  # sanity: pp fit still works off 2 points
    assert res.stiffness_p_eff is None
    assert res.stiffness_S is None
    assert res.shmin_stiffness is None


def test_stale_stiffness_pick_reports_none_when_gate_fails():
    """A stale pick with the underlying arrays gone (e.g. PC-F chosen after the pick was made)
    must report nothing -- shmin_stiffness is only ever set inside the same gate as the arrays."""
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    res2 = compute_all(st, td)
    assert res2.shmin_stiffness is not None  # sanity: pick + gate both satisfied

    st.postclosure_scenario = "PC-F no peak"
    res3 = compute_all(st, td)
    assert res3.stiffness_p_eff is None
    assert res3.stiffness_S is None
    assert res3.shmin_stiffness is None


# --------------------------------------------------------------------------------------------------
# model.compute_all: stiffness_pick_P joins the stale-pick warning
# --------------------------------------------------------------------------------------------------
def test_stale_stiffness_pick_beyond_trim_warns():
    """stiffness_pick_P is stored in pressure, not G, so it can't be compared against
    diagnostics.G[-1] like contact/min-dP/dG/closure -- it must instead be compared against
    the trimmed record's lowest kept pressure (model.compute_all uses np.nanmin(rs.p) for this,
    since rs.p is no longer guaranteed monotonic once a sustained rise can be kept too -- this
    fixture's own record is a plain decline with no rise anywhere in it, so rs.p[-1] still equals
    that minimum here and is used directly below for simplicity). A trim that raises that low end
    above the pick means the pick no longer sits on the curve."""
    td, st, res = _state_with_pore_pressure()
    dg = res.diagnostics
    rs = res.resampled
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, dg.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])

    # Trim right at the picked sample's own dt -- the retrimmed record's low end (rs.p[-1])
    # then sits above (higher pressure than) the picked value.
    st.tail_trim_dt = float(rs.dt[idx])
    res2 = compute_all(st, td)

    assert res2.resampled.p[-1] > st.stiffness_pick_P  # sanity: pick now off the curve
    assert any("stiffness" in w for w in res2.warnings)
    assert any("beyond the tail trim" in w for w in res2.warnings)


def test_stale_stiffness_pick_absent_when_pick_survives_the_trim():
    td, st, res = _state_with_pore_pressure()
    dg = res.diagnostics
    rs = res.resampled
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, dg.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])

    # Trim well after the picked sample's own dt -- the pick still sits on the retrimmed curve.
    st.tail_trim_dt = float(rs.dt[idx + 5])
    res2 = compute_all(st, td)

    assert res2.resampled.p[-1] <= st.stiffness_pick_P  # sanity: pick still on the curve
    assert not any("stiffness" in w and "beyond the tail trim" in w for w in res2.warnings)


# --------------------------------------------------------------------------------------------------
# model.compute_all: Shmin(stiffness) = picked pressure - 75 psi
# --------------------------------------------------------------------------------------------------
def test_shmin_stiffness_is_pick_minus_75_psi():
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    pick_P = float(res.stiffness_p_eff[1:][idx])
    st.stiffness_pick_P = pick_P

    res2 = compute_all(st, td)

    assert res2.shmin_stiffness == pytest.approx(pick_P - interpret.COMPLIANCE_OFFSET_PSI)
    assert res2.shmin_stiffness == pytest.approx(interpret.shmin_compliance(pick_P))


# --------------------------------------------------------------------------------------------------
# picks.commit_stiffness_point
# --------------------------------------------------------------------------------------------------
def test_commit_stiffness_point_is_a_pure_one_liner():
    from dfit_tool.model import PickState
    st = PickState()
    picks.commit_stiffness_point(st, 4321.5)
    assert st.stiffness_pick_P == pytest.approx(4321.5)


# --------------------------------------------------------------------------------------------------
# picks.seed_stiffness: non-destructive, no-op without arrays, upturn pick from the suggestion
# --------------------------------------------------------------------------------------------------
def test_seed_stiffness_no_op_when_arrays_missing():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)  # no gfunction/pp picks -> res.stiffness_S is None
    assert res.stiffness_S is None

    picks.seed_stiffness(st, res)

    assert st.stiffness_pick_P is None


def test_seed_stiffness_non_destructive():
    td, st, res = _state_with_pore_pressure()
    st.stiffness_pick_P = 1234.5

    picks.seed_stiffness(st, res)

    assert st.stiffness_pick_P == pytest.approx(1234.5)


def test_seed_stiffness_sets_pick_from_the_upturn_suggestion():
    td, st, res = _state_with_pore_pressure()
    assert st.stiffness_pick_P is None

    picks.seed_stiffness(st, res)

    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    expected = float(res.stiffness_p_eff[1:][idx])
    assert st.stiffness_pick_P == pytest.approx(expected)


def test_seed_stiffness_lands_below_the_early_noise_not_near_the_top_of_the_span():
    """S is noisy near G=0; the unmasked global min used to land on early noise near the
    HIGHEST p_eff instead of the actual upturn. The g_min=1.0 mask fixes that -- the chosen
    sample's G must sit at or above 1.0 (the suggest_min_dpdg_index convention)."""
    td, st, res = _state_with_pore_pressure()

    picks.seed_stiffness(st, res)

    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    assert res.diagnostics.G[1:][idx] >= 1.0


def test_seed_stiffness_is_registered_in_seeders():
    assert picks.SEEDERS["stiffness"] is picks.seed_stiffness


# --------------------------------------------------------------------------------------------------
# plots.render_stiffness
# --------------------------------------------------------------------------------------------------
def test_render_stiffness_guard_branch_when_arrays_missing():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    assert res.stiffness_S is None
    fig = Figure()
    ax = fig.add_subplot(111)

    defaults = plots.render_stiffness(ax, td, st, res)

    assert defaults == plots.ViewDefaults()
    assert "requires" in ax.get_title().lower()


def test_render_stiffness_guard_branch_when_all_s_non_positive_no_warnings():
    """An all-non-positive S must not reach ax.set_yscale("log") -- that drew an axis-only log
    plot plus a matplotlib UserWarning ("Data has no positive values...") before this fix."""
    import warnings
    from dfit_tool.model import DerivedResults

    td = make_testdata()
    st = injection_state(td)
    res = DerivedResults()
    res.stiffness_p_eff = np.array([5000.0, 4900.0, 4800.0, 4700.0])
    res.stiffness_S = np.array([-1.0, 0.0, np.nan])
    fig = Figure()
    ax = fig.add_subplot(111)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        defaults = plots.render_stiffness(ax, td, st, res)

    assert defaults == plots.ViewDefaults()
    assert "no positive" in ax.get_title().lower()


def test_render_stiffness_guard_branch_all_non_positive_with_no_upturn_flag_titles_finding():
    """The all-non-positive early return must not silently drop the recorded finding when
    stiffness_no_upturn is also set -- otherwise the degenerate-curve title looks like a data
    problem instead of the analyst's own "no slope change apparent" call."""
    from dfit_tool.model import DerivedResults

    td = make_testdata()
    st = injection_state(td)
    st.stiffness_no_upturn = True
    res = DerivedResults()
    res.stiffness_p_eff = np.array([5000.0, 4900.0, 4800.0, 4700.0])
    res.stiffness_S = np.array([-1.0, 0.0, np.nan])
    fig = Figure()
    ax = fig.add_subplot(111)

    defaults = plots.render_stiffness(ax, td, st, res)

    assert defaults == plots.ViewDefaults()
    title = ax.get_title().lower()
    assert "no positive" in title
    assert "no slope change apparent" in title


def test_render_stiffness_full_branch_has_log_yscale_and_pick_gid():
    td, st, res = _state_with_pore_pressure()
    picks.seed_stiffness(st, res)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)

    defaults = plots.render_stiffness(ax, td, st, res)

    assert ax.get_yscale() == "log"
    assert any(l.get_gid() == "stiffness_pick" for l in ax.get_lines())
    assert defaults.ylim is not None
    assert defaults.ylim[0] > 0.0


def test_render_stiffness_no_pick_line_when_pick_unset():
    td, st, res = _state_with_pore_pressure()
    assert st.stiffness_pick_P is None
    fig = Figure()
    ax = fig.add_subplot(111)

    plots.render_stiffness(ax, td, st, res)

    assert not any(l.get_gid() == "stiffness_pick" for l in ax.get_lines())


def test_stiffness_is_in_renderers_after_porepressure():
    keys = list(plots.RENDERERS.keys())
    assert keys.index("stiffness") == keys.index("porepressure") + 1


# --------------------------------------------------------------------------------------------------
# model.stiffness_skipped mirrors porepressure_skipped
# --------------------------------------------------------------------------------------------------
def test_stiffness_skipped_mirrors_porepressure_skipped_under_pcf():
    st = PickState(postclosure_scenario="PC-F no peak")
    assert stiffness_skipped(st) is True
    assert stiffness_skipped(st) == porepressure_skipped(st)


def test_stiffness_skipped_mirrors_porepressure_skipped_otherwise():
    st = PickState(postclosure_scenario="PC-A linear")
    assert stiffness_skipped(st) is False
    assert stiffness_skipped(st) == porepressure_skipped(st)


# --------------------------------------------------------------------------------------------------
# ui.STEPS / picks.SEEDERS / plots.RENDERERS / store.STEP_KEYS all agree, "stiffness" included.
# --------------------------------------------------------------------------------------------------
def test_stiffness_is_the_last_ui_step():
    assert ui.STEPS[-1][0] == "stiffness"


def test_stiffness_in_all_four_registries():
    assert "stiffness" in picks.SEEDERS
    assert "stiffness" in plots.RENDERERS
    assert "stiffness" in store.STEP_KEYS
    assert "stiffness" in [k for k, _ in ui.STEPS]


# --------------------------------------------------------------------------------------------------
# ui.DfitApp._goto redirects "stiffness" to "loglog" under PC-F, same as "porepressure" --
# duck-typed stand-in pattern from test_pcf_skip.py's _goto_stub, no real tk.Tk().
# --------------------------------------------------------------------------------------------------
def _goto_stub(postclosure_scenario):
    stub = types.SimpleNamespace()
    stub.td = object()
    stub.state = PickState(postclosure_scenario=postclosure_scenario,
                           step_status={k: "visited" for k, _ in ui.STEPS})
    stub.step = "injection"
    stub._seed_step = lambda key: None
    stub._refresh_calls = []
    stub.refresh = lambda: stub._refresh_calls.append(True)
    stub._goto = types.MethodType(DfitApp._goto, stub)
    return stub


def test_goto_stiffness_redirects_to_loglog_under_pcf():
    stub = _goto_stub("PC-F no peak")
    stub._goto("stiffness")
    assert stub.step == "loglog"


def test_goto_stiffness_lands_on_stiffness_under_non_pcf():
    stub = _goto_stub("PC-A linear")
    stub._goto("stiffness")
    assert stub.step == "stiffness"


def test_last_step_is_stiffness_for_non_pcf():
    stub = types.SimpleNamespace()
    stub.state = PickState(postclosure_scenario="PC-A linear")
    stub._last_step = types.MethodType(DfitApp._last_step, stub)
    assert stub._last_step() == "stiffness"


def test_last_step_is_loglog_under_pcf():
    stub = types.SimpleNamespace()
    stub.state = PickState(postclosure_scenario="PC-F no peak")
    stub._last_step = types.MethodType(DfitApp._last_step, stub)
    assert stub._last_step() == "loglog"


# --------------------------------------------------------------------------------------------------
# ui.PANEL_FIELDS / FIELD_STEP: the new "Shmin stiffness" row
# --------------------------------------------------------------------------------------------------
def test_shmin_stiffness_panel_field_maps_to_stiffness_step():
    assert "Shmin stiffness" in ui.PANEL_FIELDS
    assert ui.FIELD_STEP["Shmin stiffness"] == "stiffness"


# --------------------------------------------------------------------------------------------------
# store.status_for: stiffness is accounted for like porepressure, including the PC-F carve-out.
# --------------------------------------------------------------------------------------------------
def test_status_for_pcf_without_stiffness_step_is_done():
    st = PickState(
        postclosure_scenario="PC-F no peak",
        step_status={
            "overview": "done", "injection": "done", "isip": "done", "gfunction": "done",
            "tangent": "done", "loglog": "done",
        },
    )
    assert store.status_for(st) == "done"


def test_status_for_non_pcf_missing_stiffness_step_is_in_progress():
    st = PickState(
        postclosure_scenario="PC-A linear",
        step_status={
            "overview": "done", "injection": "done", "isip": "done", "gfunction": "done",
            "tangent": "done", "loglog": "done", "porepressure": "done",
        },
    )
    assert store.status_for(st) == "in_progress"


def test_status_for_all_eight_steps_done_is_done():
    st = PickState(
        postclosure_scenario="PC-A linear",
        step_status={k: "done" for k in store.STEP_KEYS},
    )
    assert store.status_for(st) == "done"


# --------------------------------------------------------------------------------------------------
# store.LOG_COLUMNS / build_log_row: Shmin_stiffness + Shmin_stiffness_gradient, tail-appended.
# --------------------------------------------------------------------------------------------------
def test_log_columns_has_shmin_stiffness_appended_at_the_tail():
    # "stiffness_no_upturn", "tail_guard_override", and "tangent_uninterpretable" were appended
    # after these two later still, so this checks the -5:-3 slice rather than the very tail.
    assert store.LOG_COLUMNS[-5:-3] == ["Shmin_stiffness", "Shmin_stiffness_gradient"]


def test_build_log_row_round_trips_shmin_stiffness_and_gradient(tmp_path):
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    st.tvd_ft = 10000.0
    res = compute_all(st, td)
    assert res.shmin_stiffness is not None
    assert res.shmin_stiffness_gradient == pytest.approx(res.shmin_stiffness / st.tvd_ft)

    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = str(tmp_path / "well1.csv")
    row = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)

    assert row["Shmin_stiffness"] == pytest.approx(res.shmin_stiffness)
    assert row["Shmin_stiffness_gradient"] == pytest.approx(res.shmin_stiffness_gradient)
    assert list(row.keys()) == store.LOG_COLUMNS


# --------------------------------------------------------------------------------------------------
# ui.DfitApp._update_stepbar force-disables the "porepressure" and "stiffness" breadcrumbs under
# PC-F -- duck-typed stand-in pattern (fake buttons capturing .state()/.configure() calls), no
# real tk.Tk(), same approach as test_folder_mode.py's _FakeButton for .config().
# --------------------------------------------------------------------------------------------------
class _FakeStepButton:
    def __init__(self):
        self.disabled = None
        self.style = None

    def state(self, specs):
        self.disabled = "disabled" in specs

    def configure(self, **kw):
        if "style" in kw:
            self.style = kw["style"]


class _FakeNextButton:
    def __init__(self):
        self.text = None
        self.style = None

    def configure(self, **kw):
        if "text" in kw:
            self.text = kw["text"]
        if "style" in kw:
            self.style = kw["style"]


def _update_stepbar_stub(postclosure_scenario, step_status, step="loglog"):
    stub = types.SimpleNamespace()
    stub.state = PickState(postclosure_scenario=postclosure_scenario, step_status=step_status)
    stub.step = step
    stub.step_buttons = {k: _FakeStepButton() for k, _ in ui.STEPS}
    stub.next_btn = _FakeNextButton()
    stub._update_skip_test_btn = lambda: None
    stub._last_step = types.MethodType(DfitApp._last_step, stub)
    stub._update_stepbar = types.MethodType(DfitApp._update_stepbar, stub)
    return stub


def test_update_stepbar_force_disables_porepressure_and_stiffness_under_pcf():
    visited = {k: "visited" for k, _ in ui.STEPS}  # every breadcrumb otherwise reachable
    stub = _update_stepbar_stub("PC-F no peak", visited)

    stub._update_stepbar()

    assert stub.step_buttons["porepressure"].disabled is True
    assert stub.step_buttons["stiffness"].disabled is True
    # a non-PC-F-affected, already-visited step stays reachable.
    assert stub.step_buttons["loglog"].disabled is False


def test_update_stepbar_leaves_porepressure_and_stiffness_enabled_without_pcf():
    visited = {k: "visited" for k, _ in ui.STEPS}
    stub = _update_stepbar_stub("PC-A linear", visited)

    stub._update_stepbar()

    assert stub.step_buttons["porepressure"].disabled is False
    assert stub.step_buttons["stiffness"].disabled is False


# --------------------------------------------------------------------------------------------------
# "No slope change apparent" (state.stiffness_no_upturn): explicit negative finding -- blanks
# shmin_stiffness while leaving the pick/arrays/curve alone, and the step still finishes "done".
# --------------------------------------------------------------------------------------------------
def test_no_upturn_flag_blanks_shmin_but_keeps_arrays():
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    st.stiffness_no_upturn = True

    res2 = compute_all(st, td)

    assert res2.shmin_stiffness is None
    assert res2.stiffness_p_eff is not None
    assert res2.stiffness_S is not None


def test_no_upturn_flag_suppresses_the_stale_pick_warning():
    """Same setup as test_stale_stiffness_pick_beyond_trim_warns (a pick pushed off the curve
    by a trim) -- with the flag set, the pick reports nothing, so it can't be reported stale."""
    td, st, res = _state_with_pore_pressure()
    dg = res.diagnostics
    rs = res.resampled
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, dg.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    st.tail_trim_dt = float(rs.dt[idx])
    st.stiffness_no_upturn = True

    res2 = compute_all(st, td)

    assert res2.resampled.p[-1] > st.stiffness_pick_P  # sanity: pick still off the curve
    assert not any("stiffness" in w and "beyond the tail trim" in w for w in res2.warnings)


def test_render_stiffness_no_upturn_draws_no_pick_and_titles_the_finding():
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])  # a pick exists...
    st.stiffness_no_upturn = True                              # ...but is suppressed
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)

    plots.render_stiffness(ax, td, st, res)

    assert not any(l.get_gid() == "stiffness_pick" for l in ax.get_lines())
    assert "no slope change apparent" in ax.get_title().lower()
    # the curve itself is still evidence -- one plotted line remains (the stiffness curve).
    assert len(ax.get_lines()) == 1


def test_seed_stiffness_no_op_when_no_upturn_flag_set():
    td, st, res = _state_with_pore_pressure()
    st.stiffness_no_upturn = True
    assert st.stiffness_pick_P is None

    picks.seed_stiffness(st, res)

    assert st.stiffness_pick_P is None


def test_infer_step_status_marks_stiffness_done_on_flag_alone():
    st = PickState(stiffness_no_upturn=True)
    status = infer_step_status(st)
    assert status["stiffness"] == "done"


def test_infer_step_status_stiffness_not_done_without_pick_or_flag():
    st = PickState()
    status = infer_step_status(st)
    assert "stiffness" not in status


def test_no_upturn_flag_round_trips_through_json(tmp_path):
    st = PickState(stiffness_no_upturn=True)
    path = str(tmp_path / "picks.json")
    st.to_json(path)

    loaded = PickState.from_json(path)

    assert loaded.stiffness_no_upturn is True


def test_missing_key_decodes_to_false(tmp_path):
    """A dict without the key (an old save) must decode to the dataclass default, False --
    no migration needed, per the known-field filter in model._decode."""
    from dfit_tool.model import _decode

    d = {"well_name": "w1"}
    st = _decode(d)
    assert st.stiffness_no_upturn is False


def test_log_row_has_stiffness_no_upturn_column_at_the_tail(tmp_path):
    # "tail_guard_override" and "tangent_uninterpretable" were appended after this one -- see
    # tests/test_tail_trim.py's test_log_columns_tail_is_tail_guard_override.
    assert store.LOG_COLUMNS[-3] == "stiffness_no_upturn"

    td, st, res = _state_with_pore_pressure()
    st.stiffness_no_upturn = True
    res = compute_all(st, td)

    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = str(tmp_path / "well1.csv")
    row = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)

    assert row["stiffness_no_upturn"] is True
    assert list(row.keys()) == store.LOG_COLUMNS

    st.stiffness_no_upturn = False
    res = compute_all(st, td)
    row2 = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)
    assert row2["stiffness_no_upturn"] is False


def test_build_log_row_blanks_stiffness_no_upturn_under_pcf(tmp_path):
    """A stiffness_no_upturn flag set before the analyst backs up and switches to PC-F must log
    blank, not the stale True -- the stiffness step is unreachable under PC-F (so the checkbox
    can never be unchecked again), and every other stiffness output already blanks in that
    scenario (see test_stiffness_arrays_none_under_pcf)."""
    td, st, res = _state_with_pore_pressure()
    st.stiffness_no_upturn = True
    st.postclosure_scenario = "PC-F no peak"
    res = compute_all(st, td)
    assert stiffness_skipped(st) is True  # sanity

    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = str(tmp_path / "well1.csv")
    row = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)

    assert row["stiffness_no_upturn"] == ""

    # Flipping back to a non-PC-F scenario revives the (never-cleared) flag in the logged row.
    st.postclosure_scenario = "PC-A linear"
    res = compute_all(st, td)
    row2 = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)
    assert row2["stiffness_no_upturn"] is True


def test_full_walk_with_no_upturn_flag_derives_done_not_skipped():
    """A full workflow walk where every other step has real picks and stiffness instead carries
    stiffness_no_upturn -- store.status_for must derive "done", not "skipped": the per-step Skip
    button is the only thing that marks a step (and thence, via explicit_status, a whole test)
    "skipped"; this flag is a different, legitimate way to finish the step."""
    td, st, res = _state_with_pore_pressure()
    # _state_with_pore_pressure sets pp_window (via seed_pp) but never the log-log span itself --
    # set it directly so infer_step_status also accounts for "loglog".
    st.loglog_window = (float(res.diagnostics.t[0]), float(res.diagnostics.t[-1]))
    st.stiffness_no_upturn = True
    assert st.stiffness_pick_P is None
    st.step_status = infer_step_status(st)
    # infer_step_status deliberately never backfills "overview" (see its docstring) -- add it
    # here the same way a real session would (Overview is always visited first).
    st.step_status["overview"] = "done"

    assert st.step_status["stiffness"] == "done"
    assert store.status_for(st) == "done"


# --------------------------------------------------------------------------------------------------
# ui.DfitApp._on_stiffness_no_upturn: duck-typed stand-in pattern (no real tk.Tk()), mirroring
# test_folder_mode.py's _on_source_change tests.
# --------------------------------------------------------------------------------------------------
class _Var:
    def __init__(self, value=None):
        self.value = value

    def set(self, v):
        self.value = v

    def get(self):
        return self.value


def test_on_stiffness_no_upturn_uncheck_reseeds_when_pick_missing():
    td, st, res = _state_with_pore_pressure()
    st.stiffness_no_upturn = True
    stub = types.SimpleNamespace()
    stub.state = st
    stub.res = res
    stub.var_stiffness_no_upturn = _Var(False)  # simulating the analyst unchecking the box
    stub._refresh_calls = []
    stub.refresh = lambda: stub._refresh_calls.append(True)
    stub._on_stiffness_no_upturn = types.MethodType(DfitApp._on_stiffness_no_upturn, stub)

    stub._on_stiffness_no_upturn()

    assert stub.state.stiffness_no_upturn is False
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    expected = float(res.stiffness_p_eff[1:][idx])
    assert stub.state.stiffness_pick_P == pytest.approx(expected)
    assert stub._refresh_calls == [True]


def test_on_stiffness_no_upturn_uncheck_leaves_existing_pick_alone():
    td, st, res = _state_with_pore_pressure()
    st.stiffness_no_upturn = True
    st.stiffness_pick_P = 1234.5  # pre-existing pick, must survive the uncheck untouched
    stub = types.SimpleNamespace()
    stub.state = st
    stub.res = res
    stub.var_stiffness_no_upturn = _Var(False)
    stub.refresh = lambda: None
    stub._on_stiffness_no_upturn = types.MethodType(DfitApp._on_stiffness_no_upturn, stub)

    stub._on_stiffness_no_upturn()

    assert stub.state.stiffness_pick_P == pytest.approx(1234.5)


def test_on_stiffness_no_upturn_check_sets_flag_without_reseeding():
    td, st, res = _state_with_pore_pressure()
    stub = types.SimpleNamespace()
    stub.state = st
    stub.res = res
    stub.var_stiffness_no_upturn = _Var(True)  # simulating the analyst checking the box
    stub.refresh = lambda: None
    stub._on_stiffness_no_upturn = types.MethodType(DfitApp._on_stiffness_no_upturn, stub)

    stub._on_stiffness_no_upturn()

    assert stub.state.stiffness_no_upturn is True
    assert stub.state.stiffness_pick_P is None


# --------------------------------------------------------------------------------------------------
# ui.DfitApp._update_panel_visibility: duck-typed stand-in pattern (fake pack/pack_forget frames),
# no real tk.Tk(). Covers frm_cscen/frm_pcscen/frm_stiffness visibility across all eight steps and
# the var_stiffness_no_upturn resync on entry to "stiffness".
# --------------------------------------------------------------------------------------------------
class _FakeFrame:
    def __init__(self):
        self.packed = False
        self.pack_calls = []
        self.forget_calls = 0

    def pack(self, **kw):
        self.packed = True
        self.pack_calls.append(kw)

    def pack_forget(self):
        self.packed = False
        self.forget_calls += 1


def _panel_visibility_stub(stiffness_no_upturn=False):
    stub = types.SimpleNamespace()
    stub.frm_cscen = _FakeFrame()
    stub.frm_pcscen = _FakeFrame()
    stub.frm_stiffness = _FakeFrame()
    stub.frm_tangent = _FakeFrame()
    stub.sep_before_notes = object()
    stub.var_stiffness_no_upturn = _Var()
    stub.var_tangent_uninterpretable = _Var()
    stub.state = PickState(stiffness_no_upturn=stiffness_no_upturn)
    stub._update_ppaxis_enabled = lambda: None
    stub._update_panel_visibility = types.MethodType(DfitApp._update_panel_visibility, stub)
    return stub


def test_update_panel_visibility_frm_stiffness_only_on_stiffness_step():
    stub = _panel_visibility_stub()
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_stiffness.packed == (key == "stiffness")


def test_update_panel_visibility_frm_cscen_only_on_gfunction():
    stub = _panel_visibility_stub()
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_cscen.packed == (key == "gfunction")


def test_update_panel_visibility_frm_pcscen_only_on_loglog_and_porepressure():
    stub = _panel_visibility_stub()
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_pcscen.packed == (key in ("loglog", "porepressure"))


def test_update_panel_visibility_resyncs_var_from_state_on_stiffness_entry():
    stub = _panel_visibility_stub(stiffness_no_upturn=True)
    stub.step = "stiffness"

    stub._update_panel_visibility()

    assert stub.var_stiffness_no_upturn.get() is True


def test_update_panel_visibility_does_not_touch_var_off_the_stiffness_step():
    stub = _panel_visibility_stub(stiffness_no_upturn=True)
    stub.var_stiffness_no_upturn.set(False)  # simulate a stale value from a previous step
    stub.step = "gfunction"

    stub._update_panel_visibility()

    assert stub.var_stiffness_no_upturn.get() is False
