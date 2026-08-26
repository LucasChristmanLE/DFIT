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

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import compute_all
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
    """n=1: no j<i terms at all, isolating the (p_eff[0]-Pres)*sqrt(dt+te/2) offset term."""
    dt_s = np.array([0.0])
    p_eff = np.array([5000.0])
    h = interpret.h_function(dt_s, p_eff, 3000.0, 600.0)
    assert h[0] == pytest.approx((5000.0 - 3000.0) * math.sqrt(0.0 + 600.0 / 2.0))


def test_h_function_te_over_two_offset_shifts_every_term():
    """Doubling te shifts the sqrt(dt + te/2) offset term alone; hand-check against the oracle
    with a different te to pin the te/2 factor specifically (not just te)."""
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
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S)
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    res2 = compute_all(st, td)
    assert res2.shmin_stiffness is not None  # sanity: pick + gate both satisfied

    st.postclosure_scenario = "PC-F no peak"
    res3 = compute_all(st, td)
    assert res3.stiffness_p_eff is None
    assert res3.stiffness_S is None
    assert res3.shmin_stiffness is None


# --------------------------------------------------------------------------------------------------
# model.compute_all: Shmin(stiffness) = picked pressure - 75 psi
# --------------------------------------------------------------------------------------------------
def test_shmin_stiffness_is_pick_minus_75_psi():
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S)
    pick_P = float(res.stiffness_p_eff[1:][idx])
    st.stiffness_pick_P = pick_P

    res2 = compute_all(st, td)

    assert res2.shmin_stiffness == pytest.approx(pick_P - interpret.COMPLIANCE_OFFSET_PSI)
    assert res2.shmin_stiffness == pytest.approx(interpret.shmin_compliance(pick_P))
