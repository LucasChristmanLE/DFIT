"""Unit tests for the pure commit_* functions: no matplotlib, just PickState mutation. Every
AnchorLineController commit_fn is called as commit_fn(kind, anchor_x, anchor_y, slope) with the
controller's *final* geometry -- see picks.py's module docstring above the commit_* functions."""

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import PickState, TangentPick, compute_all
from tests.helpers import make_testdata, injection_state


def _res():
    td = make_testdata()
    st = injection_state(td)
    return td, st, compute_all(st, td)


# --------------------------------------------------------------------------------------------------
def test_commit_isip_tangent_anchor_snaps_and_refits_ignoring_passed_y_and_slope():
    td, st, res = _res()
    idx = picks._nearest(td.t_s, res.t_shutin_s + 120.0)
    expected_x, expected_y, expected_slope = interpret.tangent_from_index(
        td.t_s, res.bhp_all, idx, half=interpret.ISIP_ANCHOR_HALF)

    state = PickState()
    picks.commit_isip_tangent(state, td, res, "anchor",
                              anchor_x=float(td.t_s[idx]), anchor_y=-99999.0, slope=99999.0)

    assert state.isip_tangent.anchor_x == pytest.approx(expected_x)
    assert state.isip_tangent.anchor_y == pytest.approx(expected_y)
    assert state.isip_tangent.slope == pytest.approx(expected_slope)


def test_commit_isip_tangent_body_translates_anchor_keeps_stored_slope():
    td, st, res = _res()
    state = PickState(isip_tangent=TangentPick(anchor_x=100.0, anchor_y=200.0, slope=5.0))

    picks.commit_isip_tangent(state, td, res, "body", anchor_x=150.0, anchor_y=250.0, slope=999.0)

    assert state.isip_tangent == TangentPick(anchor_x=150.0, anchor_y=250.0, slope=5.0)


def test_commit_isip_tangent_end_sets_slope_keeps_stored_anchor():
    td, st, res = _res()
    state = PickState(isip_tangent=TangentPick(anchor_x=100.0, anchor_y=200.0, slope=5.0))

    picks.commit_isip_tangent(state, td, res, "end", anchor_x=-1.0, anchor_y=-1.0, slope=7.0)

    assert state.isip_tangent == TangentPick(anchor_x=100.0, anchor_y=200.0, slope=7.0)


def test_commit_min_dpdg_point_sets_min_dpdg_g():
    state = PickState()
    picks.commit_min_dpdg_point(state, 5.5)
    assert state.min_dpdg_G == pytest.approx(5.5)


def test_commit_min_dpdg_point_alone_does_not_derive_eff_isip_line():
    """min_dpdg_G is a diagnostic pick only -- it does not feed the effective-ISIP tangent
    (that's contact_G; see
    test_commit_contact_point_then_compute_all_derives_eff_isip_line_compliance)."""
    td, st, res = _res()
    dg = res.diagnostics
    target_G = float(dg.G[dg.G.size // 2])

    picks.commit_min_dpdg_point(st, target_G)
    assert st.contact_G is None
    res2 = compute_all(st, td)

    assert res2.eff_isip_line_compliance is None
    assert res2.effective_isip_compliance is None


def test_commit_closure_line_rederives_closure():
    td, st, res = _res()
    dg = res.diagnostics
    seed_slope, _ = interpret.suggest_closure_tangent(dg.G, dg.GdPdG)
    # Compare with closure_departure_index, not the seed index: this fixture has no hump, and
    # the seed's fallback walk start sits just outside 2%, so the seed reports it anyway.
    seed_idx = interpret.closure_departure_index(dg.G, dg.GdPdG, seed_slope)

    state = PickState(closure_G=42.0)
    picks.commit_closure_line(state, res, "end", anchor_x=0.0, anchor_y=0.0, slope=seed_slope)
    assert state.closure_slope == pytest.approx(seed_slope)
    assert state.closure_G == pytest.approx(float(dg.G[seed_idx]))

    new_slope = 0.8 * seed_slope
    idx = interpret.closure_departure_index(dg.G, dg.GdPdG, new_slope)
    picks.commit_closure_line(state, res, "end", anchor_x=0.0, anchor_y=0.0, slope=new_slope)
    assert state.closure_slope == pytest.approx(new_slope)
    assert state.closure_G == pytest.approx(float(dg.G[idx]))


def test_commit_closure_line_without_diagnostics_keeps_closure():
    td, st, res = _res()
    res.diagnostics = None
    state = PickState(closure_G=42.0)
    picks.commit_closure_line(state, res, "end", anchor_x=0.0, anchor_y=0.0, slope=0.75)
    assert state.closure_slope == pytest.approx(0.75)
    assert state.closure_G == 42.0


def test_commit_contact_point_sets_contact_g():
    state = PickState()
    picks.commit_contact_point(state, 12.5)
    assert state.contact_G == pytest.approx(12.5)


def test_eff_isip_line_anchors_at_min_dpdg_under_c_a():
    """URTeC-2019-123 §2.2 step 5 / §3.1.1 and the ResFrac guide: the P-vs-G line starts at the
    min-dP/dG point, not the contact. contact_G gates the line (no contact, no compliance ISIP)
    but does not position it."""
    td, st, res = _res()
    dg = res.diagnostics
    min_G = float(dg.G[dg.G.size // 3])
    contact_G = float(dg.G[2 * dg.G.size // 3])

    picks.commit_min_dpdg_point(st, min_G)
    picks.commit_contact_point(st, contact_G)
    st.closure_scenario = "C-A clear"  # the triangle anchors only under C-A
    res2 = compute_all(st, td)

    idx = int(np.nanargmin(np.abs(dg.G - min_G)))
    expected_x, expected_y, expected_slope = interpret.tangent_from_index(
        dg.G, res.resampled.p, idx, half=4)
    ln = res2.eff_isip_line_compliance
    assert ln is not None
    assert ln.anchor_x == pytest.approx(expected_x)
    assert ln.anchor_y == pytest.approx(expected_y)
    assert ln.slope == pytest.approx(expected_slope)
    assert res2.effective_isip_compliance == pytest.approx(
        interpret.effective_isip(expected_x, expected_y, expected_slope))


def test_commit_contact_point_then_compute_all_derives_eff_isip_line_compliance():
    """With no min-dP/dG pick, the effective-ISIP tangent falls back to the contact point. Derived
    by compute_all (model.py), not stored -- see DerivedResults.eff_isip_line_compliance."""
    td, st, res = _res()
    dg = res.diagnostics
    target_G = float(dg.G[dg.G.size // 2])

    picks.commit_contact_point(st, target_G)
    res2 = compute_all(st, td)

    idx = int(np.nanargmin(np.abs(dg.G - target_G)))
    expected_x, expected_y, expected_slope = interpret.tangent_from_index(
        dg.G, res.resampled.p, idx, half=4)
    assert res2.eff_isip_line_compliance is not None
    assert res2.eff_isip_line_compliance.anchor_x == pytest.approx(expected_x)
    assert res2.eff_isip_line_compliance.anchor_y == pytest.approx(expected_y)
    assert res2.eff_isip_line_compliance.slope == pytest.approx(expected_slope)
    assert res2.effective_isip_compliance is not None
    assert np.isfinite(res2.effective_isip_compliance)


def test_commit_closure_point_sets_closure_g():
    state = PickState()
    picks.commit_closure_point(state, 7.25)
    assert state.closure_G == pytest.approx(7.25)


def test_commit_tail_trim_sets_seconds():
    state = PickState()
    picks.commit_tail_trim(state, 123.5)
    assert state.tail_trim_dt == pytest.approx(123.5)


def test_commit_tail_trim_none_clears():
    state = PickState(tail_trim_dt=123.5)
    picks.commit_tail_trim(state, None)
    assert state.tail_trim_dt is None
