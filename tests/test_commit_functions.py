"""Unit tests for the pure commit_* functions: no matplotlib, just PickState mutation. Every
AnchorLineController commit_fn is called as commit_fn(kind, anchor_x, anchor_y, slope) with the
controller's *final* geometry -- see picks.py's module docstring above the commit_* functions."""

import pytest

from dfit_tool import picks
from dfit_tool.model import PickState, TangentPick, compute_all
from tests.helpers import make_testdata, overview_state


def _res():
    td = make_testdata()
    st = overview_state(td)
    return td, st, compute_all(st, td)


# --------------------------------------------------------------------------------------------------
def test_commit_isip_tangent_anchor_snaps_and_refits_ignoring_passed_y_and_slope():
    td, st, res = _res()
    idx = picks._nearest(td.t_s, res.t_shutin_s + 120.0)
    expected_slope = picks._local_slope(td.t_s, res.bhp_all, idx, half=30)

    state = PickState()
    picks.commit_isip_tangent(state, td, res, "anchor",
                              anchor_x=float(td.t_s[idx]), anchor_y=-99999.0, slope=99999.0)

    assert state.isip_tangent.anchor_x == pytest.approx(float(td.t_s[idx]))
    assert state.isip_tangent.anchor_y == pytest.approx(float(res.bhp_all[idx]))
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


def test_commit_eff_isip_line_anchor_snaps_and_refits():
    td, st, res = _res()
    dg, rs = res.diagnostics, res.resampled
    idx = picks._nearest(dg.G, float(dg.G[dg.G.size // 2]))
    expected_slope = picks._local_slope(dg.G, rs.p, idx, half=4)

    state = PickState()
    picks.commit_eff_isip_line(state, res, "anchor",
                               anchor_x=float(dg.G[idx]), anchor_y=-1.0, slope=-1.0)

    assert state.eff_isip_line.anchor_x == pytest.approx(float(dg.G[idx]))
    assert state.eff_isip_line.anchor_y == pytest.approx(float(rs.p[idx]))
    assert state.eff_isip_line.slope == pytest.approx(expected_slope)


def test_commit_eff_isip_line_body_and_end():
    td, st, res = _res()
    state = PickState(eff_isip_line=TangentPick(anchor_x=1.0, anchor_y=2.0, slope=3.0))
    picks.commit_eff_isip_line(state, res, "body", anchor_x=4.0, anchor_y=5.0, slope=999.0)
    assert state.eff_isip_line == TangentPick(anchor_x=4.0, anchor_y=5.0, slope=3.0)

    state2 = PickState(eff_isip_line=TangentPick(anchor_x=1.0, anchor_y=2.0, slope=3.0))
    picks.commit_eff_isip_line(state2, res, "end", anchor_x=-1.0, anchor_y=-1.0, slope=6.0)
    assert state2.eff_isip_line == TangentPick(anchor_x=1.0, anchor_y=2.0, slope=6.0)


def test_commit_closure_line_sets_only_slope():
    td, st, res = _res()
    state = PickState(closure_G=42.0)
    picks.commit_closure_line(state, res, "end", anchor_x=0.0, anchor_y=0.0, slope=0.75)
    assert state.closure_slope == pytest.approx(0.75)
    assert state.closure_G == 42.0  # untouched


def test_commit_contact_point_sets_contact_g():
    state = PickState()
    picks.commit_contact_point(state, 12.5)
    assert state.contact_G == pytest.approx(12.5)


def test_commit_closure_point_sets_closure_g():
    state = PickState()
    picks.commit_closure_point(state, 7.25)
    assert state.closure_G == pytest.approx(7.25)
