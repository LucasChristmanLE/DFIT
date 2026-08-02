"""Unit tests for the variable compliance method (compute_all): shmin_variable / eff-ISIP
(tangent) / eff-ISIP (variable), their guards on contact_G/closure_G, and the per-method net
pressures -- see the pattern in
test_commit_functions.py::test_commit_contact_point_then_compute_all_derives_eff_isip_line_compliance."""

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import PickState, TangentPick, compute_all
from tests.helpers import make_testdata, overview_state


def _res():
    td = make_testdata()
    st = overview_state(td)
    return td, st, compute_all(st, td)


def _picked_gs(dg):
    """Two distinct G-times, well clear of the array edges, to anchor contact/closure at."""
    return float(dg.G[dg.G.size // 4]), float(dg.G[3 * dg.G.size // 4])


# --------------------------------------------------------------------------------------------------
def test_variable_compliance_both_picks_set_matches_independent_calc():
    td, st, res = _res()
    dg = res.diagnostics
    contact_G, closure_G = _picked_gs(dg)

    picks.commit_contact_point(st, contact_G)
    picks.commit_closure_point(st, closure_G)
    res2 = compute_all(st, td)

    G_var = (contact_G + closure_G) / 2.0
    expected_shmin_var = float(np.interp(G_var, dg.G, res.resampled.p))
    assert res2.shmin_variable == pytest.approx(expected_shmin_var)

    idx_var = int(np.nanargmin(np.abs(dg.G - G_var)))
    x, y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx_var, half=4)
    expected_eff_var = interpret.effective_isip(x, y, slope)
    assert res2.effective_isip_variable == pytest.approx(expected_eff_var)

    idx_tan = int(np.nanargmin(np.abs(dg.G - closure_G)))
    x2, y2, slope2 = interpret.tangent_from_index(dg.G, res.resampled.p, idx_tan, half=4)
    expected_eff_tan = interpret.effective_isip(x2, y2, slope2)
    assert res2.effective_isip_tangent == pytest.approx(expected_eff_tan)


def test_variable_compliance_guard_only_contact_g_set_leaves_variable_and_tangent_fields_none():
    td, st, res = _res()
    dg = res.diagnostics
    contact_G, _ = _picked_gs(dg)

    picks.commit_contact_point(st, contact_G)
    res2 = compute_all(st, td)

    assert res2.effective_isip_compliance is not None  # unaffected by this guard
    assert res2.effective_isip_tangent is None
    assert res2.shmin_variable is None
    assert res2.effective_isip_variable is None


def test_variable_compliance_guard_only_closure_g_set_sets_tangent_but_not_variable():
    td, st, res = _res()
    dg = res.diagnostics
    _, closure_G = _picked_gs(dg)

    picks.commit_closure_point(st, closure_G)
    res2 = compute_all(st, td)

    assert res2.effective_isip_tangent is not None
    assert res2.shmin_variable is None
    assert res2.effective_isip_variable is None


# --------------------------------------------------------------------------------------------------
# per-method net pressure
# --------------------------------------------------------------------------------------------------
def test_net_pressures_each_use_their_own_effective_isip_and_can_differ():
    td, st, res = _res()
    dg = res.diagnostics
    contact_G, closure_G = _picked_gs(dg)

    picks.commit_contact_point(st, contact_G)
    picks.commit_closure_point(st, closure_G)
    res2 = compute_all(st, td)

    assert res2.net_pressure_compliance == pytest.approx(
        interpret.net_pressure(res2.effective_isip_compliance, res2.shmin_compliance))
    assert res2.net_pressure_tangent == pytest.approx(
        interpret.net_pressure(res2.effective_isip_tangent, res2.shmin_tangent))
    assert res2.net_pressure_variable == pytest.approx(
        interpret.net_pressure(res2.effective_isip_variable, res2.shmin_variable))

    # The three anchors (contact_G, closure_G, and their midpoint) sit on different parts of a
    # non-linear decline curve, so the three method-owned eff ISIPs -- and hence net pressures --
    # are not all equal.
    refs = {res2.effective_isip_compliance, res2.effective_isip_tangent,
            res2.effective_isip_variable}
    assert len(refs) > 1


def test_net_pressure_falls_back_to_literal_isip_when_effective_isip_unavailable(monkeypatch):
    td, st, res = _res()
    dg = res.diagnostics
    contact_G, closure_G = _picked_gs(dg)

    # Literal ISIP needs a shut-in tangent; state.isip_tangent is a stored pick (not derived), so
    # a stand-in TangentPick anchored at t_shutin_s is enough to give literal_isip a value.
    st.isip_tangent = TangentPick(anchor_x=res.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    picks.commit_contact_point(st, contact_G)
    picks.commit_closure_point(st, closure_G)

    # Force every method's effective ISIP to be unavailable without touching shmin_*, which is
    # computed independently of interpret.effective_isip.
    monkeypatch.setattr(interpret, "effective_isip", lambda *a, **kw: None)
    res2 = compute_all(st, td)

    assert res2.effective_isip_compliance is None
    assert res2.effective_isip_tangent is None
    assert res2.effective_isip_variable is None
    assert res2.literal_isip is not None
    assert res2.shmin_compliance is not None
    assert res2.shmin_tangent is not None
    assert res2.shmin_variable is not None

    assert res2.net_pressure_compliance == pytest.approx(
        interpret.net_pressure(res2.literal_isip, res2.shmin_compliance))
    assert res2.net_pressure_tangent == pytest.approx(
        interpret.net_pressure(res2.literal_isip, res2.shmin_tangent))
    assert res2.net_pressure_variable == pytest.approx(
        interpret.net_pressure(res2.literal_isip, res2.shmin_variable))
