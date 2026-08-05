"""Unit tests for near-wellbore complexity (CLAUDE.md TODO #5): apparent ISIP minus the
shared reference effective ISIP, and the additive identity

    Shmin + net pressure + complexity = apparent ISIP

which holds for all three methods because every net pressure subtracts its own Shmin from
that same shared reference. Mirrors the direct-_resolve_net_pressures style of
tests/test_net_pressure_reference.py.
"""

import pytest

from dfit_tool import model, picks
from dfit_tool.model import DerivedResults, TangentPick, compute_all
from tests.helpers import make_testdata, overview_state


def _res(**kw):
    """A DerivedResults with only the fields the shared-reference block reads."""
    return DerivedResults(**kw)


def test_complexity_uses_compliance_reference():
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=9000.0,
             effective_isip_tangent=8800.0, shmin_compliance=7000.0))
    assert r.net_pressure_isip_source == "compliance"
    assert r.near_wellbore_complexity == pytest.approx(500.0)


def test_complexity_falls_back_to_tangent_reference():
    # C-C style: no contact pick -> no compliance eff ISIP, so the tangent eff ISIP is the
    # shared reference for both net pressure and complexity.
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=None,
             effective_isip_tangent=8800.0, shmin_tangent=6900.0))
    assert r.net_pressure_isip_source == "tangent"
    assert r.near_wellbore_complexity == pytest.approx(700.0)


def test_complexity_none_when_no_reference_isip():
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=None,
             effective_isip_tangent=None, shmin_tangent=6900.0))
    assert r.net_pressure_isip_source == ""
    assert r.near_wellbore_complexity is None


def test_complexity_none_when_no_apparent_isip():
    # The isip step has not been picked yet. Net pressure is unaffected; only complexity needs
    # the apparent ISIP.
    r = model._resolve_net_pressures(
        _res(apparent_isip=None, effective_isip_compliance=9000.0, shmin_compliance=7000.0))
    assert r.net_pressure_compliance is not None
    assert r.near_wellbore_complexity is None


def test_complexity_negative_reported_as_is_with_no_warning():
    # Apparent ISIP below the P-vs-G extrapolation is physically odd (bad ISIP tangent pick),
    # but the tool reports the arithmetic rather than clamping or warning.
    r = model._resolve_net_pressures(
        _res(apparent_isip=8960.0, effective_isip_compliance=9000.0, shmin_compliance=7000.0))
    assert r.near_wellbore_complexity == pytest.approx(-40.0)
    assert r.warnings == []


def test_cd_complexity_falls_back_to_tangent_reference():
    # C-D clears only the contact pick (apply_closure_scenario), so effective_isip_compliance
    # is None -- but the tangent eff ISIP still builds off the auto-seeded closure_G
    # (seed_tangent), so the shared reference falls back to tangent and complexity IS
    # reported here, even though C-D's own Shmin is shmin_rapid, not shmin_tangent. Pins the
    # real pipeline behavior so the CLAUDE.md/spec correction (docs previously claimed C-D
    # gets no complexity at all) can never silently regress.
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)

    picks.seed_gfunction(st, res)
    st.closure_scenario = "C-D rapid"
    picks.apply_closure_scenario(st, res)
    picks.seed_tangent(st, res)

    # apparent_isip needs a stored shut-in tangent; overview_state sets none. Anchoring at the
    # shut-in instant makes apparent_isip == anchor_y (same stand-in as
    # tests/test_variable_compliance.py::test_net_pressure_none_when_no_effective_isip_available).
    st.isip_tangent = TangentPick(anchor_x=res.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    res = compute_all(st, td)

    assert st.contact_G is None
    assert res.effective_isip_compliance is None
    assert res.effective_isip_tangent is not None
    assert res.net_pressure_isip_source == "tangent"
    assert res.shmin_rapid is not None
    assert res.near_wellbore_complexity == pytest.approx(
        res.apparent_isip - res.effective_isip_tangent)


def test_identity_shmin_plus_net_plus_complexity_equals_apparent_isip():
    # End-to-end through compute_all on synthetic data: the identity must close for each of
    # the three Shmin methods.
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    dg = res.diagnostics

    # Two distinct G-times well clear of the array edges, to anchor contact/closure at.
    contact_G = float(dg.G[dg.G.size // 4])
    closure_G = float(dg.G[3 * dg.G.size // 4])

    # apparent_isip needs a stored shut-in tangent; overview_state sets none. Anchoring at the
    # shut-in instant makes apparent_isip == anchor_y (same stand-in as
    # tests/test_variable_compliance.py::test_net_pressure_none_when_no_effective_isip_available).
    st.isip_tangent = TangentPick(anchor_x=res.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    picks.commit_contact_point(st, contact_G)
    picks.commit_closure_point(st, closure_G)
    r = compute_all(st, td)

    assert r.apparent_isip is not None
    assert r.near_wellbore_complexity is not None
    for shmin, net in ((r.shmin_compliance, r.net_pressure_compliance),
                       (r.shmin_tangent, r.net_pressure_tangent),
                       (r.shmin_variable, r.net_pressure_variable)):
        assert shmin is not None
        assert net is not None
        assert shmin + net + r.near_wellbore_complexity == pytest.approx(r.apparent_isip)
