"""C-D rapid closure: compute_all sets DerivedResults.shmin_rapid from the apparent ISIP, and
leaves the compliance-method effective ISIP / Shmin / net pressure untouched (decision D2 -- a
separate field so the C-D value can never leak into net_pressure_compliance)."""

from __future__ import annotations

from dfit_tool import interpret, picks
from dfit_tool.model import compute_all
from tests.helpers import make_testdata, overview_state


def _seeded_isip(closure_scenario: str = ""):
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.closure_scenario = closure_scenario
    res = compute_all(st, td)
    assert res.apparent_isip is not None
    return td, st, res


def test_cd_sets_shmin_rapid_from_apparent_isip():
    td, st, res = _seeded_isip("C-D rapid")
    assert res.shmin_rapid is not None
    assert res.shmin_rapid == interpret.shmin_rapid(res.apparent_isip)
    assert res.shmin_rapid == res.apparent_isip - 175.0


def test_cd_leaves_compliance_fields_none():
    """The D2 regression guard: with no contact pick (C-D never sets one -- see
    apply_closure_scenario, which clears contact_G for C-D), effective_isip_compliance,
    shmin_compliance, and net_pressure_compliance all stay None, while shmin_rapid is set and
    does NOT feed net_pressure_compliance (that would be a bogus flat 175 psi net, D2's whole
    point)."""
    td, st, res = _seeded_isip("C-D rapid")
    assert st.contact_G is None
    assert res.effective_isip_compliance is None
    assert res.shmin_compliance is None
    assert res.net_pressure_compliance is None
    assert res.shmin_rapid is not None


def test_non_cd_scenarios_never_set_shmin_rapid():
    for scen in ("", "C-A clear", "C-B adequate", "C-C no-contact"):
        _, _, res = _seeded_isip(scen)
        assert res.shmin_rapid is None
