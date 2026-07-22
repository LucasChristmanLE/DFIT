"""Scenario-driven contact pick: d2P/dG2 curve, suggestion rules, scenario application,
and the G-function render additions (dotted contact vline, d2 overlay)."""

import numpy as np

from dfit_tool import resample


def test_diagnostics_has_d2pdg2():
    dt = np.linspace(0.0, 3600.0, 200)
    p = 5000.0 - 1500.0 * (1.0 - np.exp(-dt / 900.0))
    rs = resample.resample_pressure_increment(dt, p, step=5.0)
    dg = resample.diagnostics(rs, te=300.0)
    assert dg.d2PdG2.shape == dg.G.shape
    np.testing.assert_allclose(dg.d2PdG2, np.gradient(dg.dPdG, dg.G))


from dfit_tool import interpret


def _s_curve():
    """C-A shape: dP/dG dips to a min at G=5 then rises (parabola)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 50.0 + (G - 5.0) ** 2
    return G, dPdG


def _monotonic_decline():
    """C-C shape: dP/dG only ever falls."""
    G = np.linspace(0.0, 12.0, 241)
    return G, 100.0 * np.exp(-G / 3.0)


def _decline_with_inflection():
    """C-B shape: monotonic decline, steep -> flat (at G=6) -> steep.
    slope(G) = -(1 + (G-6)^2), so d2 = gradient(dPdG, G) has its interior max at G=6."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 6.0) ** 3 / 3.0)
    return G, dPdG


def test_clear_rule_finds_first_10pct_rise():
    G, dPdG = _s_curve()
    min_idx = int(np.argmin(dPdG))
    idx = interpret.suggest_contact_clear_index(dPdG, min_idx)
    assert idx is not None and idx > min_idx
    # threshold: 10% above the min value; the found sample is the FIRST at/above it
    thr = dPdG[min_idx] * 1.10
    assert dPdG[idx] >= thr
    assert np.all(dPdG[min_idx + 1:idx] < thr)
    # analytic crossing: 50*(G-5)^2 rise of 5 -> G = 5 + sqrt(5)
    assert abs(G[idx] - (5.0 + np.sqrt(5.0))) < 0.1


def test_clear_rule_none_on_monotonic_decline():
    G, dPdG = _monotonic_decline()
    min_idx = int(np.argmin(dPdG))  # the last sample
    assert interpret.suggest_contact_clear_index(dPdG, min_idx) is None


def test_inflection_rule_finds_flattening():
    G, dPdG = _decline_with_inflection()
    assert np.all(np.diff(dPdG) < 0)  # sanity: monotone decline
    idx = interpret.suggest_contact_inflection_index(G, dPdG)
    assert idx is not None
    assert abs(G[idx] - 6.0) < 0.2


def test_inflection_rule_none_without_inflection():
    G, dPdG = _monotonic_decline()
    assert interpret.suggest_contact_inflection_index(G, dPdG) is None


def test_inflection_rule_respects_g_min():
    """A flattening before g_min is masked out (early water-hammer region)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 0.5) ** 3 / 3.0)  # flattening at G=0.5 < g_min
    assert interpret.suggest_contact_inflection_index(G, dPdG, g_min=1.0) is None


from dfit_tool import picks
from dfit_tool.model import DerivedResults, PickState
from dfit_tool.resample import Diagnostics


def _res_with(G, dPdG):
    z = np.zeros_like(G)
    dg = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                     t=G, p=z, dp=z, tdpdt=z)
    return DerivedResults(diagnostics=dg)


def test_scenario_ca_sets_contact_right_of_min():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", contact_G=99.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert state.min_dpdg_G is not None and abs(state.min_dpdg_G - 5.0) < 0.2
    assert state.contact_G is not None and state.contact_G > state.min_dpdg_G
    assert abs(state.contact_G - (5.0 + np.sqrt(5.0))) < 0.1


def test_scenario_ca_uses_existing_min_pick():
    """The rule anchors at the user's (possibly dragged) min pick, not a re-detected min."""
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=6.0)
    picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert state.min_dpdg_G == 6.0  # untouched
    # threshold from the value AT the pick: (50 + 1) * 1.1 = 56.1 -> (G-5)^2 >= 6.1
    assert abs(state.contact_G - (5.0 + np.sqrt(6.1))) < 0.1


def test_scenario_ca_hints_when_no_rise():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-A clear", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0  # left unchanged


def test_scenario_cb_sets_contact_at_inflection():
    G, dPdG = _decline_with_inflection()
    state = PickState(closure_scenario="C-B adequate")
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert abs(state.contact_G - 6.0) < 0.2


def test_scenario_cb_hints_when_no_inflection():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-B adequate", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0


def test_scenario_cc_cd_clear_contact():
    G, dPdG = _s_curve()
    for scen in ("C-C no-contact", "C-D rapid"):
        state = PickState(closure_scenario=scen, contact_G=7.0, min_dpdg_G=5.0)
        assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
        assert state.contact_G is None
        assert state.min_dpdg_G == 5.0  # the diagnostic pick survives


def test_scenario_noop_cases():
    G, dPdG = _s_curve()
    # empty scenario
    state = PickState(closure_scenario="", contact_G=7.0)
    assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
    assert state.contact_G == 7.0
    # missing diagnostics
    state = PickState(closure_scenario="C-A clear", contact_G=7.0)
    assert picks.apply_closure_scenario(state, DerivedResults()) is None
    assert state.contact_G == 7.0
