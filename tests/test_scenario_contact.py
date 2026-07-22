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
