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
