import numpy as np
from dfit_tool import model, interpret
from dfit_tool.model import DerivedResults


def _res(**kw):
    """A DerivedResults with only the fields the net-pressure block reads."""
    return DerivedResults(**kw)


def test_source_compliance_when_compliance_present():
    # Mirror compute_all's net-pressure block by calling the real function under test.
    # Compliance eff ISIP present -> shared ref is compliance, source "compliance".
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=9000.0, effective_isip_tangent=8800.0,
             shmin_compliance=7000.0, shmin_tangent=6900.0, shmin_variable=6950.0))
    assert r.net_pressure_isip_source == "compliance"
    assert r.net_pressure_compliance == interpret.net_pressure(9000.0, 7000.0)
    assert r.net_pressure_tangent == interpret.net_pressure(9000.0, 6900.0)
    assert r.net_pressure_variable == interpret.net_pressure(9000.0, 6950.0)


def test_source_tangent_when_compliance_cleared():
    # C-C style: no contact pick -> no compliance eff ISIP and no shmin_compliance/variable.
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=None, effective_isip_tangent=8800.0,
             shmin_compliance=None, shmin_tangent=6900.0, shmin_variable=None))
    assert r.net_pressure_isip_source == "tangent"
    assert r.net_pressure_compliance is None
    assert r.net_pressure_tangent == interpret.net_pressure(8800.0, 6900.0)
    assert r.net_pressure_variable is None


def test_source_blank_when_both_cleared():
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=None, effective_isip_tangent=None,
             shmin_compliance=None, shmin_tangent=6900.0, shmin_variable=None))
    assert r.net_pressure_isip_source == ""
    assert r.net_pressure_compliance is None
    assert r.net_pressure_tangent is None
    assert r.net_pressure_variable is None
