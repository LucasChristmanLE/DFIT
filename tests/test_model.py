"""Unit tests for model.py's pure PickState/DerivedResults logic not already covered elsewhere:
the old-save eff_isip_line -> min_dpdg_G migration in _decode (tests/test_step_status.py covers
step_status persistence specifically)."""

import json

import numpy as np
import pytest

from dfit_tool import units
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import PickState, TangentPick, _decode, compute_all

from tests.helpers import PRESSURE_COL, injection_state, make_testdata


def test_decode_migrates_old_eff_isip_line_anchor_to_min_dpdg_g(tmp_path):
    d = {
        "pressure_col": "P",
        "eff_isip_line": {"anchor_x": 12.5, "anchor_y": 4300.0, "slope": -20.0},
    }
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(d), encoding="utf-8")

    loaded = PickState.from_json(str(path))

    assert loaded.min_dpdg_G == 12.5
    assert loaded.pressure_col == "P"


def test_decode_does_not_override_an_explicit_min_dpdg_g():
    d = {
        "pressure_col": "P",
        "min_dpdg_G": 7.0,
        "eff_isip_line": {"anchor_x": 12.5, "anchor_y": 4300.0, "slope": -20.0},
    }
    loaded = _decode(d)
    assert loaded.min_dpdg_G == 7.0


def test_decode_without_eff_isip_line_leaves_min_dpdg_g_none():
    d = {"pressure_col": "P"}
    loaded = _decode(d)
    assert loaded.min_dpdg_G is None


def test_decode_coerces_null_scenario_fields_to_empty_string(tmp_path):
    """A foreign/corrupted save can carry an explicit JSON null for a field PickState defaults
    to "" -- compute_all calls state.closure_scenario.startswith(...) unconditionally, which
    raised AttributeError on None before this coercion. _decode must never raise on old or
    foreign JSON (../CLAUDE.md's persistence invariant)."""
    d = {"pressure_col": "P", "closure_scenario": None, "postclosure_scenario": None}
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(d), encoding="utf-8")

    loaded = PickState.from_json(str(path))

    assert loaded.closure_scenario == ""
    assert loaded.postclosure_scenario == ""

    td = make_testdata()
    state = injection_state(td)
    state.closure_scenario = loaded.closure_scenario
    state.postclosure_scenario = loaded.postclosure_scenario
    compute_all(state, td)  # must not raise


def test_well_name_and_formation_round_trip(tmp_path):
    state = PickState(well_name="Foo State 1H", formation="Eagle Ford")
    path = tmp_path / "picks.json"
    state.to_json(str(path))

    loaded = PickState.from_json(str(path))

    assert loaded.well_name == "Foo State 1H"
    assert loaded.formation == "Eagle Ford"


def test_decode_defaults_well_name_and_formation_for_legacy_save():
    # An old save predating these fields lacks the keys entirely -- the known-field filter in
    # _decode must default them to "" (like `notes`), not raise.
    d = {"pressure_col": "P"}
    loaded = _decode(d)
    assert loaded.well_name == ""
    assert loaded.formation == ""


# --------------------------------------------------------------------------------------------------
# unit detection wiring: compute_all refreshes td's unit-detection cache and converts a
# header-suffixed metric pressure column before any downstream value is computed (see
# io_load.refresh_unit_detection / units.py / ../CLAUDE.md's Approach section).
# --------------------------------------------------------------------------------------------------
def _kpa_testdata():
    """The same synthetic record as `helpers.make_testdata`, but with the pressure channel
    renamed to carry a "(KPA)" header suffix and its raw values rescaled so that, once
    converted back by `io_load.detect_channel_unit`, they numerically match the psi fixture."""
    td_psi = make_testdata()
    kpa_col = "PRESSURE (KPA)"
    df = td_psi.df.copy()
    df[kpa_col] = df[PRESSURE_COL] / units.KPA_TO_PSI
    del df[PRESSURE_COL]
    return IoTestData(path="<synthetic>", df=df, datetime_col=td_psi.datetime_col,
                      t_s=td_psi.t_s, columns=list(df.columns)), kpa_col


def test_decode_without_new_unit_fields_defaults_to_auto():
    # An old save predating pressure_unit/rate_unit/volume_unit lacks the keys entirely -- the
    # known-field filter in _decode must default them to "auto", not raise.
    loaded = _decode({"pressure_col": "P"})
    assert loaded.pressure_unit == "auto"
    assert loaded.rate_unit == "auto"
    assert loaded.volume_unit == "auto"


def test_compute_all_converts_header_suffixed_kpa_pressure_to_psi_range():
    td_kpa, kpa_col = _kpa_testdata()
    state = injection_state(td_kpa)
    state.pressure_col = kpa_col

    td_psi = make_testdata()
    state_psi = injection_state(td_psi)

    res_kpa = compute_all(state, td_kpa)
    res_psi = compute_all(state_psi, td_psi)

    # Converted BHP lands in the same field range as the untouched psi fixture -- not the raw
    # ~7x-larger kPa numbers.
    np.testing.assert_allclose(res_kpa.bhp_all, res_psi.bhp_all, rtol=1e-6)

    # Apparent ISIP/Shmin: a stand-in TangentPick anchored at t_shutin_s, same construction as
    # test_variable_compliance.py's, gives apparent_isip a value on both -- they must agree since
    # the conversion happened before this pick was ever interpreted.
    state.isip_tangent = TangentPick(anchor_x=res_kpa.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    state_psi.isip_tangent = TangentPick(anchor_x=res_psi.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    res_kpa = compute_all(state, td_kpa)
    res_psi = compute_all(state_psi, td_psi)

    assert res_kpa.apparent_isip is not None
    assert res_kpa.apparent_isip == pytest.approx(res_psi.apparent_isip, rel=1e-6)


def test_compute_all_kpa_conversion_warns_and_sets_note():
    td_kpa, kpa_col = _kpa_testdata()
    state = injection_state(td_kpa)
    state.pressure_col = kpa_col

    res = compute_all(state, td_kpa)

    assert any("kpa" in w.lower() for w in res.warnings)
    assert "kpa" in res.unit_conversion_note.lower()


def test_compute_all_kpa_conversion_idempotent_across_repeated_calls():
    td_kpa, kpa_col = _kpa_testdata()
    state = injection_state(td_kpa)
    state.pressure_col = kpa_col

    res1 = compute_all(state, td_kpa)
    res2 = compute_all(state, td_kpa)

    np.testing.assert_allclose(res1.bhp_all, res2.bhp_all)
    assert res1.unit_conversion_note == res2.unit_conversion_note


def test_compute_all_field_units_pressure_has_no_conversion_note():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    assert res.unit_conversion_note == ""
