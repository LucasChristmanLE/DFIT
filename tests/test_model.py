"""Unit tests for model.py's pure PickState/DerivedResults logic not already covered elsewhere:
the old-save eff_isip_line -> min_dpdg_G migration in _decode (tests/test_step_status.py covers
step_status persistence specifically)."""

import json

from dfit_tool.model import PickState, _decode, compute_all

from tests.helpers import overview_state, make_testdata


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
    state = overview_state(td)
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
