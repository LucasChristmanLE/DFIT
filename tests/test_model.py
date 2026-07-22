"""Unit tests for model.py's pure PickState/DerivedResults logic not already covered elsewhere:
the old-save eff_isip_line -> min_dpdg_G migration in _decode (tests/test_step_status.py covers
step_status persistence specifically)."""

import json

from dfit_tool.model import PickState, _decode


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
