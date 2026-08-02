"""step_status persistence (model.py) + breadcrumb navigation math + step-aware panel mapping
(ui.py). Tk-dependent parts (packing, button enable/disable, styling) are out of scope here per
the task brief -- only the pure logic is covered.
"""
from __future__ import annotations

import json

import pytest

from dfit_tool.model import PickState, TangentPick, infer_step_status
from dfit_tool.ui import (
    FIELD_STEP,
    PANEL_FIELDS,
    STEPS,
    first_not_visited_step,
    next_step,
    prev_step,
    step_index,
)


# --------------------------------------------------------------------------------------------------
# PickState.step_status persistence
# --------------------------------------------------------------------------------------------------
def test_to_json_from_json_round_trips_step_status(tmp_path):
    state = PickState(pressure_col="P", step_status={"overview": "done", "isip": "visited"})
    path = str(tmp_path / "picks.json")
    state.to_json(path)

    loaded = PickState.from_json(path)

    assert loaded.step_status == {"overview": "done", "isip": "visited"}


def test_from_json_missing_step_status_key_defaults_empty(tmp_path):
    # Hand-written JSON as an old save (pre-step_status) would look: no step_status key at all.
    d = {"pressure_col": "P", "rate_col": None, "alpha": 1.0}
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(d), encoding="utf-8")

    loaded = PickState.from_json(str(path))

    assert loaded.step_status == {}
    assert loaded.pressure_col == "P"


def test_from_json_unknown_key_is_tolerated(tmp_path):
    d = {"pressure_col": "P", "some_future_field": "abc", "step_status": {"overview": "done"}}
    path = tmp_path / "picks.json"
    path.write_text(json.dumps(d), encoding="utf-8")

    loaded = PickState.from_json(str(path))

    assert loaded.pressure_col == "P"
    assert loaded.step_status == {"overview": "done"}


# --------------------------------------------------------------------------------------------------
# infer_step_status
# --------------------------------------------------------------------------------------------------
def test_infer_step_status_empty_state_is_all_not_visited():
    assert infer_step_status(PickState()) == {}


def test_infer_step_status_marks_done_per_picks_present():
    state = PickState(
        start_idx=10, shutin_idx=20,
        isip_tangent=TangentPick(anchor_x=0.0, anchor_y=0.0, slope=1.0),
        min_dpdg_G=5.0,
        closure_G=1.5,
        loglog_window=(1.0, 2.0),
        pp_window=(3.0, 4.0),
    )

    assert infer_step_status(state) == {
        "overview": "done",
        "isip": "done",
        "gfunction": "done",
        "tangent": "done",
        "loglog": "done",
        "porepressure": "done",
    }


def test_infer_step_status_gfunction_from_contact_g_alone():
    state = PickState(contact_G=0.5)
    assert infer_step_status(state) == {"gfunction": "done"}


def test_infer_step_status_overview_from_shutin_idx_alone():
    state = PickState(shutin_idx=20)
    assert infer_step_status(state) == {"overview": "done"}


# --------------------------------------------------------------------------------------------------
# FIELD_STEP / PANEL_FIELDS
# --------------------------------------------------------------------------------------------------
def test_field_step_covers_exactly_the_panel_fields():
    assert set(FIELD_STEP.keys()) == set(PANEL_FIELDS)


def test_field_step_mapping_matches_spec():
    expected = {
        "te (min)": "overview",
        "Vinj (bbl)": "overview",
        "qmax (bpm)": "overview",
        "literal ISIP": "isip",
        "eff ISIP (compliance)": "gfunction",
        "contact P": "gfunction",
        "Shmin compliance": "gfunction",
        "tc compliance (min)": "gfunction",
        "net (compliance)": "gfunction",
        "eff ISIP (tangent)": "tangent",
        "Shmin tangent": "tangent",
        "tc tangent (min)": "tangent",
        "net (tangent)": "tangent",
        "delta closure": "tangent",
        "eff ISIP (variable)": "tangent",
        "Shmin variable": "tangent",
        "tc variable (min)": "tangent",
        "net (variable)": "tangent",
        "pore pressure": "porepressure",
    }
    assert FIELD_STEP == expected


# --------------------------------------------------------------------------------------------------
# step_index / next_step / prev_step
# --------------------------------------------------------------------------------------------------
def test_step_index_matches_steps_order():
    for i, (key, _) in enumerate(STEPS):
        assert step_index(key) == i


def test_next_step_advances_and_clamps_at_last():
    assert next_step("overview") == "isip"
    assert next_step("loglog") == "porepressure"
    assert next_step("porepressure") == "porepressure"  # no-op at last


def test_prev_step_retreats_and_clamps_at_first():
    assert prev_step("isip") == "overview"
    assert prev_step("porepressure") == "loglog"
    assert prev_step("overview") == "overview"  # no-op at first


# --------------------------------------------------------------------------------------------------
# first_not_visited_step (post-_load_picks navigation)
# --------------------------------------------------------------------------------------------------
def test_first_not_visited_step_empty_status_returns_overview():
    assert first_not_visited_step({}) == "overview"


def test_first_not_visited_step_returns_first_gap():
    status = {"overview": "done", "isip": "done", "gfunction": "not_visited"}
    assert first_not_visited_step(status) == "gfunction"


def test_first_not_visited_step_all_covered_falls_back_to_overview():
    status = {k: "done" for k, _ in STEPS}
    assert first_not_visited_step(status) == "overview"
