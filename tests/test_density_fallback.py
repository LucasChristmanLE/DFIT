"""Fallback density checkbox: sets density to 8.33 ppg and records the fallback in the log."""

from __future__ import annotations

import json
import os
import types

from dfit_tool import store
from dfit_tool.model import FALLBACK_DENSITY_PPG, PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata


class _Var:
    def __init__(self, value=None):
        self.value = value

    def set(self, v):
        self.value = v

    def get(self):
        return self.value


class _Entry:
    def __init__(self):
        self.state = "normal"

    def config(self, **kw):
        self.state = kw.get("state", self.state)


def _log_row(tmp_path, st, td):
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                               st, td, res)


def test_fallback_constant_is_fresh_water():
    assert FALLBACK_DENSITY_PPG == 8.33


def test_log_row_records_fallback(tmp_path):
    td = make_testdata()
    st = injection_state(td)
    st.density_ppg = FALLBACK_DENSITY_PPG
    st.density_fallback = True
    row = _log_row(tmp_path, st, td)
    assert row["density_fallback"] is True
    assert row["fluid_density"] == 8.33


def test_log_row_fallback_false_by_default(tmp_path):
    td = make_testdata()
    st = injection_state(td)
    assert _log_row(tmp_path, st, td)["density_fallback"] is False


def test_log_column_appended_at_tail():
    assert store.LOG_COLUMNS[-1] == "density_fallback"


def test_picks_json_round_trip(tmp_path):
    path = str(tmp_path / "p.json")
    PickState(density_ppg=8.33, density_fallback=True).to_json(path)
    assert PickState.from_json(path).density_fallback is True


def test_old_save_without_key_loads_false(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"density_ppg": 8.6}), encoding="utf-8")
    st = PickState.from_json(str(path))
    assert st.density_fallback is False
    assert st.density_ppg == 8.6


def _toggle_stub(density_text):
    stub = types.SimpleNamespace()
    stub.var_density = _Var(density_text)
    stub.var_density_fallback = _Var(False)
    stub.ent_density = _Entry()
    stub._density_before_fallback = ""
    stub._apply_calls = []
    stub._apply_config = lambda: stub._apply_calls.append(1)
    stub._set_density_fallback_widgets = types.MethodType(
        DfitApp._set_density_fallback_widgets, stub)
    stub._on_density_fallback = types.MethodType(DfitApp._on_density_fallback, stub)
    return stub


def test_check_sets_fallback_and_disables_entry():
    stub = _toggle_stub("8.6")
    stub.var_density_fallback.set(True)
    stub._on_density_fallback()
    assert stub.var_density.get() == "8.33"
    assert stub.ent_density.state == "disabled"
    assert stub._apply_calls == [1]


def test_uncheck_restores_previous_value_and_enables_entry():
    stub = _toggle_stub("8.6")
    stub.var_density_fallback.set(True)
    stub._on_density_fallback()
    stub.var_density_fallback.set(False)
    stub._on_density_fallback()
    assert stub.var_density.get() == "8.6"
    assert stub.ent_density.state == "normal"
    assert stub._apply_calls == [1, 1]


def test_sync_forces_fallback_density():
    stub = types.SimpleNamespace(td=None, state=PickState(density_ppg=9.0))
    for name, val in [("var_pressure", ""), ("var_rate", ""), ("var_volume", ""),
                      ("var_isbhp", False), ("var_pressure_unit", "auto"),
                      ("var_rate_unit", "auto"), ("var_volume_unit", "auto"),
                      ("var_density", "9.0"), ("var_density_fallback", True),
                      ("var_tvd", ""), ("var_well", ""), ("var_formation", ""),
                      ("var_alpha", "1.0"), ("var_step", "30")]:
        setattr(stub, name, _Var(val))
    DfitApp._sync_state_from_widgets(stub)
    assert stub.state.density_fallback is True
    assert stub.state.density_ppg == FALLBACK_DENSITY_PPG
