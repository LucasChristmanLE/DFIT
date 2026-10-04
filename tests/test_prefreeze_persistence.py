"""Pre-freeze data-loss and crash fixes (batch A): root-close save, unreadable log, corrupt/foreign
picks JSON, out-of-range saved indices, missing saved source, navigation save failures."""
from __future__ import annotations

import json
import os
import types

import pytest

from dfit_tool import model, picks, store, ui
from dfit_tool.model import PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata
from tests.test_folder_mode import _finish_stub, _skip_test_real_refresh_stub


def _entry(tmp_path, name="w1"):
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    csv = d / f"{name}.csv"
    csv.write_text("t,p\n")
    return store.TestEntry(test_id=name, folder=str(d), csv_path=str(csv))


def _cp1252_log():
    return "test_id,formation\nw9,150°F\n".encode("cp1252")


# ---- Q1 --------------------------------------------------------------------------------------
def _close_stub(folder_mode, save):
    stub = types.SimpleNamespace()
    stub.current_entry = object() if folder_mode else None
    stub.td = object()
    stub.destroyed = []
    stub.root = types.SimpleNamespace(destroy=lambda: stub.destroyed.append(1))
    stub._save_current_queue_picks = save
    stub._on_root_close = types.MethodType(DfitApp._on_root_close, stub)
    return stub


def test_root_close_saves_folder_picks_then_destroys():
    calls = []
    stub = _close_stub(True, lambda: calls.append(1))
    stub._on_root_close()
    assert calls == [1] and stub.destroyed == [1]


def test_root_close_save_failure_asks_and_aborts_on_no(monkeypatch):
    def boom():
        raise OSError("locked")
    asked = []
    monkeypatch.setattr(ui.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
    stub = _close_stub(True, boom)
    stub._on_root_close()
    assert stub.destroyed == [] and "locked" in asked[0][1]


def test_root_close_save_failure_closes_on_yes(monkeypatch):
    def boom():
        raise OSError("locked")
    monkeypatch.setattr(ui.messagebox, "askyesno", lambda *a, **k: True)
    stub = _close_stub(True, boom)
    stub._on_root_close()
    assert stub.destroyed == [1]


def test_root_close_single_file_just_closes():
    calls = []
    stub = _close_stub(False, lambda: calls.append(1))
    stub._on_root_close()
    assert calls == [] and stub.destroyed == [1]


# ---- Q2 --------------------------------------------------------------------------------------
def test_load_log_unreadable_raises_and_does_not_truncate(tmp_path):
    p = tmp_path / store.LOG_FILENAME
    raw = _cp1252_log()
    p.write_bytes(raw)
    with pytest.raises(store.LogReadError) as ei:
        store.load_log(str(tmp_path))
    assert str(p) in str(ei.value.path)
    assert p.read_bytes() == raw


def test_list_tests_survives_unreadable_log(tmp_path):
    (tmp_path / "w1.csv").write_text("a")
    (tmp_path / store.LOG_FILENAME).write_bytes(_cp1252_log())
    entries, df = store.list_tests(str(tmp_path))
    assert len(entries) == 1 and len(df) == 0
    assert df.attrs.get("read_error")


def test_finish_with_unreadable_log_keeps_log_and_saves_picks(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)
    log = tmp_path / store.LOG_FILENAME
    raw = _cp1252_log()
    log.write_bytes(raw)
    errors = []
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    stub._finish()
    assert log.read_bytes() == raw
    assert os.path.exists(entry.picks_path)
    assert errors and "NOT written" in errors[0][1] and store.LOG_FILENAME in errors[0][1]


# ---- Q3 / Q4 ---------------------------------------------------------------------------------
def test_decode_tolerates_extra_keys_in_nested_tangent():
    d = {"isip_tangent": {"anchor_x": 1.0, "anchor_y": 2.0, "slope": -3.0, "future": 9}}
    st = model._decode(d)
    assert st.isip_tangent.slope == -3.0


def test_decode_bad_nested_tangent_becomes_none():
    assert model._decode({"isip_tangent": {"anchor_x": 1.0}}).isip_tangent is None
    assert model._decode({"isip_tangent": [1, 2, 3]}).isip_tangent is None


def test_load_picks_renames_undecodable_file_aside(tmp_path):
    e = _entry(tmp_path)
    open(e.picks_path, "w").write("{not json")
    assert store.load_picks_for(e) is None
    assert not os.path.exists(e.picks_path)
    assert open(e.picks_path + ".corrupt").read() == "{not json"
    open(e.picks_path, "w").write("[1,2")
    assert store.load_picks_for(e) is None
    assert os.path.exists(e.picks_path + ".corrupt.1")


def test_load_picks_missing_file_is_none_and_untouched(tmp_path):
    e = _entry(tmp_path)
    assert store.load_picks_for(e) is None
    assert not os.path.exists(e.picks_path + ".corrupt")


def test_decode_coerces_wrong_container_types():
    st = model._decode({"step_status": None, "mask_intervals": "x", "keep_intervals": 5,
                        "loglog_window": [1], "pp_window": "ab"})
    assert st.step_status == {} and st.mask_intervals == [] and st.keep_intervals == []
    assert st.loglog_window is None and st.pp_window is None
    st = model._decode({"step_status": [1], "loglog_window": [1, 2], "pp_window": 7})
    assert st.step_status == {} and st.loglog_window == (1, 2) and st.pp_window is None


def test_null_step_status_does_not_abort_scan(tmp_path):
    (tmp_path / "w1.csv").write_text("a")
    (tmp_path / "w1.dfit_picks.json").write_text(json.dumps({"step_status": None}))
    entries, _ = store.list_tests(str(tmp_path))
    assert entries[0].status == "new"


def test_list_tests_status_failure_marks_entry_new(tmp_path, monkeypatch):
    (tmp_path / "w1.csv").write_text("a")
    (tmp_path / "w1.dfit_picks.json").write_text("{}")

    def boom(state):
        raise TypeError("x")
    monkeypatch.setattr(store, "status_for", boom)
    entries, _ = store.list_tests(str(tmp_path))
    assert entries[0].status == "new"
    assert (tmp_path / "w1.dfit_picks.json").read_text() == "{}"


# ---- Q5 --------------------------------------------------------------------------------------
@pytest.mark.parametrize("start,shutin", [(-5, 200), (50, 10_000), (50, -1), (10_000, 20_000)])
def test_out_of_range_indices_warned_not_raised(start, shutin):
    """A warning, not a blocker: a blocker pins the UI to Overview, which has no injection
    lines, so the analyst could never re-pick. seed_injection re-seeds them instead."""
    td = make_testdata()
    st = injection_state(td)
    st.start_idx, st.shutin_idx = start, shutin
    res = compute_all(st, td)
    assert model.OUT_OF_RANGE_INJECTION not in res.blockers
    assert any("outside this file's data" in w for w in res.warnings)
    assert res.t_shutin_s is None


@pytest.mark.parametrize("start, shutin", [(10, 10**9), (-5, 100), (10**9, 10**9 + 1)])
def test_seed_injection_reseeds_out_of_range_picks(start, shutin):
    td = make_testdata()
    fresh = injection_state(td)
    fresh.start_idx = fresh.shutin_idx = None
    picks.seed_injection(fresh, td)
    good = (fresh.start_idx, fresh.shutin_idx)
    assert None not in good
    st = injection_state(td)
    st.start_idx, st.shutin_idx = start, shutin
    picks.seed_injection(st, td)
    assert (st.start_idx, st.shutin_idx) == good


def test_in_range_indices_not_blocked():
    td = make_testdata()
    res = compute_all(injection_state(td), td)
    assert not any("outside this file's data" in b for b in res.blockers + res.warnings)


def test_load_picks_failure_shows_error(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{nope")
    monkeypatch.setattr(ui.filedialog, "askopenfilename", lambda **k: str(p))
    errors = []
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    stub = types.SimpleNamespace(_apply_loaded_state=lambda s: pytest.fail("applied"))
    DfitApp._load_picks(stub)
    assert errors


# ---- Q6 --------------------------------------------------------------------------------------
def test_plan_source_resume_saved_available():
    e = store.TestEntry(test_id="a", folder="f", csv_path="a.csv", xlsx_path="a.xlsx")
    st = PickState(active_source="xlsx")
    assert ui._plan_source_load(e, st) == ("XLSX", None)


def test_plan_source_missing_saved_source_starts_fresh():
    e = store.TestEntry(test_id="a", folder="f", csv_path="a.csv")
    st = PickState(active_source="xlsx")
    src, missing = ui._plan_source_load(e, st)
    assert src == "CSV" and missing == "xlsx"


def test_load_test_missing_source_backs_up_and_starts_fresh(tmp_path, monkeypatch):
    e = _entry(tmp_path)
    store.save_picks_for(e, PickState(active_source="xlsx", start_idx=3))
    warns = []
    monkeypatch.setattr(ui.messagebox, "showwarning", lambda *a, **k: warns.append(a))
    applied = []
    stub = types.SimpleNamespace(
        state=PickState(), root=types.SimpleNamespace(title=lambda t: None),
        _load_common=lambda path, well_hint=None: True,
        _apply_loaded_state=lambda s: applied.append(s),
        _refresh_queue_row=lambda e: None, _update_folder_controls=lambda: None)
    DfitApp._load_test(stub, e)
    assert applied == []
    assert os.path.exists(e.picks_path + ".xlsx.bak")
    assert os.path.exists(e.picks_path)
    assert warns and "xlsx" in warns[0][1].lower() and "missing" in warns[0][1]
    assert stub.state.active_source == "csv"


# ---- Q10 -------------------------------------------------------------------------------------
def test_queue_select_save_failure_does_not_navigate(monkeypatch):
    errors = []
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *a, **k: errors.append(a))
    e1 = store.TestEntry(test_id="w1", folder="f")
    e2 = store.TestEntry(test_id="w2", folder="f")

    def boom():
        raise OSError("locked")
    loaded = []
    stub = types.SimpleNamespace(
        queue_tree=types.SimpleNamespace(selection=lambda: ("w2",), selection_set=lambda i: None),
        current_entry=e1, queue_entries=[e1, e2], _save_current_queue_picks=boom,
        _load_test=lambda e: loaded.append(e))
    DfitApp._on_queue_select(stub)
    assert errors and loaded == []


def test_skip_test_save_failure_does_not_advance(tmp_path, monkeypatch):
    stub, entry1, entry2 = _skip_test_real_refresh_stub(tmp_path)
    errors = []
    monkeypatch.setattr(ui.messagebox, "showerror", lambda *a, **k: errors.append(a))

    def boom(entry, state):
        raise OSError("locked")
    monkeypatch.setattr(ui.store, "save_picks_for", boom)
    stub._skip_test()
    assert errors and stub._load_test_calls == []
