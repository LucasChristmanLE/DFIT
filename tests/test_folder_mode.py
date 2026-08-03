"""Headless coverage for the folder-mode UI shell added in ui.py: source resolution for
_load_test, _save_current_queue_picks persistence, the single-file _load wrapper's exit-from-
folder-mode path, the queue's progress counter, and _apply_loaded_state (factored out of
_load_picks). DfitApp needs a real tk.Tk() root, which this headless (Agg, no display) suite
can't construct -- same duck-typed stand-in approach as test_build_sliders.py/test_step_gate.py:
bind the real DfitApp methods onto a types.SimpleNamespace exposing only what each method
touches.
"""
from __future__ import annotations

import os
import types

from dfit_tool import store
from dfit_tool.model import PickState, infer_step_status
from dfit_tool.ui import STEPS, DfitApp, _resolve_load_source, first_not_visited_step


# --------------------------------------------------------------------------------------------------
# _resolve_load_source: the pure function factored out of _load_test's source resolution.
# --------------------------------------------------------------------------------------------------
def test_resolve_load_source_uses_saved_active_source_when_available():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv", dbs_path="a.dbs")
    saved = PickState(active_source="dbs")
    assert _resolve_load_source(entry, saved) == "DBS"


def test_resolve_load_source_no_saved_picks_defaults_to_first_available():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv", dbs_path="a.dbs")
    assert _resolve_load_source(entry, None) == "CSV"  # CSV first per available_sources order


def test_resolve_load_source_saved_source_not_available_falls_back_to_first():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv")  # no dbs_path
    saved = PickState(active_source="dbs")
    assert _resolve_load_source(entry, saved) == "CSV"


def test_resolve_load_source_dbs_only_entry_defaults_to_dbs():
    entry = store.TestEntry(test_id="w", folder="f", dbs_path="a.dbs")
    assert _resolve_load_source(entry, None) == "DBS"


# --------------------------------------------------------------------------------------------------
# _load_test: when `source` is None (the common case), the source-resolution probe and the
# resume read must not both hit disk -- one store.load_picks_for call, reused for both.
# --------------------------------------------------------------------------------------------------
def test_load_test_reads_picks_json_only_once_when_source_is_none(tmp_path, monkeypatch):
    entry = store.TestEntry(test_id="w1", folder=str(tmp_path), csv_path=str(tmp_path / "w1.csv"))
    store.save_picks_for(entry, PickState(pressure_col="P", active_source="csv"))

    calls = []
    real_load_picks_for = store.load_picks_for

    def counting_load_picks_for(e):
        calls.append(e)
        return real_load_picks_for(e)

    monkeypatch.setattr(store, "load_picks_for", counting_load_picks_for)

    stub = types.SimpleNamespace()
    stub._load_common = lambda path: True
    stub.state = PickState()
    stub._apply_loaded_state_calls = []

    def _apply(state):
        stub._apply_loaded_state_calls.append(state)
        stub.state = state
    stub._apply_loaded_state = _apply
    stub._refresh_queue_row = lambda e: None
    stub.root = types.SimpleNamespace(title=lambda t: None)
    stub._load_test = types.MethodType(DfitApp._load_test, stub)

    stub._load_test(entry)

    assert len(calls) == 1
    assert len(stub._apply_loaded_state_calls) == 1
    assert stub._apply_loaded_state_calls[0].pressure_col == "P"


def test_load_test_reads_picks_json_once_when_source_is_explicit(tmp_path, monkeypatch):
    """An explicit `source=` (Task C's source switching) skips the resolution probe entirely,
    so the one resume read still happens (unless force_reset) -- also exactly once."""
    entry = store.TestEntry(test_id="w1", folder=str(tmp_path), csv_path=str(tmp_path / "w1.csv"),
                            dbs_path=str(tmp_path / "w1.dbs"))
    store.save_picks_for(entry, PickState(pressure_col="P", active_source="csv"))

    calls = []
    real_load_picks_for = store.load_picks_for

    def counting_load_picks_for(e):
        calls.append(e)
        return real_load_picks_for(e)

    monkeypatch.setattr(store, "load_picks_for", counting_load_picks_for)

    stub = types.SimpleNamespace()
    stub._load_common = lambda path: True
    stub.state = PickState()
    stub._apply_loaded_state = lambda state: setattr(stub, "state", state)
    stub._refresh_queue_row = lambda e: None
    stub.root = types.SimpleNamespace(title=lambda t: None)
    stub._load_test = types.MethodType(DfitApp._load_test, stub)

    stub._load_test(entry, source="CSV")

    assert len(calls) == 1


# --------------------------------------------------------------------------------------------------
# _save_current_queue_picks: no-op in single-file mode / before a file is loaded; otherwise
# writes the picks JSON via store and recomputes+refreshes the queue row.
# --------------------------------------------------------------------------------------------------
def _save_stub(current_entry, td, notes="some notes"):
    stub = types.SimpleNamespace()
    stub.current_entry = current_entry
    stub.td = td
    stub.state = PickState(step_status={"overview": "done"})
    stub.txt_notes = types.SimpleNamespace(get=lambda *a, **kw: notes)
    stub._refresh_calls = []
    stub._refresh_queue_row = lambda entry: stub._refresh_calls.append(entry)
    stub._save_current_queue_picks = types.MethodType(DfitApp._save_current_queue_picks, stub)
    return stub


def test_save_current_queue_picks_noop_when_no_current_entry():
    stub = _save_stub(current_entry=None, td=object())
    stub._save_current_queue_picks()
    assert stub._refresh_calls == []


def test_save_current_queue_picks_noop_when_no_file_loaded():
    entry = store.TestEntry(test_id="w", folder="somewhere")
    stub = _save_stub(current_entry=entry, td=None)
    stub._save_current_queue_picks()
    assert stub._refresh_calls == []


def test_save_current_queue_picks_writes_json_and_refreshes_row(tmp_path):
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    stub = _save_stub(current_entry=entry, td=object(), notes="hello")

    stub._save_current_queue_picks()

    assert os.path.exists(entry.picks_path)
    loaded = store.load_picks_for(entry)
    assert loaded is not None
    assert loaded.notes == "hello"
    assert entry.status == store.status_for(loaded)
    assert stub._refresh_calls == [entry]


# --------------------------------------------------------------------------------------------------
# _load wrapper: exits folder mode (saving the outgoing queue test's picks first) and hands off
# to _load_common. Every touch is recorded on the stand-in rather than exercised for real.
# --------------------------------------------------------------------------------------------------
class _FakeTree:
    def __init__(self):
        self.deleted = None

    def get_children(self):
        return ("a", "b")

    def delete(self, *args):
        self.deleted = args


def test_load_wrapper_exits_folder_mode_and_delegates_to_load_common():
    stub = types.SimpleNamespace()
    stub._save_calls = []
    stub._save_current_queue_picks = lambda: stub._save_calls.append(True)
    stub.current_entry = store.TestEntry(test_id="w", folder="f")
    stub.folder_root = "f"
    stub.queue_entries = [stub.current_entry]
    stub._hide_calls = []
    stub._hide_queue = lambda: stub._hide_calls.append(True)
    stub.queue_tree = _FakeTree()
    titles = []
    stub.root = types.SimpleNamespace(title=lambda t: titles.append(t))
    stub._load_common_calls = []
    stub._load_common = lambda path: stub._load_common_calls.append(path) or True
    stub._load = types.MethodType(DfitApp._load, stub)

    stub._load("some/path.csv")

    assert stub._save_calls == [True]
    assert stub.current_entry is None
    assert stub.folder_root is None
    assert stub.queue_entries == []
    assert stub._hide_calls == [True]
    assert stub.queue_tree.deleted == ("a", "b")
    assert titles == ["DFIT interpretation (first build)"]
    assert stub._load_common_calls == ["some/path.csv"]


# --------------------------------------------------------------------------------------------------
# _open_folder_path: current_entry must never be left stale (from a previous folder/test) if the
# auto-opened first entry's _load_test fails (e.g. _load_common's corrupt/unreadable file path,
# which returns early without ever assigning current_entry). The mode invariant is
# "current_entry is None => no test loaded", not "no *possibly wrong* test loaded".
# --------------------------------------------------------------------------------------------------
def test_open_folder_path_failed_auto_open_leaves_current_entry_none(tmp_path):
    (tmp_path / "well1.csv").write_text("a")
    (tmp_path / "well2.csv").write_text("a")

    stub = types.SimpleNamespace()
    stale_entry = store.TestEntry(test_id="stale", folder=os.path.join("other", "folder"))
    stub.current_entry = stale_entry  # a test from a DIFFERENT, previously-open folder
    stub.td = None
    stub.folder_root = os.path.join("other", "folder")
    stub.queue_entries = []
    stub.log_df = None
    stub._save_current_queue_picks = lambda: None
    stub._populate_queue = lambda: None
    stub._show_queue = lambda: None
    stub.warn_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub._load_test_calls = []

    def _failing_load_test(entry, source=None, force_reset=False):
        # Simulates _load_common failing inside the real _load_test -- it returns early and
        # never assigns self.current_entry.
        stub._load_test_calls.append(entry)

    stub._load_test = _failing_load_test
    stub._open_folder_path = types.MethodType(DfitApp._open_folder_path, stub)

    stub._open_folder_path(str(tmp_path))

    assert stub.current_entry is None  # not the stale entry from the previous folder
    assert len(stub._load_test_calls) == 1
    assert stub.folder_root == str(tmp_path)
    assert len(stub.queue_entries) == 2  # the queue is still populated despite the failed load


def test_on_queue_select_tolerates_current_entry_none():
    """After the scenario above (folder open, no test loaded), selecting a row must not crash
    on a None current_entry -- _on_queue_select's guard only compares test_id when it isn't."""
    entry = store.TestEntry(test_id="w1", folder="f")
    stub = types.SimpleNamespace()
    stub.current_entry = None
    stub.queue_entries = [entry]
    stub.queue_tree = types.SimpleNamespace(selection=lambda: ("w1",))
    stub._save_calls = []
    stub._save_current_queue_picks = lambda: stub._save_calls.append(True)
    stub._load_calls = []
    stub._load_test = lambda e: stub._load_calls.append(e)
    stub._on_queue_select = types.MethodType(DfitApp._on_queue_select, stub)

    stub._on_queue_select()

    assert stub._save_calls == [True]
    assert stub._load_calls == [entry]


# --------------------------------------------------------------------------------------------------
# Progress counter: "{n}/{total}" where n counts "done" or "skipped" entries.
# --------------------------------------------------------------------------------------------------
class _FakeLabel:
    def __init__(self):
        self.text = None

    def config(self, **kw):
        if "text" in kw:
            self.text = kw["text"]


def test_update_progress_label_counts_done_and_skipped():
    stub = types.SimpleNamespace()
    stub.queue_entries = [
        store.TestEntry(test_id="a", folder="f", status="done"),
        store.TestEntry(test_id="b", folder="f", status="new"),
    ]
    stub.progress_lbl = _FakeLabel()
    stub._update_progress_label = types.MethodType(DfitApp._update_progress_label, stub)

    stub._update_progress_label()
    assert stub.progress_lbl.text == "1/2"

    stub.queue_entries[1].status = "skipped"
    stub._update_progress_label()
    assert stub.progress_lbl.text == "2/2"


def test_update_progress_label_zero_of_n():
    stub = types.SimpleNamespace()
    stub.queue_entries = [
        store.TestEntry(test_id="a", folder="f", status="new"),
        store.TestEntry(test_id="b", folder="f", status="in_progress"),
    ]
    stub.progress_lbl = _FakeLabel()
    stub._update_progress_label = types.MethodType(DfitApp._update_progress_label, stub)

    stub._update_progress_label()
    assert stub.progress_lbl.text == "0/2"


# --------------------------------------------------------------------------------------------------
# _apply_loaded_state: preserves _load_picks' prior behavior -- infer_step_status backfill for
# an old save with no breadcrumb history, widget reflection, and resume-on-first-not-visited.
# --------------------------------------------------------------------------------------------------
class _Var:
    def __init__(self, value=None):
        self.value = value

    def set(self, v):
        self.value = v

    def get(self):
        return self.value


class _Text:
    def __init__(self):
        self.content = ""

    def delete(self, *a):
        self.content = ""

    def insert(self, idx, text):
        self.content += text


def _apply_stub():
    stub = types.SimpleNamespace()
    for name in ("var_pressure", "var_rate", "var_volume", "var_density", "var_tvd",
                 "var_alpha", "var_step", "var_cscen", "var_pcscen", "var_ppaxis"):
        setattr(stub, name, _Var())
    stub.var_isbhp = _Var()
    stub.var_showd2 = _Var()
    stub.quest_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.txt_notes = _Text()
    stub._views = {"stale": "leftover"}
    stub._goto_calls = []
    stub._goto = lambda step: stub._goto_calls.append(step)
    stub._apply_loaded_state = types.MethodType(DfitApp._apply_loaded_state, stub)
    return stub


def test_apply_loaded_state_infers_step_status_when_missing():
    stub = _apply_stub()
    state = PickState(pressure_col="P", shutin_idx=20, step_status={})

    stub._apply_loaded_state(state)

    assert stub.state is state
    assert stub.state.step_status == infer_step_status(state)
    assert stub._goto_calls == [first_not_visited_step(stub.state.step_status)]


def test_apply_loaded_state_preserves_existing_step_status():
    stub = _apply_stub()
    state = PickState(pressure_col="P", step_status={"overview": "done"})

    stub._apply_loaded_state(state)

    assert stub.state.step_status == {"overview": "done"}
    assert stub._goto_calls == [first_not_visited_step({"overview": "done"})]


def test_apply_loaded_state_reflects_widgets_and_resets_views():
    stub = _apply_stub()
    state = PickState(pressure_col="P", rate_col="R", volume_col=None, pressure_is_bhp=True,
                      density_ppg=8.5, tvd_ft=9000.0, alpha=0.5, resample_step=15.0,
                      closure_scenario="C-A clear", postclosure_scenario="PC-A linear",
                      pp_axis="tm1", show_d2pdg2=True, notes="hi", step_status={"overview": "done"})

    stub._apply_loaded_state(state)

    assert stub.var_pressure.value == "P"
    assert stub.var_rate.value == "R"
    assert stub.var_volume.value == ""
    assert stub.var_isbhp.value is True
    assert stub.var_density.value == "8.5"
    assert stub.var_tvd.value == "9000.0"
    assert stub.var_alpha.value == "0.5"
    assert stub.var_step.value == "15.0"
    assert stub.var_cscen.value == "C-A clear"
    assert stub.var_pcscen.value == "PC-A linear"
    assert stub.var_ppaxis.value == "tm1"
    assert stub.var_showd2.value is True
    assert stub.txt_notes.content == "hi"
    assert stub._views == {k: None for k, _ in STEPS}
