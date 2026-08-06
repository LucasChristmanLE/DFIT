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

import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import store, ui
from dfit_tool.model import PickState, compute_all, infer_step_status
from dfit_tool.ui import (STEPS, DfitApp, _next_new_index, _resolve_load_source,
                          first_not_visited_step)
from tests.helpers import make_testdata, overview_state


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
    stub._update_folder_controls = lambda: None
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
    stub._update_folder_controls = lambda: None
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
    # Widget stand-ins for _sync_state_from_widgets, defaulted to match PickState's own
    # defaults so existing assertions (which only look at notes/status) are unaffected.
    stub.var_pressure = _Var("")
    stub.var_rate = _Var(None)
    stub.var_volume = _Var(None)
    stub.var_isbhp = _Var(False)
    stub.var_density = _Var(None)
    stub.var_tvd = _Var(None)
    stub.var_well = _Var("")
    stub.var_formation = _Var("")
    stub.var_alpha = _Var(1.0)
    stub.var_step = _Var(30.0)
    stub._refresh_calls = []
    stub._refresh_queue_row = lambda entry: stub._refresh_calls.append(entry)
    stub._sync_state_from_widgets = types.MethodType(DfitApp._sync_state_from_widgets, stub)
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


def test_save_current_queue_picks_captures_unapplied_widget_edits(tmp_path):
    """An entry-widget edit that was never Applied must still be captured on save (the bug this
    task fixes) -- density/well/formation are typed as NEW values while self.state still holds
    the OLD ones; _save_current_queue_picks must sync the widgets into state before saving."""
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    stub = _save_stub(current_entry=entry, td=object(), notes="hello")
    stub.state.density_ppg = 8.0
    stub.state.well_name = "Old Well"
    stub.state.formation = "Old Formation"
    stub.var_density.set(9.3)
    stub.var_well.set("New Well 1H")
    stub.var_formation.set("Eagle Ford")

    stub._save_current_queue_picks()

    loaded = store.load_picks_for(entry)
    assert loaded is not None
    assert loaded.density_ppg == 9.3
    assert loaded.well_name == "New Well 1H"
    assert loaded.formation == "Eagle Ford"


def test_sync_state_from_widgets_garbled_density_preserves_prior_value(tmp_path):
    """This can fire mid-edit (e.g. on autosave while the analyst is still typing), so a
    non-empty but unparseable Density/TVD entry ("8." or "8.x") must not null out a
    previously-good value -- only an explicitly emptied box should clear it."""
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    stub = _save_stub(current_entry=entry, td=object(), notes="hello")
    stub.state.density_ppg = 8.0
    stub.state.tvd_ft = 9000.0
    stub.var_density.set("8.")  # garbled, mid-keystroke
    stub.var_tvd.set("9000.x")  # garbled

    stub._sync_state_from_widgets()

    assert stub.state.density_ppg == 8.0
    assert stub.state.tvd_ft == 9000.0

    # An explicitly emptied box is a real, intentional clear -- that still zeroes it out.
    stub.var_density.set("")
    stub.var_tvd.set("")

    stub._sync_state_from_widgets()

    assert stub.state.density_ppg is None
    assert stub.state.tvd_ft is None


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
    stub._update_folder_controls_calls = []
    stub._update_folder_controls = lambda: stub._update_folder_controls_calls.append(True)
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
    assert stub._update_folder_controls_calls == [True]


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
    stub._update_folder_controls_calls = []
    stub._update_folder_controls = lambda: stub._update_folder_controls_calls.append(True)
    # The real _make_scan_progress needs a live tk.Tk() root -- stub it out with a no-op
    # progress window/setter, same duck-typing approach as the rest of this stand-in.
    stub._make_scan_progress = lambda: (
        types.SimpleNamespace(grab_release=lambda: None, destroy=lambda: None),
        lambda text: None,
    )

    def _failing_load_test(entry, source=None, force_reset=False):
        # Simulates _load_common failing inside the real _load_test -- it returns early and
        # never assigns self.current_entry.
        stub._load_test_calls.append(entry)

    stub._load_test = _failing_load_test
    stub._open_folder_path = types.MethodType(DfitApp._open_folder_path, stub)

    stub._open_folder_path(str(tmp_path))

    assert stub.current_entry is None  # not the stale entry from the previous folder
    assert len(stub._load_test_calls) == 1
    # _open_folder_path calls _update_folder_controls itself -- _load_test's early return
    # (simulated above) never reaches its own call, so this is the only thing that resyncs the
    # Source combobox and Skip-test button to the now-None current_entry.
    assert stub._update_folder_controls_calls == [True]
    assert stub.folder_root == str(tmp_path)
    assert len(stub.queue_entries) == 2  # the queue is still populated despite the failed load


# --------------------------------------------------------------------------------------------------
# _open_folder_path: empty folder. Regression guard -- the "No DFIT tests found" messagebox must
# still fire after the progress modal is torn down; an earlier version buried the empty-entries
# check inside the try, so its `return` ran the `finally` and exited before ever reaching it.
# --------------------------------------------------------------------------------------------------
def test_open_folder_path_no_tests_found_shows_messagebox(tmp_path, monkeypatch):
    info_calls = []
    monkeypatch.setattr(ui.messagebox, "showinfo", lambda *a, **kw: info_calls.append((a, kw)))
    monkeypatch.setattr(
        store, "list_tests",
        lambda path, progress=None: ([], pd.DataFrame(columns=store.LOG_COLUMNS)),
    )

    stub = types.SimpleNamespace()
    stub.current_entry = None
    stub.td = None
    stub._make_scan_progress = lambda: (
        types.SimpleNamespace(grab_release=lambda: None, destroy=lambda: None),
        lambda text: None,
    )
    stub._open_folder_path = types.MethodType(DfitApp._open_folder_path, stub)

    stub._open_folder_path(str(tmp_path))

    assert len(info_calls) == 1


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
                 "var_well", "var_formation",
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
                      density_ppg=8.5, tvd_ft=9000.0, well_name="Foo State 1H",
                      formation="Eagle Ford", alpha=0.5, resample_step=15.0,
                      closure_scenario="C-A clear", postclosure_scenario="PC-A linear",
                      pp_axis="tm1", show_d2pdg2=True, notes="hi", step_status={"overview": "done"})

    stub._apply_loaded_state(state)

    assert stub.var_pressure.value == "P"
    assert stub.var_rate.value == "R"
    assert stub.var_volume.value == ""
    assert stub.var_isbhp.value is True
    assert stub.var_density.value == "8.5"
    assert stub.var_tvd.value == "9000.0"
    assert stub.var_well.value == "Foo State 1H"
    assert stub.var_formation.value == "Eagle Ford"
    assert stub.var_alpha.value == "0.5"
    assert stub.var_step.value == "15.0"
    assert stub.var_cscen.value == "C-A clear"
    assert stub.var_pcscen.value == "PC-A linear"
    assert stub.var_ppaxis.value == "tm1"
    assert stub.var_showd2.value is True
    assert stub.txt_notes.content == "hi"


# --------------------------------------------------------------------------------------------------
# _next_new_index: the pure selection logic behind Finish's and Skip test's auto-advance.
# --------------------------------------------------------------------------------------------------
def test_next_new_index_finds_next_after_current():
    assert _next_new_index(["done", "new", "in_progress"], 0) == 1


def test_next_new_index_wraps_circularly():
    assert _next_new_index(["new", "done", "new"], 2) == 0


def test_next_new_index_skips_done_in_progress_and_skipped():
    assert _next_new_index(["done", "in_progress", "skipped", "new"], 0) == 3


def test_next_new_index_none_when_none_remain():
    assert _next_new_index(["done", "skipped", "in_progress"], 0) is None


def test_next_new_index_none_when_current_is_only_new():
    # index 1 is itself "new", but it's the current entry (its work was just saved) -- it must
    # never be reported back as its own "next".
    assert _next_new_index(["done", "new", "skipped"], 1) is None


def test_next_new_index_empty_statuses_is_none():
    assert _next_new_index([], 0) is None


# --------------------------------------------------------------------------------------------------
# _update_folder_controls: the one sync point for the Source combobox and the Skip-test button.
# --------------------------------------------------------------------------------------------------
class _FakeCombo:
    def __init__(self):
        self.values = None
        self.state_ = None

    def __setitem__(self, key, value):
        assert key == "values"
        self.values = value

    def config(self, **kw):
        if "state" in kw:
            self.state_ = kw["state"]


class _FakeButton:
    def __init__(self):
        self.state_ = None
        self.text = None

    def config(self, **kw):
        if "state" in kw:
            self.state_ = kw["state"]
        if "text" in kw:
            self.text = kw["text"]


def _folder_controls_stub():
    stub = types.SimpleNamespace()
    stub.var_source = _Var()
    stub.cmb_source = _FakeCombo()
    stub.btn_skip_test = _FakeButton()
    stub._update_skip_test_btn = types.MethodType(DfitApp._update_skip_test_btn, stub)
    stub._update_folder_controls = types.MethodType(DfitApp._update_folder_controls, stub)
    return stub


def test_update_folder_controls_single_file_mode_disables_and_clears():
    stub = _folder_controls_stub()
    stub.current_entry = None

    stub._update_folder_controls()

    assert stub.var_source.value == ""
    assert stub.cmb_source.values == []
    assert stub.cmb_source.state_ == "disabled"
    assert stub.btn_skip_test.state_ == "disabled"
    assert stub.btn_skip_test.text == "Skip test"


def test_update_folder_controls_folder_mode_multi_source_enables_readonly():
    stub = _folder_controls_stub()
    entry = store.TestEntry(test_id="w1", folder="f", csv_path="a.csv", dbs_path="a.dbs")
    stub.current_entry = entry
    stub.state = PickState(active_source="dbs", explicit_status="skipped")

    stub._update_folder_controls()

    assert stub.cmb_source.values == ["CSV", "DBS"]
    assert stub.var_source.value == "DBS"
    assert stub.cmb_source.state_ == "readonly"
    assert stub.btn_skip_test.state_ == "normal"
    assert stub.btn_skip_test.text == "Unskip test"


def test_update_folder_controls_folder_mode_single_source_disables_source_combo():
    stub = _folder_controls_stub()
    entry = store.TestEntry(test_id="w1", folder="f", csv_path="a.csv")
    stub.current_entry = entry
    stub.state = PickState(active_source="csv", explicit_status=None)

    stub._update_folder_controls()

    assert stub.cmb_source.values == ["CSV"]
    assert stub.cmb_source.state_ == "disabled"
    assert stub.btn_skip_test.state_ == "normal"
    assert stub.btn_skip_test.text == "Skip test"


# --------------------------------------------------------------------------------------------------
# _on_source_change: switching sources resets picks (confirm first); decline reverts the
# combobox, accept delegates to _load_test(source=..., force_reset=True).
# --------------------------------------------------------------------------------------------------
def test_on_source_change_noop_when_same_source_selected():
    stub = types.SimpleNamespace()
    stub.state = PickState(active_source="csv")
    stub.var_source = _Var("CSV")
    stub._load_test_calls = []
    stub._load_test = lambda *a, **kw: stub._load_test_calls.append((a, kw))
    stub._on_source_change = types.MethodType(DfitApp._on_source_change, stub)

    stub._on_source_change()

    assert stub._load_test_calls == []


def test_on_source_change_decline_reverts_combobox(monkeypatch):
    monkeypatch.setattr(ui.messagebox, "askyesno", lambda *a, **kw: False)
    stub = types.SimpleNamespace()
    stub.state = PickState(active_source="csv")
    stub.current_entry = store.TestEntry(test_id="w1", folder="f")
    stub.var_source = _Var("DBS")
    stub._load_test_calls = []
    stub._load_test = lambda *a, **kw: stub._load_test_calls.append((a, kw))
    stub._on_source_change = types.MethodType(DfitApp._on_source_change, stub)

    stub._on_source_change()

    assert stub.var_source.value == "CSV"
    assert stub._load_test_calls == []


def test_on_source_change_accept_calls_load_test_with_force_reset(monkeypatch):
    monkeypatch.setattr(ui.messagebox, "askyesno", lambda *a, **kw: True)
    stub = types.SimpleNamespace()
    stub.state = PickState(active_source="csv")
    entry = store.TestEntry(test_id="w1", folder="f")
    stub.current_entry = entry
    stub.var_source = _Var("DBS")
    stub._load_test_calls = []
    stub._load_test = lambda *a, **kw: stub._load_test_calls.append((a, kw))
    stub._on_source_change = types.MethodType(DfitApp._on_source_change, stub)

    stub._on_source_change()

    assert stub._load_test_calls == [((entry,), {"source": "DBS", "force_reset": True})]


# --------------------------------------------------------------------------------------------------
# _advance_queue: the shared auto-advance tail of Finish and Skip test. Advances to the next
# "new" entry (scanning circularly from just after the current one), or reports the queue is
# exhausted.
# --------------------------------------------------------------------------------------------------
def test_advance_queue_loads_next_new_entry():
    entry1 = store.TestEntry(test_id="w1", folder="f1", status="done")
    entry2 = store.TestEntry(test_id="w2", folder="f2", status="new")
    stub = types.SimpleNamespace()
    stub.current_entry = entry1
    stub.queue_entries = [entry1, entry2]
    stub._load_test_calls = []
    stub._load_test = lambda e: stub._load_test_calls.append(e)
    stub._advance_queue = types.MethodType(DfitApp._advance_queue, stub)

    stub._advance_queue()

    assert stub._load_test_calls == [entry2]


def test_advance_queue_reports_no_new_tests_remain(monkeypatch):
    info_calls = []
    monkeypatch.setattr(ui.messagebox, "showinfo", lambda *a, **kw: info_calls.append((a, kw)))
    entry1 = store.TestEntry(test_id="w1", folder="f1", status="done")
    stub = types.SimpleNamespace()
    stub.current_entry = entry1
    stub.queue_entries = [entry1]
    stub._load_test_calls = []
    stub._load_test = lambda e: stub._load_test_calls.append(e)
    stub._advance_queue = types.MethodType(DfitApp._advance_queue, stub)

    stub._advance_queue()

    assert stub._load_test_calls == []
    assert len(info_calls) == 1


# --------------------------------------------------------------------------------------------------
# _finish / _skip_test: the folder branch saves picks only via store.save_picks_for (no
# <stem>_picks.json duplicate), upserts dfit_log.csv, and (unlike single-file mode) advances the
# queue via _advance_queue. The second queue entry (`entry2`) lets each test choose, via
# `second_status`, whether an advance target exists.
# --------------------------------------------------------------------------------------------------
def _finish_stub(tmp_path, folder_mode, monkeypatch, second_status="new"):
    png_calls = []

    def _fake_save_pngs(*a, **kw):
        png_calls.append(1)
        return []
    monkeypatch.setattr(ui.plots, "save_all_step_pngs", _fake_save_pngs)
    data_dir = tmp_path / "w1" if folder_mode else tmp_path
    if folder_mode:
        data_dir.mkdir()
    csv_path = data_dir / "w1.csv"
    csv_path.write_text("t,p\n")

    td = make_testdata()
    td.path = str(csv_path)
    state = overview_state(td)
    res = compute_all(state, td)

    stub = types.SimpleNamespace()
    stub.td = td
    stub.state = state
    stub.res = res
    stub.step = "porepressure"
    stub._views = {}
    stub._png_calls = png_calls
    stub.txt_notes = types.SimpleNamespace(get=lambda *a, **kw: "")
    stub.refresh = lambda: None
    stub._refresh_calls = []
    stub._refresh_queue_row = lambda e: stub._refresh_calls.append(e)
    stub._update_skip_test_btn = lambda: None
    # Widget stand-ins for _sync_state_from_widgets, defaulted to match the state built above
    # (real _finish/_skip_test both sync these before refresh()).
    stub.var_pressure = _Var(state.pressure_col)
    stub.var_rate = _Var(state.rate_col)
    stub.var_volume = _Var(state.volume_col)
    stub.var_isbhp = _Var(state.pressure_is_bhp)
    stub.var_density = _Var(state.density_ppg)
    stub.var_tvd = _Var(state.tvd_ft)
    stub.var_well = _Var(state.well_name)
    stub.var_formation = _Var(state.formation)
    stub.var_alpha = _Var(state.alpha)
    stub.var_step = _Var(state.resample_step)
    stub._sync_state_from_widgets = types.MethodType(DfitApp._sync_state_from_widgets, stub)

    entry2 = None
    if folder_mode:
        entry = store.TestEntry(test_id="w1", folder=str(data_dir), csv_path=str(csv_path))
        stub.current_entry = entry
        stub.folder_root = str(tmp_path)
        stub.log_df = store.load_log(str(tmp_path))
        entry2_dir = tmp_path / "w2"
        entry2_dir.mkdir()
        csv2 = entry2_dir / "w2.csv"
        csv2.write_text("t,p\n")
        entry2 = store.TestEntry(test_id="w2", folder=str(entry2_dir), csv_path=str(csv2),
                                 status=second_status)
        stub.queue_entries = [entry, entry2]
        stub._load_test_calls = []
        stub._load_test = lambda e: stub._load_test_calls.append(e)
    else:
        entry = None
        stub.current_entry = None
        stub.folder_root = None
        stub.log_df = None

    stub._write_log_row = types.MethodType(DfitApp._write_log_row, stub)
    stub._advance_queue = types.MethodType(DfitApp._advance_queue, stub)
    stub._finish = types.MethodType(DfitApp._finish, stub)
    stub._skip_test = types.MethodType(DfitApp._skip_test, stub)
    return stub, entry, data_dir, entry2


def test_finish_folder_branch_saves_via_store_and_writes_log(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)

    stub._finish()

    assert os.path.exists(entry.picks_path)
    assert not (data_dir / "w1_picks.json").exists()  # no stem-JSON duplicate in folder mode

    log_path = os.path.join(str(tmp_path), store.LOG_FILENAME)
    assert os.path.exists(log_path)
    log_df = store.load_log(str(tmp_path))
    assert "w1" in log_df["test_id"].tolist()
    assert stub._refresh_calls == [entry]


def test_finish_folder_branch_advances_to_next_new_entry(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)

    stub._finish()

    assert stub._load_test_calls == [entry2]
    assert stub._png_calls == [1]  # Finish still exports PNGs, unlike Skip test


def test_finish_folder_branch_reports_no_new_tests_remain(tmp_path, monkeypatch):
    info_calls = []
    monkeypatch.setattr(ui.messagebox, "showinfo", lambda *a, **kw: info_calls.append((a, kw)))
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch,
                                                 second_status="done")

    stub._finish()

    assert stub._load_test_calls == []
    assert len(info_calls) == 1


def test_finish_single_file_branch_writes_stem_json_no_log(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=False, monkeypatch=monkeypatch)

    stub._finish()

    assert (data_dir / "w1_picks.json").exists()
    assert not (data_dir / store.LOG_FILENAME).exists()


def test_finish_preserves_skip_on_last_step(tmp_path, monkeypatch):
    # "Skip >" on the last step ("porepressure") writes step_status["porepressure"] =
    # "skipped", then next_step clamps in place so the button reads "Finish". Finish must not
    # rewrite that "skipped" back to "done" -- the test should still report skipped overall.
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)
    for key in store.STEP_KEYS:
        stub.state.step_status[key] = "done"
    stub.state.step_status["porepressure"] = "skipped"

    stub._finish()

    assert stub.state.step_status["porepressure"] == "skipped"
    loaded = store.load_picks_for(entry)
    assert loaded.step_status["porepressure"] == "skipped"
    assert store.status_for(loaded) == "skipped"
    assert entry.status == "skipped"


def test_finish_preserves_skip_under_pcf_on_loglog(tmp_path, monkeypatch):
    # Under PC-F, _goto redirects a "porepressure" destination back to "loglog", so Finish can
    # land on self.step == "loglog" with that step already flagged "skipped" -- same hole as
    # the last-step case above, just via a different step key.
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)
    stub.step = "loglog"
    for key in store.STEP_KEYS:
        stub.state.step_status[key] = "done"
    stub.state.step_status["loglog"] = "skipped"

    stub._finish()

    assert stub.state.step_status["loglog"] == "skipped"
    loaded = store.load_picks_for(entry)
    assert loaded.step_status["loglog"] == "skipped"
    assert store.status_for(loaded) == "skipped"
    assert entry.status == "skipped"


def test_finish_unparks_whole_test_skip(tmp_path, monkeypatch):
    # A test previously flagged via Skip test (explicit_status == "skipped") that is later
    # walked to completion and Finished should report "done", not the stale park flag.
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)
    stub.state.explicit_status = "skipped"
    for key in store.STEP_KEYS:
        stub.state.step_status[key] = "done"

    stub._finish()

    assert stub.state.explicit_status is None
    loaded = store.load_picks_for(entry)
    assert loaded.explicit_status is None
    assert store.status_for(loaded) == "done"
    assert entry.status == "done"


def test_skip_test_flags_writes_skipped_status_and_advances(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)

    stub._skip_test()

    loaded = store.load_picks_for(entry)
    assert loaded.explicit_status == "skipped"
    assert entry.status == "skipped"

    log_df = store.load_log(str(tmp_path))
    row = log_df[log_df["test_id"] == "w1"].iloc[0]
    assert row["status"] == "skipped"

    assert stub._load_test_calls == [entry2]
    assert stub._png_calls == []  # a skipped test produces no plots


def test_skip_test_toggle_clears_flag_and_does_not_advance(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=True, monkeypatch=monkeypatch)
    stub.state.explicit_status = "skipped"

    stub._skip_test()

    loaded = store.load_picks_for(entry)
    assert loaded.explicit_status is None
    assert stub._load_test_calls == []


def test_skip_test_noop_in_single_file_mode(tmp_path, monkeypatch):
    stub, entry, data_dir, entry2 = _finish_stub(tmp_path, folder_mode=False, monkeypatch=monkeypatch)

    stub._skip_test()  # must not raise despite no queue/log attributes existing

    assert not (data_dir / store.LOG_FILENAME).exists()


# --------------------------------------------------------------------------------------------------
# _skip_test with REAL DfitApp.refresh() (not the `refresh = lambda: None` stand-in _finish_stub
# uses): this is the regression guard the since-removed test_save_and_next_captures_
# unapplied_widget_edits used to provide for Save & Next, ported onto Skip test now that it
# absorbed that button's "save + log + advance" responsibility. Same headless-Agg real-Figure
# plumbing as test_view_state.py's _refresh_stub.
# --------------------------------------------------------------------------------------------------
def _skip_test_real_refresh_stub(tmp_path):
    entry1_dir = tmp_path / "w1"
    entry1_dir.mkdir()
    csv1 = entry1_dir / "w1.csv"
    csv1.write_text("t,p\n")
    entry1 = store.TestEntry(test_id="w1", folder=str(entry1_dir), csv_path=str(csv1))

    entry2_dir = tmp_path / "w2"
    entry2_dir.mkdir()
    csv2 = entry2_dir / "w2.csv"
    csv2.write_text("t,p\n")
    entry2 = store.TestEntry(test_id="w2", folder=str(entry2_dir), csv_path=str(csv2))

    td = make_testdata()
    td.path = str(csv1)
    state = overview_state(td)
    state.active_source = "csv"
    res = compute_all(state, td)

    stub = types.SimpleNamespace()
    stub.current_entry = entry1
    stub.td = td
    stub.state = state
    stub.res = res
    stub.step = "overview"
    stub.folder_root = str(tmp_path)
    stub.log_df = store.load_log(str(tmp_path))
    stub.queue_entries = [entry1, entry2]
    stub.txt_notes = types.SimpleNamespace(get=lambda *a, **kw: "skip test notes")
    # Widget stand-ins for _sync_state_from_widgets, defaulted to match the state built above
    # (real _skip_test syncs these before refresh()/the log write, same as _finish).
    stub.var_pressure = _Var(state.pressure_col)
    stub.var_rate = _Var(state.rate_col)
    stub.var_volume = _Var(state.volume_col)
    stub.var_isbhp = _Var(state.pressure_is_bhp)
    stub.var_density = _Var(state.density_ppg)
    stub.var_tvd = _Var(state.tvd_ft)
    stub.var_well = _Var(state.well_name)
    stub.var_formation = _Var(state.formation)
    stub.var_alpha = _Var(state.alpha)
    stub.var_step = _Var(state.resample_step)
    stub._sync_state_from_widgets = types.MethodType(DfitApp._sync_state_from_widgets, stub)
    # Real refresh() plumbing (matplotlib only, no Tkinter needed) -- same headless approach as
    # test_view_state.py's _refresh_stub -- so self.res comes out of a real recompute rather
    # than being faked, which is the whole point of the regression guard below.
    stub.fig = Figure()
    stub.ax = stub.fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(stub.fig)
    stub._views = {}
    stub.gate_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub._attach_controllers = lambda: None
    stub._update_stepbar = lambda: None
    stub._update_panel_visibility = lambda: None
    stub._update_panel = lambda: None
    stub._make_range_slider = types.MethodType(DfitApp._make_range_slider, stub)
    stub._build_sliders = types.MethodType(DfitApp._build_sliders, stub)
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    stub._d2_axes = types.MethodType(DfitApp._d2_axes, stub)
    stub._reconcile_pp_axis = types.MethodType(DfitApp._reconcile_pp_axis, stub)
    stub.refresh = types.MethodType(DfitApp.refresh, stub)
    stub._refresh_calls = []
    stub._refresh_queue_row = lambda e: stub._refresh_calls.append(e)
    stub._update_skip_test_btn = lambda: None
    stub._load_test_calls = []
    stub._load_test = lambda e: stub._load_test_calls.append(e)
    stub._write_log_row = types.MethodType(DfitApp._write_log_row, stub)
    stub._advance_queue = types.MethodType(DfitApp._advance_queue, stub)
    stub._skip_test = types.MethodType(DfitApp._skip_test, stub)
    return stub, entry1, entry2


def test_skip_test_captures_unapplied_widget_edits_and_notes(tmp_path):
    """Same regression guard as the since-removed test_save_and_next_captures_unapplied_
    widget_edits: the saved picks JSON and the dfit_log.csv row must reflect the NEW widget
    values, not the OLD ones still sitting in self.state, and non-empty notes must round-trip
    into the saved picks."""
    stub, entry1, entry2 = _skip_test_real_refresh_stub(tmp_path)
    stub.state.density_ppg = 8.0
    stub.state.well_name = "Old Well"
    stub.state.formation = "Old Formation"
    stub.var_density.set(9.3)
    stub.var_well.set("New Well 1H")
    stub.var_formation.set("Eagle Ford")
    # pressure_is_bhp alone flips ChannelConfig.bhp_inputs_ready() regardless of density/tvd, so
    # this is a res-DERIVED column (build_log_row reads res.pressure_is_bhp, not
    # state.pressure_is_bhp directly) -- it only reflects the new widget value if _skip_test's
    # refresh() actually recomputed self.res on the synced state, not just the sync itself.
    assert stub.state.pressure_is_bhp is False
    stub.var_isbhp.set(True)

    stub._skip_test()

    loaded = store.load_picks_for(entry1)
    assert loaded.density_ppg == 9.3
    assert loaded.well_name == "New Well 1H"
    assert loaded.formation == "Eagle Ford"
    assert loaded.pressure_is_bhp is True
    assert loaded.notes == "skip test notes"

    log_df = store.load_log(str(tmp_path))
    row = log_df[log_df["test_id"] == "w1"].iloc[0]
    assert row["well_name"] == "New Well 1H"
    assert row["formation"] == "Eagle Ford"
    assert row["fluid_density"] == 9.3
    assert row["pressure_source"] == "BHP"
    assert row["notes"] == "skip test notes"
