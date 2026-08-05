"""Unit tests for store.py: folder scanning, picks persistence, status semantics, and the
dfit_log.csv master log. Headless (tests/conftest.py forces Agg; no Tk anywhere) -- store.py
itself never imports tkinter or ui.

Also covers the two new PickState fields (active_source, explicit_status) added alongside
store.py in the same task: a plain encode/decode round trip.
"""

from __future__ import annotations

import datetime
import getpass
import os

import pandas as pd
import pytest

from dfit_tool import store
from dfit_tool.model import PickState, _decode, compute_all
from tests.helpers import make_testdata, overview_state


# --------------------------------------------------------------------------------------------------
# Part 1: PickState.active_source / explicit_status
# --------------------------------------------------------------------------------------------------
def test_pickstate_new_fields_default():
    st = PickState()
    assert st.active_source == "csv"
    assert st.explicit_status is None


def test_pickstate_roundtrip_preserves_new_fields(tmp_path):
    path = tmp_path / "picks.json"
    st = PickState(active_source="dbs", explicit_status="done")
    st.to_json(str(path))

    loaded = PickState.from_json(str(path))

    assert loaded.active_source == "dbs"
    assert loaded.explicit_status == "done"


def test_decode_without_new_fields_takes_defaults():
    loaded = _decode({"pressure_col": "P"})
    assert loaded.active_source == "csv"
    assert loaded.explicit_status is None


# --------------------------------------------------------------------------------------------------
# Part 2: TestEntry
# --------------------------------------------------------------------------------------------------
def test_test_entry_picks_path():
    entry = store.TestEntry(test_id="well1", folder=os.path.join("root", "well1"))
    assert entry.picks_path == os.path.join("root", "well1", "well1" + store.PICKS_SUFFIX)


def test_test_entry_picks_path_uses_picks_basename_override():
    entry = store.TestEntry(
        test_id="CustomerA/Well1", folder=os.path.join("root", "Well1"), picks_basename="Well1_DFIT"
    )
    assert entry.picks_path == os.path.join("root", "Well1", "Well1_DFIT" + store.PICKS_SUFFIX)


def test_test_entry_display_label_nested():
    entry = store.TestEntry(test_id="CustomerA/Well1", folder="f")
    assert entry.display_label == "CustomerA / Well1"


def test_test_entry_display_label_bare_unchanged():
    entry = store.TestEntry(test_id="well1", folder="f")
    assert entry.display_label == "well1"


def test_test_entry_available_sources_csv_first():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv", dbs_path="a.dbs")
    assert entry.available_sources == ["CSV", "DBS"]


def test_test_entry_available_sources_dbs_only():
    entry = store.TestEntry(test_id="w", folder="f", dbs_path="a.dbs")
    assert entry.available_sources == ["DBS"]


def test_test_entry_data_path_case_insensitive():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv", dbs_path="a.dbs")
    assert entry.data_path("csv") == "a.csv"
    assert entry.data_path("DBS") == "a.dbs"


def test_test_entry_data_path_unavailable_raises():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv")
    with pytest.raises(ValueError):
        entry.data_path("dbs")


def test_test_entry_data_path_unknown_source_raises():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv")
    with pytest.raises(ValueError):
        entry.data_path("xlsx")


# --------------------------------------------------------------------------------------------------
# scan_root
# --------------------------------------------------------------------------------------------------
def test_scan_root_subfolder_layout(tmp_path):
    sub = tmp_path / "well1"
    sub.mkdir()
    (sub / "well1.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].test_id == "well1"
    assert entries[0].folder == str(sub)
    assert entries[0].csv_path == str(sub / "well1.csv")
    assert entries[0].dbs_path is None


def test_scan_root_flat_layout_loose_csv(tmp_path):
    (tmp_path / "well2.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].test_id == "well2"
    assert entries[0].folder == str(tmp_path)
    assert entries[0].csv_path == str(tmp_path / "well2.csv")


def test_scan_root_mixed_csv_dbs_same_stem_merges(tmp_path):
    sub = tmp_path / "well3"
    sub.mkdir()
    (sub / "well3.csv").write_text("a")
    (sub / "well3.dbs").write_bytes(b"x")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.csv_path == str(sub / "well3.csv")
    assert entry.dbs_path == str(sub / "well3.dbs")


def test_scan_root_flat_layout_same_stem_csv_dbs_merges(tmp_path):
    (tmp_path / "well4.csv").write_text("a")
    (tmp_path / "well4.dbs").write_bytes(b"x")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.csv_path == str(tmp_path / "well4.csv")
    assert entry.dbs_path == str(tmp_path / "well4.dbs")


def test_scan_root_multiple_stems_in_one_dir_yield_two_tests_no_warning(tmp_path):
    sub = tmp_path / "well5"
    sub.mkdir()
    (sub / "b.csv").write_text("a")
    (sub / "a.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["well5/a", "well5/b"]
    assert all(not e.scan_warnings for e in entries)


def test_scan_root_ignores_other_extensions(tmp_path):
    sub = tmp_path / "well6"
    sub.mkdir()
    (sub / "well6.csv").write_text("a")
    (sub / "well6.zip").write_bytes(b"x")
    (sub / "well6.INP").write_text("x")
    (sub / "well6.pdf").write_bytes(b"x")
    (sub / "well6questionnaire.xlsx").write_bytes(b"x")
    (sub / "well6.json").write_text("{}")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].csv_path == str(sub / "well6.csv")


def test_scan_root_excludes_own_log_file(tmp_path):
    (tmp_path / store.LOG_FILENAME).write_text("file,test_id\n")
    (tmp_path / "DFIT_LOG.CSV").write_text("file,test_id\n")  # case-insensitive check
    (tmp_path / "well7.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["well7"]


def test_scan_root_directory_with_no_data_files_skipped(tmp_path):
    sub = tmp_path / "not_a_test"
    sub.mkdir()
    (sub / "readme.txt").write_text("x")

    entries = store.scan_root(str(tmp_path))

    assert entries == []


def test_scan_root_case_insensitive_extension(tmp_path):
    sub = tmp_path / "well8"
    sub.mkdir()
    (sub / "well8.DBS").write_bytes(b"x")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].dbs_path == str(sub / "well8.DBS")


def test_scan_root_loose_file_collides_with_dir_dropped_with_warning(tmp_path):
    sub = tmp_path / "well1"
    sub.mkdir()
    (sub / "well1.csv").write_text("a")
    (tmp_path / "well1.csv").write_text("b")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.test_id == "well1"
    assert entry.folder == str(sub)
    assert entry.csv_path == str(sub / "well1.csv")
    assert any("well1" in w for w in entry.scan_warnings)


def test_scan_root_sorted_by_test_id(tmp_path):
    for name in ("zeta", "alpha", "mid"):
        sub = tmp_path / name
        sub.mkdir()
        (sub / f"{name}.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["alpha", "mid", "zeta"]


def test_scan_root_depth_two_nested_single_stem(tmp_path):
    sub = tmp_path / "CustomerA" / "Well1"
    sub.mkdir(parents=True)
    (sub / "Well1_DFIT.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.test_id == "CustomerA/Well1"
    assert entry.display_label == "CustomerA / Well1"
    assert entry.picks_basename == "Well1_DFIT"
    assert entry.csv_path == str(sub / "Well1_DFIT.csv")


def test_scan_root_depth_three_nested(tmp_path):
    sub = tmp_path / "Customer" / "Well" / "Stage"
    sub.mkdir(parents=True)
    (sub / "data.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].test_id == "Customer/Well/Stage"


def test_scan_root_multiple_loose_wells_in_customer_folder(tmp_path):
    sub = tmp_path / "CustomerB"
    sub.mkdir()
    (sub / "well_x.csv").write_text("a")
    (sub / "well_y.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["CustomerB/well_x", "CustomerB/well_y"]
    assert entries[0].picks_basename == "well_x"
    assert entries[1].picks_basename == "well_y"
    assert all(not e.scan_warnings for e in entries)


def test_scan_root_nested_multi_stem_dir_merges_csv_and_dbs(tmp_path):
    sub = tmp_path / "CustomerC"
    sub.mkdir()
    (sub / "well_x.csv").write_text("a")
    (sub / "well_x.dbs").write_bytes(b"x")
    (sub / "well_y.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))

    entry = next(e for e in entries if e.test_id == "CustomerC/well_x")
    assert entry.csv_path == str(sub / "well_x.csv")
    assert entry.dbs_path == str(sub / "well_x.dbs")


def test_scan_root_prunes_dfit_plots_export_dir(tmp_path):
    (tmp_path / "well1.csv").write_text("a")
    export_dir = tmp_path / "well1 DFIT plots"
    export_dir.mkdir()
    (export_dir / "stray.csv").write_text("x")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["well1"]


def test_scan_root_root_loose_vs_subfolder_still_dedups(tmp_path):
    sub = tmp_path / "well1"
    sub.mkdir()
    (sub / "well1.csv").write_text("a")
    (tmp_path / "well1.csv").write_text("b")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.test_id == "well1"
    assert entry.folder == str(sub)
    assert any("well1" in w for w in entry.scan_warnings)


# --------------------------------------------------------------------------------------------------
# scan_root progress callback: the UI-facing hook a folder-open modal pumps to show life during a
# slow scan. Optional and additive -- must not alter what gets scanned.
# --------------------------------------------------------------------------------------------------
def test_scan_root_progress_callback_reports_running_count_without_altering_results(tmp_path):
    for name in ("well1", "well2"):
        sub = tmp_path / name
        sub.mkdir()
        (sub / f"{name}.csv").write_text("a")

    calls = []
    entries = store.scan_root(str(tmp_path), progress=lambda dirs, tests: calls.append((dirs, tests)))

    assert calls  # called at least once
    assert calls[-1][1] == len(entries)  # final tests_found matches what was returned
    assert entries == store.scan_root(str(tmp_path))  # progress must not alter results


def test_scan_root_and_list_tests_default_progress_none_unchanged(tmp_path):
    (tmp_path / "well1.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))
    assert [e.test_id for e in entries] == ["well1"]

    entries2, log_df = store.list_tests(str(tmp_path))
    assert [e.test_id for e in entries2] == ["well1"]
    assert list(log_df.columns) == store.LOG_COLUMNS


def test_nested_entry_save_and_load_picks_roundtrip(tmp_path):
    sub = tmp_path / "CustomerA" / "Well1"
    sub.mkdir(parents=True)
    (sub / "Well1_DFIT.csv").write_text("a")

    entries = store.scan_root(str(tmp_path))
    entry = entries[0]
    assert "/" in entry.test_id

    st = PickState(pressure_col="P")
    store.save_picks_for(entry, st)

    expected_path = os.path.join(str(sub), "Well1_DFIT" + store.PICKS_SUFFIX)
    assert os.path.exists(expected_path)
    assert entry.picks_path == expected_path

    loaded = store.load_picks_for(entry)
    assert loaded is not None
    assert loaded.pressure_col == "P"


# --------------------------------------------------------------------------------------------------
# Picks persistence
# --------------------------------------------------------------------------------------------------
def test_load_picks_for_missing_returns_none(tmp_path):
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    assert store.load_picks_for(entry) is None


def test_load_picks_for_corrupt_json_returns_none(tmp_path):
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    with open(entry.picks_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")

    assert store.load_picks_for(entry) is None


def test_save_and_load_picks_roundtrip(tmp_path):
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    st = PickState(pressure_col="P", active_source="dbs", explicit_status="skipped")

    store.save_picks_for(entry, st)
    loaded = store.load_picks_for(entry)

    assert loaded is not None
    assert loaded.pressure_col == "P"
    assert loaded.active_source == "dbs"
    assert loaded.explicit_status == "skipped"


def test_save_picks_for_is_atomic_no_leftover_temp_files(tmp_path):
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    store.save_picks_for(entry, PickState())

    names = os.listdir(tmp_path)
    assert names == [os.path.basename(entry.picks_path)]


# --------------------------------------------------------------------------------------------------
# status_for
# --------------------------------------------------------------------------------------------------
def test_status_for_none_is_new():
    assert store.status_for(None) == "new"


def test_status_for_empty_step_status_is_new():
    st = PickState()
    assert store.status_for(st) == "new"


def test_status_for_partial_is_in_progress():
    st = PickState(step_status={"overview": "done", "isip": "done"})
    assert store.status_for(st) == "in_progress"


def test_status_for_all_done_is_done():
    st = PickState(step_status={k: "done" for k in store.STEP_KEYS})
    assert store.status_for(st) == "done"


def test_status_for_done_and_skipped_mix_is_done():
    st = PickState(step_status={
        "overview": "done", "isip": "done", "gfunction": "skipped",
        "tangent": "done", "loglog": "done", "porepressure": "done",
    })
    assert store.status_for(st) == "done"


def test_status_for_explicit_status_done_overrides():
    st = PickState(step_status={}, explicit_status="done")
    assert store.status_for(st) == "done"


def test_status_for_explicit_status_skipped_overrides():
    st = PickState(step_status={"overview": "done"}, explicit_status="skipped")
    assert store.status_for(st) == "skipped"


def test_status_for_pcf_without_porepressure_step_is_done():
    st = PickState(
        postclosure_scenario="PC-F no peak",
        step_status={
            "overview": "done", "isip": "done", "gfunction": "done",
            "tangent": "done", "loglog": "done",
        },
    )
    assert store.status_for(st) == "done"


def test_status_for_same_step_status_without_pcf_is_in_progress():
    st = PickState(
        postclosure_scenario="PC-A linear",
        step_status={
            "overview": "done", "isip": "done", "gfunction": "done",
            "tangent": "done", "loglog": "done",
        },
    )
    assert store.status_for(st) == "in_progress"


# --------------------------------------------------------------------------------------------------
# Master log
# --------------------------------------------------------------------------------------------------
def test_load_log_missing_returns_empty_with_columns(tmp_path):
    df = store.load_log(str(tmp_path))
    assert list(df.columns) == store.LOG_COLUMNS
    assert len(df) == 0


def test_save_and_load_log_roundtrip(tmp_path):
    df = pd.DataFrame([{c: "" for c in store.LOG_COLUMNS}])
    df.loc[0, "test_id"] = "well1"

    store.save_log(str(tmp_path), df)
    loaded = store.load_log(str(tmp_path))

    assert list(loaded.columns) == store.LOG_COLUMNS
    assert loaded.loc[0, "test_id"] == "well1"


def test_load_log_adds_missing_newer_columns(tmp_path):
    old_columns = [c for c in store.LOG_COLUMNS if c != "Shmin_rapid"]
    old_df = pd.DataFrame([{c: "" for c in old_columns}])
    old_df.loc[0, "test_id"] = "well1"
    path = tmp_path / store.LOG_FILENAME
    old_df.to_csv(path, index=False)

    loaded = store.load_log(str(tmp_path))

    assert list(loaded.columns) == store.LOG_COLUMNS
    assert loaded.loc[0, "test_id"] == "well1"
    assert pd.isna(loaded.loc[0, "Shmin_rapid"])


def test_load_log_backfills_well_name_and_formation(tmp_path):
    # A dfit_log.csv from before this feature has neither column -- load_log must append both
    # (empty) without disturbing existing row data (a pre-existing regression risk, since
    # inserting columns ahead of others in LOG_COLUMNS could otherwise misalign a reindex).
    old_columns = [c for c in store.LOG_COLUMNS if c not in ("well_name", "formation")]
    old_df = pd.DataFrame([{c: "" for c in old_columns}])
    old_df.loc[0, "test_id"] = "well1"
    old_df.loc[0, "status"] = "done"
    path = tmp_path / store.LOG_FILENAME
    old_df.to_csv(path, index=False)

    loaded = store.load_log(str(tmp_path))

    assert list(loaded.columns) == store.LOG_COLUMNS
    assert loaded.loc[0, "test_id"] == "well1"
    assert loaded.loc[0, "status"] == "done"
    assert pd.isna(loaded.loc[0, "well_name"])
    assert pd.isna(loaded.loc[0, "formation"])


def test_load_log_zero_byte_file_returns_empty_with_columns(tmp_path):
    (tmp_path / store.LOG_FILENAME).write_text("")

    df = store.load_log(str(tmp_path))

    assert list(df.columns) == store.LOG_COLUMNS
    assert len(df) == 0


def test_load_log_zero_byte_file_does_not_break_list_tests(tmp_path):
    (tmp_path / store.LOG_FILENAME).write_text("")
    (tmp_path / "well1.csv").write_text("a")

    entries, df = store.list_tests(str(tmp_path))

    assert [e.test_id for e in entries] == ["well1"]
    assert list(df.columns) == store.LOG_COLUMNS


def test_load_log_numeric_test_id_stays_string(tmp_path):
    df = pd.DataFrame(columns=store.LOG_COLUMNS)
    row = {c: None for c in store.LOG_COLUMNS}
    row["test_id"] = "7170"
    row["status"] = "new"
    df = store.upsert_log_row(df, row)
    store.save_log(str(tmp_path), df)

    loaded = store.load_log(str(tmp_path))
    row2 = dict(row)
    row2["status"] = "done"
    updated = store.upsert_log_row(loaded, row2)
    store.save_log(str(tmp_path), updated)
    reloaded = store.load_log(str(tmp_path))

    matches = reloaded[reloaded["test_id"] == "7170"]
    assert len(matches) == 1
    assert matches.iloc[0]["status"] == "done"


def test_upsert_log_row_insert(tmp_path):
    df = pd.DataFrame(columns=store.LOG_COLUMNS)
    row = {c: None for c in store.LOG_COLUMNS}
    row["test_id"] = "well1"

    out = store.upsert_log_row(df, row)

    assert len(out) == 1
    assert out.iloc[0]["test_id"] == "well1"


def test_upsert_log_row_update_replaces_existing(tmp_path):
    df = pd.DataFrame(columns=store.LOG_COLUMNS)
    row1 = {c: None for c in store.LOG_COLUMNS}
    row1["test_id"] = "well1"
    row1["status"] = "new"
    df = store.upsert_log_row(df, row1)

    row2 = dict(row1)
    row2["status"] = "done"
    out = store.upsert_log_row(df, row2)

    assert len(out) == 1
    assert out.iloc[0]["status"] == "done"


def test_list_tests_keeps_orphaned_log_rows(tmp_path):
    (tmp_path / "well1.csv").write_text("a")
    df = pd.DataFrame(columns=store.LOG_COLUMNS)
    row = {c: None for c in store.LOG_COLUMNS}
    row["test_id"] = "orphan_well"
    df = store.upsert_log_row(df, row)
    store.save_log(str(tmp_path), df)

    entries, loaded_df = store.list_tests(str(tmp_path))

    assert [e.test_id for e in entries] == ["well1"]
    assert "orphan_well" in loaded_df["test_id"].tolist()


def test_list_tests_recomputes_status_from_picks(tmp_path):
    (tmp_path / "well1.csv").write_text("a")
    entry_stub = store.TestEntry(test_id="well1", folder=str(tmp_path))
    store.save_picks_for(entry_stub, PickState(explicit_status="done"))

    entries, _ = store.list_tests(str(tmp_path))

    assert entries[0].status == "done"


# --------------------------------------------------------------------------------------------------
# build_log_row
# --------------------------------------------------------------------------------------------------
def _built_row(tmp_path, **state_kwargs):
    td = make_testdata()
    state = overview_state(td)
    for k, v in state_kwargs.items():
        setattr(state, k, v)
    res = compute_all(state, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = os.path.join(str(tmp_path), "well1.csv")
    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)
    return row, state, res


def test_log_columns_includes_well_name_and_formation():
    assert "well_name" in store.LOG_COLUMNS
    assert "formation" in store.LOG_COLUMNS


def test_build_log_row_has_every_column_in_order(tmp_path):
    row, _, _ = _built_row(tmp_path)
    assert list(row.keys()) == store.LOG_COLUMNS


def test_build_log_row_maps_well_name_and_formation(tmp_path):
    row, _, _ = _built_row(tmp_path, well_name="Foo State 1H", formation="Eagle Ford")
    assert row["well_name"] == "Foo State 1H"
    assert row["formation"] == "Eagle Ford"


def test_build_log_row_stamps_interpreter_and_review_date(tmp_path):
    row, _, _ = _built_row(tmp_path)
    assert row["interpreter"] == getpass.getuser()
    assert row["review_date"] == datetime.date.today().isoformat()


def test_build_log_row_file_is_relative_to_root(tmp_path):
    row, _, _ = _built_row(tmp_path)
    assert row["file"] == "well1.csv"


def test_build_log_row_closure_quality_prefix_map(tmp_path):
    row, _, _ = _built_row(tmp_path, closure_scenario="C-B adequate")
    assert row["closure_quality"] == "adequate"


def test_build_log_row_closure_quality_empty_for_unknown(tmp_path):
    row, _, _ = _built_row(tmp_path, closure_scenario="")
    assert row["closure_quality"] == ""


def test_build_log_row_postclosure_trend_prefix_map(tmp_path):
    row, _, _ = _built_row(tmp_path, postclosure_scenario="PC-B false-radial")
    assert row["postclosure_trend"] == "false-radial"


def test_build_log_row_pp_confidence_low_for_pce(tmp_path):
    row, _, _ = _built_row(tmp_path, postclosure_scenario="PC-E no trend")
    assert row["pp_confidence"] == "low"


def test_build_log_row_pp_confidence_empty_otherwise(tmp_path):
    row, _, _ = _built_row(tmp_path, postclosure_scenario="PC-A linear")
    assert row["pp_confidence"] == ""


def test_build_log_row_closure_time_seconds_to_minutes(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    res.closure_time_compliance_s = 120.0
    res.closure_time_tangent_s = 60.0
    res.closure_time_variable_s = 90.0
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = os.path.join(str(tmp_path), "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)

    assert row["closure_time_compliance_min"] == 2.0
    assert row["closure_time_tangent_min"] == 1.0
    assert row["closure_time_variable_min"] == 1.5


def test_build_log_row_closure_time_none_stays_none(tmp_path):
    row, _, _ = _built_row(tmp_path)
    assert row["closure_time_compliance_min"] is None


def test_build_log_row_pressure_source_bhp_vs_whp(tmp_path):
    row, _, res = _built_row(tmp_path)
    assert row["pressure_source"] == ("BHP" if res.pressure_is_bhp else "WHP")


def test_log_row_has_net_pressure_isip_source(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    res.net_pressure_isip_source = "compliance"
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = os.path.join(str(tmp_path), "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)

    assert "net_pressure_isip_source" in store.LOG_COLUMNS
    assert row["net_pressure_isip_source"] == "compliance"


def test_log_row_has_near_wellbore_complexity(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    res.near_wellbore_complexity = 107.0
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = os.path.join(str(tmp_path), "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)

    # Appended last so an existing dfit_log.csv stays loadable.
    assert store.LOG_COLUMNS[-1] == "near_wellbore_complexity"
    assert row["near_wellbore_complexity"] == 107.0


def test_load_log_backfills_missing_near_wellbore_complexity(tmp_path):
    old_columns = [c for c in store.LOG_COLUMNS if c != "near_wellbore_complexity"]
    old_df = pd.DataFrame([{c: "" for c in old_columns}])
    old_df.loc[0, "test_id"] = "well1"
    path = tmp_path / store.LOG_FILENAME
    old_df.to_csv(path, index=False)

    loaded = store.load_log(str(tmp_path))

    assert list(loaded.columns) == store.LOG_COLUMNS
    assert loaded.loc[0, "test_id"] == "well1"
    assert pd.isna(loaded.loc[0, "near_wellbore_complexity"])
