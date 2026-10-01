"""Tests for XLSX support in store.py: TestEntry.xlsx_path, available_sources/data_path with a
third source, and scan_root's sniff-gated inclusion of .xlsx data files (paired with a same-stem
CSV, on its own, excluded when it's a questionnaire, and excluded when it doesn't sniff as data).

Headless (tests/conftest.py forces Agg; no Tk anywhere).
"""

from __future__ import annotations

import datetime as dt

import openpyxl
import pytest

from dfit_tool import store


def _write_data_xlsx(path, n: int = 5) -> None:
    """A minimal workbook that sniffs as DFIT time-series data (see io_load.sniff_xlsx_data)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    d0 = dt.datetime(2020, 1, 1)
    for i in range(n):
        ws.append([d0 + dt.timedelta(seconds=i), 5000 - i])
    wb.save(str(path))


def _write_summary_xlsx(path) -> None:
    """A workbook that does NOT sniff as data (no datetime-named header)."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Joint", "Length (ft)", "Grade"])
    for i in range(5):
        ws.append([i, 42.5, "P110"])
    wb.save(str(path))


# --------------------------------------------------------------------------------------------------
# TestEntry.xlsx_path / available_sources / data_path
# --------------------------------------------------------------------------------------------------
def test_test_entry_available_sources_all_three():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv", dbs_path="a.dbs",
                            xlsx_path="a.xlsx")
    assert entry.available_sources == ["CSV", "DBS", "XLSX"]


def test_test_entry_available_sources_xlsx_only():
    entry = store.TestEntry(test_id="w", folder="f", xlsx_path="a.xlsx")
    assert entry.available_sources == ["XLSX"]


def test_test_entry_data_path_xlsx():
    entry = store.TestEntry(test_id="w", folder="f", xlsx_path="a.xlsx")
    assert entry.data_path("xlsx") == "a.xlsx"
    assert entry.data_path("XLSX") == "a.xlsx"


def test_test_entry_data_path_xlsx_unavailable_raises():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv")
    with pytest.raises(ValueError):
        entry.data_path("xlsx")


def test_test_entry_data_path_truly_unknown_source_raises():
    entry = store.TestEntry(test_id="w", folder="f", csv_path="a.csv")
    with pytest.raises(ValueError):
        entry.data_path("parquet")


# --------------------------------------------------------------------------------------------------
# scan_root
# --------------------------------------------------------------------------------------------------
def test_scan_root_pairs_xlsx_with_same_stem_csv(tmp_path):
    sub = tmp_path / "well1"
    sub.mkdir()
    (sub / "well1.csv").write_text("a")
    _write_data_xlsx(sub / "well1.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.csv_path == str(sub / "well1.csv")
    assert entry.xlsx_path == str(sub / "well1.xlsx")
    assert entry.available_sources == ["CSV", "XLSX"]


def test_scan_root_loose_xlsx_only(tmp_path):
    _write_data_xlsx(tmp_path / "well2.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].test_id == "well2"
    assert entries[0].xlsx_path == str(tmp_path / "well2.xlsx")
    assert entries[0].csv_path is None


def test_scan_root_excludes_non_data_xlsx(tmp_path):
    _write_summary_xlsx(tmp_path / "casing_tally.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert entries == []


def test_scan_root_excludes_questionnaire_named_xlsx(tmp_path):
    # A questionnaire-shaped filename is excluded by name alone, even if its content happened to
    # sniff as data -- is_questionnaire_filename is checked first (see _group_data_files).
    _write_data_xlsx(tmp_path / "well3 questionnaire.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert entries == []


def test_scan_root_excludes_xlsx_lock_file(tmp_path):
    # Excel lock files ("~$...") are covered by the same is_questionnaire_filename predicate.
    _write_data_xlsx(tmp_path / "well4.xlsx")
    (tmp_path / "~$well4.xlsx").write_bytes(b"lock")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].xlsx_path == str(tmp_path / "well4.xlsx")


def test_scan_root_mixed_csv_dbs_xlsx_same_stem_merges(tmp_path):
    sub = tmp_path / "well5"
    sub.mkdir()
    (sub / "well5.csv").write_text("a")
    (sub / "well5.dbs").write_bytes(b"x")
    _write_data_xlsx(sub / "well5.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    entry = entries[0]
    assert entry.csv_path == str(sub / "well5.csv")
    assert entry.dbs_path == str(sub / "well5.dbs")
    assert entry.xlsx_path == str(sub / "well5.xlsx")
    assert entry.available_sources == ["CSV", "DBS", "XLSX"]


# --------------------------------------------------------------------------------------------------
# test_id stability (FIX 1): adding an xlsx must never change an existing csv/dbs test's id
# --------------------------------------------------------------------------------------------------
def test_scan_root_xlsx_only_extra_stem_does_not_shift_csv_test_id(tmp_path):
    # Before xlsx-only stems counted toward the collapse decision, adding this unrelated data
    # xlsx to the folder would have grown by_stem from 1 to 2 groups and pushed well6's own
    # test_id from "well6" out to "well6/well6" -- silently invalidating any already-saved
    # picks JSON keyed on the old id.
    sub = tmp_path / "well6"
    sub.mkdir()
    (sub / "well6.csv").write_text("a")
    _write_data_xlsx(sub / "unrelated_pump_log.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert {e.test_id for e in entries} == {"well6", "well6/unrelated_pump_log"}
    csv_entry = next(e for e in entries if e.test_id == "well6")
    assert csv_entry.csv_path == str(sub / "well6.csv")
    xlsx_entry = next(e for e in entries if e.test_id == "well6/unrelated_pump_log")
    assert xlsx_entry.xlsx_path == str(sub / "unrelated_pump_log.xlsx")
    assert xlsx_entry.csv_path is None


def test_scan_root_two_csv_dbs_groups_plus_xlsx_only_stem_all_qualified(tmp_path):
    # With >1 csv/dbs stem group already, the "multiple stem groups" branch already qualifies
    # every stem including an extra xlsx-only one -- this just confirms the collapse-candidate
    # counting change doesn't disturb that pre-existing, already-correct case.
    sub = tmp_path / "well7"
    sub.mkdir()
    (sub / "a.csv").write_text("a")
    (sub / "b.csv").write_text("a")
    _write_data_xlsx(sub / "c.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert {e.test_id for e in entries} == {"well7/a", "well7/b", "well7/c"}


def test_scan_root_xlsx_only_folder_keeps_old_single_stem_collapse(tmp_path):
    # No csv/dbs at all in this folder -- the collapse rule falls back to the original
    # single-total-stem-group behavior, unchanged.
    sub = tmp_path / "well8"
    sub.mkdir()
    _write_data_xlsx(sub / "well8.xlsx")

    entries = store.scan_root(str(tmp_path))

    assert [e.test_id for e in entries] == ["well8"]
    assert entries[0].xlsx_path == str(sub / "well8.xlsx")


# --------------------------------------------------------------------------------------------------
# lock files (FIX 8): a "~$..." file is excluded outright, before ever being sniffed
# --------------------------------------------------------------------------------------------------
def test_scan_root_excludes_xlsx_lock_file_without_questionnaire_in_name(tmp_path):
    # A lock file for a DATA workbook (no "questionnaire" in its name at all) is not caught by
    # is_questionnaire_filename's own "questionnaire" substring check -- only the explicit "~$"
    # prefix check in _group_data_files excludes it. Written as a byte-identical copy of a real
    # sniffable workbook, so this proves the explicit filename check fires (and skips ever
    # opening/sniffing it), not merely that a garbage lock file happens to fail the sniff.
    sub = tmp_path / "well9"
    sub.mkdir()
    real = sub / "well9.xlsx"
    _write_data_xlsx(real)
    (sub / "~$well9.xlsx").write_bytes(real.read_bytes())

    entries = store.scan_root(str(tmp_path))

    assert len(entries) == 1
    assert entries[0].xlsx_path == str(real)


def test_scan_root_xlsx_only_subfolder_does_not_displace_loose_csv(tmp_path):
    (tmp_path / "W.csv").write_text("a")
    (tmp_path / "Z.csv").write_text("a")
    sub = tmp_path / "W"
    sub.mkdir()
    _write_data_xlsx(sub / "pump.xlsx")

    entries = store.scan_root(str(tmp_path))

    ids = {e.test_id for e in entries}
    assert ids == {"W", "Z", "W/pump"}
    w = next(e for e in entries if e.test_id == "W")
    assert w.csv_path == str(tmp_path / "W.csv")
    moved = next(e for e in entries if e.test_id == "W/pump")
    assert moved.xlsx_path == str(sub / "pump.xlsx") and moved.csv_path is None
    assert any("renamed" in x for x in moved.scan_warnings)
