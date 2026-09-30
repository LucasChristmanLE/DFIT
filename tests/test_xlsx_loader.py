"""Tests for the XLSX loader (``load_xlsx``, ``sniff_xlsx_data``) in dfit_tool/io_load.py.

All fixtures are synthetic workbooks built with openpyxl under ``tmp_path`` -- no test here reads
anything under ``C:\\DFIT Data``. Each fixture reproduces, in miniature, one of the three real
corpus shapes named in CLAUDE.md (a single "Job Data Listing" sheet with a full-datetime "Time"
column; a chart sheet + that same data sheet + a "comments" sheet; and a downhole-gauge sheet
with a preamble, a split "Real Date"/"Real Time" pair, and a units row).
"""

from __future__ import annotations

import datetime as dt

import openpyxl
import pytest

from dfit_tool import io_load


def _save(wb, path) -> str:
    wb.save(str(path))
    return str(path)


# --------------------------------------------------------------------------------------------------
# preamble + header + units row + data (Southern Ute downhole-gauge shape)
# --------------------------------------------------------------------------------------------------
def _build_gauge_workbook(path, n: int = 20):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(["Company Name: RED WILLOW CONSTRUCTION"])
    ws.append(["Well Name: SOUTE 32-8"])
    ws.append([])
    ws.append([])
    ws.append([])
    ws.append([])
    ws.append(["Real Date", "Real Time", "Test Time", "Pressure", "Temperature"])
    ws.append(["MM/DD/YY", "HH:MM:SS", "minutes", "psiG", "deg F."])
    d0 = dt.datetime(2020, 1, 1, 0, 0, 0)
    for i in range(n):
        ws.append([d0, dt.time(0, 0, i), i / 60.0, 5000 - i, 150 + i])
    comments = wb.create_sheet("comments")
    comments.append(["some comment text"])
    return _save(wb, path)


def test_load_xlsx_preamble_header_units_row(tmp_path):
    path = _build_gauge_workbook(tmp_path / "gauge.xlsx")
    td = io_load.load_xlsx(path)

    assert td.datetime_col == "Real Date (MM/DD/YY)"
    assert set(["Real Date (MM/DD/YY)", "Real Time (HH:MM:SS)", "Test Time (minutes)",
                "Pressure (psiG)", "Temperature (deg F.)"]) <= set(td.columns)
    assert td.n == 20
    # Full 1 Hz resolution retained -- the companion "Real Time" join must not collapse every
    # sample onto the same midnight date.
    assert td.t_s[0] == pytest.approx(0.0)
    assert td.t_s[-1] == pytest.approx(19.0)
    assert list(td.t_s) == sorted(td.t_s)


def test_sniff_xlsx_data_true_for_gauge_shape(tmp_path):
    path = _build_gauge_workbook(tmp_path / "gauge.xlsx")
    assert io_load.sniff_xlsx_data(path) is True


def test_load_xlsx_companion_time_join_full_resolution(tmp_path):
    # Same shape, larger n and a companion "Test Time" elapsed-minutes column that must NOT be
    # mistaken for the "Real Time" companion (see io_load._companion_time_col).
    path = _build_gauge_workbook(tmp_path / "gauge2.xlsx", n=5)
    td = io_load.load_xlsx(path)
    guess = io_load.suggest_channels(td.columns)
    assert guess["datetime"] == "Real Date (MM/DD/YY)"
    assert guess["time"] == "Real Time (HH:MM:SS)"


def test_load_xlsx_folds_units_row_into_header_names(tmp_path):
    path = _build_gauge_workbook(tmp_path / "gauge3.xlsx")
    td = io_load.load_xlsx(path)
    assert "Pressure (psiG)" in td.columns
    assert "Temperature (deg F.)" in td.columns


# --------------------------------------------------------------------------------------------------
# chart sheet + data sheet + comments sheet (Job Data Listing shape)
# --------------------------------------------------------------------------------------------------
def _build_job_data_workbook(path, n: int = 5, with_chart: bool = True):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Job Data Listing"
    ws.append(["Time", "Tubing Pressure", "Backside Pressure", "I1 Flow 1 Rate", "Job Clean Vol"])
    ws.append(["(hh:mm:ss)", "(psi)", "(psi)", "(bpm)", "(gal)"])
    t0 = dt.datetime(2020, 1, 1, 0, 0, 0)
    for i in range(n):
        ws.append([t0 + dt.timedelta(seconds=i), 5000 + i, 100 + i, 2.5, 1000 + i])
    comments = wb.create_sheet("comments")
    comments.append(["Timeline comments"])
    if with_chart:
        from openpyxl.chart import LineChart, Reference
        chart = LineChart()
        data = Reference(ws, min_col=2, min_row=1, max_row=n + 2)
        chart.add_data(data, titles_from_data=True)
        chartsheet = wb.create_chartsheet("Chart1")
        chartsheet.add_chart(chart)
    return _save(wb, path)


def test_load_xlsx_skips_chart_and_comments_sheets(tmp_path):
    path = _build_job_data_workbook(tmp_path / "job.xlsx")
    td = io_load.load_xlsx(path)

    assert td.datetime_col == "Time (hh:mm:ss)"
    assert td.columns == [
        "Time (hh:mm:ss)", "Tubing Pressure (psi)", "Backside Pressure (psi)",
        "I1 Flow 1 Rate (bpm)", "Job Clean Vol (gal)",
    ]
    assert td.n == 5


def test_sniff_xlsx_data_true_for_job_data_shape(tmp_path):
    path = _build_job_data_workbook(tmp_path / "job2.xlsx")
    assert io_load.sniff_xlsx_data(path) is True


def test_load_xlsx_picks_sheet_with_most_data_rows(tmp_path):
    # Two qualifying sheets in one workbook -- the one with more data rows must win.
    wb = openpyxl.Workbook()
    small = wb.active
    small.title = "Small"
    small.append(["Date", "Pressure"])
    d0 = dt.datetime(2020, 1, 1)
    for i in range(3):
        small.append([d0 + dt.timedelta(seconds=i), 100 + i])
    big = wb.create_sheet("Big")
    big.append(["Date", "Pressure"])
    for i in range(50):
        big.append([d0 + dt.timedelta(seconds=i), 200 + i])
    path = _save(wb, tmp_path / "two_sheets.xlsx")

    td = io_load.load_xlsx(path)
    assert td.n == 50


# --------------------------------------------------------------------------------------------------
# non-data summary workbook -- sniff must return False
# --------------------------------------------------------------------------------------------------
def test_sniff_xlsx_data_false_for_summary_workbook(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Casing Tally"
    ws.append(["Joint", "Length (ft)", "Weight (lb/ft)", "Grade"])
    for i in range(10):
        ws.append([i, 42.5, 24.0, "P110"])
    path = _save(wb, tmp_path / "summary.xlsx")

    assert io_load.sniff_xlsx_data(path) is False
    with pytest.raises(ValueError):
        io_load.load_xlsx(path)


def test_sniff_xlsx_data_false_for_blank_workbook(tmp_path):
    wb = openpyxl.Workbook()
    path = _save(wb, tmp_path / "blank.xlsx")
    assert io_load.sniff_xlsx_data(path) is False


def test_sniff_xlsx_data_false_for_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(b"not a real xlsx file")
    assert io_load.sniff_xlsx_data(str(path)) is False


def test_sniff_xlsx_data_false_for_missing_file(tmp_path):
    assert io_load.sniff_xlsx_data(str(tmp_path / "does_not_exist.xlsx")) is False


# --------------------------------------------------------------------------------------------------
# header detection must not false-positive on a multi-field preamble row
# --------------------------------------------------------------------------------------------------
def test_load_xlsx_preamble_row_with_date_and_numbers_not_mistaken_for_header(tmp_path):
    # Southern Ute's real preamble row 1 is several separate string cells, one literally "Date",
    # sitting right next to two real datetime values -- a wide numeric-data lookahead would
    # false-positive on this line instead of reaching the real header several rows down.
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Company", "Name:", "RED", "WILLOW", "PRODUCTION", "COMPANY", "Page:", 1])
    ws.append(["Well", "Name:", "SOUTHERN", "UTE", "32-8", "NO.",
               dt.datetime(2015, 6, 1), "Date", "of", "Test:", dt.datetime(2014, 12, 18)])
    ws.append([])
    ws.append([])
    ws.append([])
    ws.append([])
    ws.append(["Real Date", "Real Time", "Test Time", "Pressure", "Temperature"])
    ws.append(["MM/DD/YY", "HH:MM:SS", "minutes", "psiG", "deg F."])
    d0 = dt.datetime(2014, 12, 18)
    for i in range(5):
        ws.append([d0, dt.time(12, 52, i), 2.0 + i / 60.0, 0.01, 66.0 + i])
    path = _save(wb, tmp_path / "preamble.xlsx")

    td = io_load.load_xlsx(path)
    assert td.datetime_col == "Real Date (MM/DD/YY)"
    assert td.n == 5


# --------------------------------------------------------------------------------------------------
# dedupe / trailing-empty trimming
# --------------------------------------------------------------------------------------------------
def test_load_xlsx_dedupes_duplicate_header_names(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure", "Pressure"])
    d0 = dt.datetime(2020, 1, 1)
    for i in range(3):
        ws.append([d0 + dt.timedelta(seconds=i), 100 + i, 200 + i])
    path = _save(wb, tmp_path / "dupe.xlsx")

    td = io_load.load_xlsx(path)
    assert td.columns == ["Date", "Pressure", "Pressure.1"]


def test_load_xlsx_drops_trailing_blank_column_and_rows(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure", None])
    d0 = dt.datetime(2020, 1, 1)
    for i in range(3):
        ws.append([d0 + dt.timedelta(seconds=i), 100 + i, None])
    # A stray, fully blank trailing row.
    ws.append([None, None, None])
    path = _save(wb, tmp_path / "trailing_blank.xlsx")

    td = io_load.load_xlsx(path)
    assert td.columns == ["Date", "Pressure"]
    assert td.n == 3


# --------------------------------------------------------------------------------------------------
# suggest_channels role mapping against folded XLSX column names
# --------------------------------------------------------------------------------------------------
def test_load_xlsx_suggest_channels_maps_roles(tmp_path):
    path = _build_job_data_workbook(tmp_path / "job3.xlsx", with_chart=False)
    td = io_load.load_xlsx(path)
    guess = io_load.suggest_channels(td.columns, column=td.column)
    assert guess["pressure"] == "Tubing Pressure (psi)"
    assert guess["rate"] == "I1 Flow 1 Rate (bpm)"
    assert guess["volume"] == "Job Clean Vol (gal)"


# --------------------------------------------------------------------------------------------------
# load() dispatch
# --------------------------------------------------------------------------------------------------
def test_load_dispatches_to_xlsx_case_insensitive(tmp_path):
    path = _build_job_data_workbook(tmp_path / "JOB.XLSX", with_chart=False)
    td = io_load.load(path)
    assert td.datetime_col == "Time (hh:mm:ss)"
