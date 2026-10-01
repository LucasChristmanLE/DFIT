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
# zip-peek sniff internals (FIX 3): cell-type conversion, shared strings, worksheet-part
# resolution, and never-raise/always-close behavior
# --------------------------------------------------------------------------------------------------
def test_xlsx_zip_convert_row_handles_each_cell_kind():
    row = [("s", 0), ("str", "inline"), ("b", "1"), ("b", "0"), ("n", "42.5"), None]
    shared = {0: "shared value"}
    converted = io_load._xlsx_zip_convert_row(row, shared)
    assert converted == ("shared value", "inline", True, False, 42.5, None)


def test_xlsx_zip_convert_row_bad_number_text_passes_through():
    row = [("n", "not-a-number")]
    assert io_load._xlsx_zip_convert_row(row, {}) == ("not-a-number",)


def test_xlsx_zip_shared_strings_missing_file_returns_empty(tmp_path):
    import zipfile
    path = tmp_path / "no_shared_strings.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", "<workbook/>")
    with zipfile.ZipFile(path) as z:
        assert io_load._xlsx_zip_shared_strings(z, {0, 1}) == {}


def test_xlsx_zip_shared_strings_empty_need_skips_read(tmp_path):
    import zipfile
    path = tmp_path / "unread_shared_strings.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/sharedStrings.xml", "not even valid xml")
    with zipfile.ZipFile(path) as z:
        assert io_load._xlsx_zip_shared_strings(z, set()) == {}


def test_xlsx_worksheet_parts_resolves_real_workbook(tmp_path):
    path = _build_job_data_workbook(tmp_path / "parts.xlsx", with_chart=True)
    import zipfile
    with zipfile.ZipFile(path) as z:
        parts = io_load._xlsx_worksheet_parts(z)
    # Exactly the real worksheet parts (Job Data Listing + comments) -- never the chartsheet.
    assert len(parts) == 2
    assert all(p.startswith("xl/worksheets/") for p in parts)


def test_xlsx_worksheet_parts_falls_back_on_malformed_workbook_xml(tmp_path):
    import zipfile
    path = tmp_path / "malformed_workbook_xml.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/workbook.xml", "not xml at all")
        z.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")
    with zipfile.ZipFile(path) as z:
        parts = io_load._xlsx_worksheet_parts(z)
    assert parts == ["xl/worksheets/sheet1.xml"]


def test_sniff_xlsx_data_true_via_inline_strings(tmp_path):
    # Hand-built worksheet XML using inlineStr cells (t="inlineStr") rather than the shared-
    # string table openpyxl's own writer always uses -- exercises the "str"/"inlineStr" cell-type
    # branch sniff_xlsx_data must also recognize.
    import zipfile
    path = tmp_path / "inline_strings.xlsx"
    ns = "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\""
    sheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet {ns}>
 <sheetData>
  <row r="1">
   <c r="A1" t="inlineStr"><is><t>Date</t></is></c>
   <c r="B1" t="inlineStr"><is><t>Pressure</t></is></c>
  </row>
  <row r="2">
   <c r="A2" t="n"><v>44197.5</v></c>
   <c r="B2" t="n"><v>5000</v></c>
  </row>
  <row r="3">
   <c r="A3" t="n"><v>44197.500011574074</v></c>
   <c r="B3" t="n"><v>4999</v></c>
  </row>
 </sheetData>
</worksheet>"""
    wb_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook {ns} xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>"""
    rels_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>"""
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("xl/workbook.xml", wb_xml)
        z.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    assert io_load.sniff_xlsx_data(str(path)) is True


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


# --------------------------------------------------------------------------------------------------
# header detection tightening (FIX 4): reject a key/value preamble row even when its lookahead
# window happens to carry one real number or date
# --------------------------------------------------------------------------------------------------
def test_xlsx_find_header_rejects_keyvalue_preamble_with_one_date(tmp_path):
    # Vermilion "...DFITTEXT.xlsx" shape: "Date, Customer, Well Name/No., Job Type" header
    # candidate whose very next row is (datetime, "text", "text", "text") -- 25% numeric/
    # datetime, correctly rejected in favor of the real header 3 rows further down.
    rows = [
        ["Date", "Customer", "Well Name/No.", "Job Type"],
        [dt.datetime(2021, 6, 4), "Vermillion Energy", "Grand 6-24TH", "DFIT"],
        [None, None, None, None],
        ["Time", "Flow Rate", "Total Flow", "Pressure"],
    ]
    d0 = dt.datetime(2021, 6, 4, 11, 57, 36)
    for i in range(5):
        rows.append([d0 + dt.timedelta(seconds=i), 1.24, 0.02 * i, 51.0 + i])
    found = io_load._xlsx_find_header(rows)
    assert found is not None
    assert found[0] == 3  # the real "Time, Flow Rate, ..." header, not row 0


def test_xlsx_find_header_rejects_keyvalue_preamble_with_one_time_value(tmp_path):
    # GMT "... dfit pump info.xlsx" shape: "Date"/"08/08/2018" (a STRING, not a real date value)
    # key/value row, whose only lookahead evidence is a lone datetime.time next to more label
    # text -- 25% pooled, correctly rejected.
    rows = [
        ["Date", "08/08/2018", None, None],
        ["Time", dt.time(14, 15, 2), None, None],
        ["Company", "Leucrotta", None, None],
        ["Location", "8-22-81-13w6", None, None],
        [None, None, None, None],
        ["date", "time", "Curb Disch", "Remote Press"],
        ["mm/dd/yyyy", None, "MPa", "MPa"],
    ]
    for i in range(5):
        rows.append(["08/08/2018", dt.time(14, 15, 5 + i), -0.1, -25.9])
    found = io_load._xlsx_find_header(rows)
    assert found is not None
    assert found[0] == 5


def test_xlsx_find_header_skips_multiple_leading_label_rows(tmp_path):
    # A real file in the GMT "... dfit pump info.xlsx" family carries TWO purely-textual rows
    # between the header and its first real data row: a descriptive label row (not a recognized
    # units token) and then a genuine units row.
    rows = [
        ["Date/Time", "Status", "COMBINED RATE", "DS QUINT PRESS"],
        [None, None, "Instantaneous value", "Instantaneous value"],
        [None, None, "M3/MIN", "MPa"],
    ]
    for i in range(5):
        rows.append([f"29/12/2014 7:05:{i:02d} PM", "OK", -0.009, -0.05])
    found = io_load._xlsx_find_header(rows)
    assert found is not None
    assert found[0] == 0


def test_xlsx_looks_like_data_rows_rejects_single_column_numeric_preamble(tmp_path):
    # Two numeric values that both sit in the SAME column (a "value" column next to a "label"
    # column) pool to exactly 50% but span only 1 distinct column -- must still be rejected.
    window = [("Test No:", 5), ("TVD:", 8500)]
    assert io_load._xlsx_looks_like_data_rows(window) is False


# --------------------------------------------------------------------------------------------------
# two-row header merge (FIX 4): a blank header cell takes a name from directly above it
# --------------------------------------------------------------------------------------------------
def test_xlsx_merge_two_row_header_fills_blanks_from_row_above():
    rows = [
        ["ignored preamble row"],
        [None, None, "Pump\nBPM", "Annulus\nPSI", "Tbg\nPSI", "dVol\nbbls", None, None],
        ["Date", "Time", None, None, None, None, "FluidIdx", "StageIdx"],
    ]
    merged = io_load._xlsx_merge_two_row_header(rows, 2, rows[2])
    assert merged == ("Date", "Time", "Pump\nBPM", "Annulus\nPSI", "Tbg\nPSI", "dVol\nbbls",
                       "FluidIdx", "StageIdx")


def test_xlsx_merge_two_row_header_never_overwrites_existing_name():
    rows = [["Above", "Above2"], ["Date", "Time"]]
    merged = io_load._xlsx_merge_two_row_header(rows, 1, rows[1])
    assert merged == ("Date", "Time")


def test_xlsx_merge_two_row_header_noop_at_row_zero():
    rows = [["Date", None]]
    assert io_load._xlsx_merge_two_row_header(rows, 0, rows[0]) == rows[0]


def test_load_xlsx_two_row_header_end_to_end(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Ticket No = 12345"])
    ws.append([None, None, "Pump\nBPM", "Annulus\nPSI"])
    ws.append(["Date", "Time", None, None])
    d0 = dt.datetime(2013, 3, 8)
    for i in range(5):
        ws.append([d0, dt.time(11, 32, 49 + i), 0.02 * i, 24.9 - i])
    path = _save(wb, tmp_path / "two_row_header.xlsx")

    td = io_load.load_xlsx(path)
    assert "Pump\nBPM" in td.columns
    assert "Annulus\nPSI" in td.columns
    assert td.n == 5


# --------------------------------------------------------------------------------------------------
# sheet choice by real row count, not ws.max_row (FIX 5)
# --------------------------------------------------------------------------------------------------
def test_load_xlsx_picks_sheet_by_real_rows_not_inflated_max_row(tmp_path):
    wb = openpyxl.Workbook()
    small_but_real = wb.active
    small_but_real.title = "Combined Data"
    small_but_real.append(["Time", "Pressure"])
    d0 = dt.datetime(2015, 2, 4)
    for i in range(20):
        small_but_real.append([d0 + dt.timedelta(seconds=i), 100 + i])

    inflated = wb.create_sheet("PRESSURE (Unadj)")
    inflated.append(["Time", "Pressure"])
    for i in range(5):
        inflated.append([d0 + dt.timedelta(seconds=i), 200 + i])
    # Force openpyxl to report a much larger max_row than the real data occupies -- a formatted
    # but empty cell far below the real rows, mirroring a real corpus workbook's inflated "used
    # range".
    inflated.cell(row=5000, column=1).font = openpyxl.styles.Font(bold=True)
    path = _save(wb, tmp_path / "inflated.xlsx")

    td = io_load.load_xlsx(path)
    assert td.n == 20


def test_load_xlsx_missing_dimension_does_not_rank_data_sheet_below_notes(tmp_path):
    # A worksheet's declared <dimension> is missing entirely (openpyxl then falls back to
    # reporting max_row == 1) -- must not rank a real, larger data sheet below a smaller notes
    # sheet that still has its own (small) <dimension> intact.
    import re
    import zipfile

    wb = openpyxl.Workbook()
    data = wb.active
    data.title = "Data"
    data.append(["Time", "Pressure"])
    d0 = dt.datetime(2020, 1, 1)
    for i in range(20):
        data.append([d0 + dt.timedelta(seconds=i), 100 + i])

    notes = wb.create_sheet("Notes")
    notes.append(["Comment"])
    notes.append(["a small notes sheet"])
    path = _save(wb, tmp_path / "no_dimension.xlsx")

    # Strip the <dimension .../> element from the "Data" sheet's part (sheet1.xml, since it's
    # the first/active sheet) to simulate a missing dimension.
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        contents = {n: z.read(n) for n in names}
    sheet1 = contents["xl/worksheets/sheet1.xml"].decode("utf-8")
    stripped = re.sub(r"<dimension[^/]*/>", "", sheet1)
    assert stripped != sheet1  # sanity: the substitution actually matched something
    contents["xl/worksheets/sheet1.xml"] = stripped.encode("utf-8")
    with zipfile.ZipFile(path, "w") as z:
        for n, data_bytes in contents.items():
            z.writestr(n, data_bytes)

    td = io_load.load_xlsx(path)
    assert td.n == 20
    assert "Pressure" in "".join(td.columns)


def test_load_xlsx_stale_undersized_dimension_does_not_rank_data_sheet_below_notes(tmp_path):
    # A worksheet's declared <dimension> UNDER-reports its real extent (e.g. "A1:B10" left over
    # from an earlier, smaller version of the sheet) rather than being missing outright -- must
    # still not rank a real, much larger data sheet below a smaller notes sheet.
    import re
    import zipfile

    wb = openpyxl.Workbook()
    data = wb.active
    data.title = "Data"
    data.append(["Date Time", "Pressure (psi)"])
    d0 = dt.datetime(2024, 1, 1, 12)
    for i in range(500):
        data.append([d0 + dt.timedelta(seconds=i), 5000 - i])

    notes = wb.create_sheet("Notes")
    notes.append(["Time", "Event"])
    for i in range(50):
        notes.append([d0 + dt.timedelta(minutes=i), 1.0])
    path = _save(wb, tmp_path / "stale_dimension.xlsx")

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        contents = {n: z.read(n) for n in names}
    sheet1 = contents["xl/worksheets/sheet1.xml"].decode("utf-8")
    understated = re.sub(r'<dimension ref="[^"]*"\s*/>', '<dimension ref="A1:B10"/>', sheet1)
    assert understated != sheet1
    contents["xl/worksheets/sheet1.xml"] = understated.encode("utf-8")
    with zipfile.ZipFile(path, "w") as z:
        for n, data_bytes in contents.items():
            z.writestr(n, data_bytes)

    td = io_load.load_xlsx(path)
    assert td.n == 500


def test_xlsx_count_data_rows_stops_after_consecutive_blanks(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Time", "Pressure"])
    for i in range(10):
        ws.append([i, i])
    # A stray far-away formatted cell inflates max_row without adding any real data row.
    ws.cell(row=5000, column=1).font = openpyxl.styles.Font(bold=True)
    path = _save(wb, tmp_path / "count_rows.xlsx")

    wb2 = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws2 = wb2.worksheets[0]
    # data_start_row=1 (0-based) -- past the "Time, Pressure" header row itself.
    assert io_load._xlsx_count_data_rows(ws2, 1) == 10
    wb2.close()


# --------------------------------------------------------------------------------------------------
# sub-second precision (FIX 7)
# --------------------------------------------------------------------------------------------------
def test_xlsx_cell_to_value_keeps_fractional_seconds():
    cell = dt.datetime(2020, 1, 2, 3, 4, 5, 123456)
    assert io_load._xlsx_cell_to_value(cell, date_only=False) == "01/02/2020 03:04:05.123456"


def test_xlsx_cell_to_value_time_keeps_fractional_seconds():
    cell = dt.time(3, 4, 5, 123456)
    assert io_load._xlsx_cell_to_value(cell, date_only=False) == "03:04:05.123456"


def test_load_xlsx_millisecond_stamped_workbook_end_to_end(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    d0 = dt.datetime(2020, 1, 1, 0, 0, 0, 0)
    for i in range(5):
        ws.append([d0 + dt.timedelta(milliseconds=250 * i), 5000 - i])
    path = _save(wb, tmp_path / "ms_stamped.xlsx")

    td = io_load.load_xlsx(path)
    assert td.n == 5
    assert td.t_s[-1] == pytest.approx(1.0)
    assert list(td.t_s) == sorted(td.t_s)


def test_load_xlsx_prefers_sheet_with_usable_time_base(tmp_path):
    # The bigger "Original Data" sheet has a midnight-only datetime column and no time column
    # (one distinct timestamp); the smaller "Cleaned Data" sheet has a real Date + Time pair.
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Original Data"
    ws.append(["Date/Time", "Pressure"])
    for i in range(40):
        ws.append([dt.datetime(2023, 1, 26), 5000 - i])
    ws2 = wb.create_sheet("Cleaned Data")
    ws2.append(["Date", "Time", "Pressure"])
    for i in range(30):
        ws2.append([dt.datetime(2023, 1, 26), dt.time(9, 0, i), 5000 - i])
    td = io_load.load_xlsx(_save(wb, tmp_path / "two_sheets.xlsx"))
    assert td.n == 30
    assert td.t_s[-1] - td.t_s[0] == pytest.approx(29.0)


def test_load_xlsx_zero_span_everywhere_raises(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date/Time", "Pressure"])
    for i in range(20):
        ws.append([dt.datetime(2023, 1, 26), 5000 - i])
    with pytest.raises(ValueError, match="no sheet with a usable time base"):
        io_load.load_xlsx(_save(wb, tmp_path / "flat.xlsx"))


def test_load_xlsx_data_read_stops_at_blank_run_before_far_styled_cell(tmp_path):
    from openpyxl.styles import PatternFill
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    for i in range(25):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    ws["A1048000"].fill = PatternFill("solid", fgColor="FFFF00")
    path = _save(wb, tmp_path / "far_styled.xlsx")
    wb2 = io_load._xlsx_open(path)
    try:
        ws2 = wb2.worksheets[0]
        scan = io_load._xlsx_scan_sheet(ws2)
        df, _ = io_load._xlsx_read_sheet_frame(ws2, scan)
    finally:
        wb2.close()
    assert len(df) == 25
    assert io_load.load_xlsx(path).n == 25


def test_load_xlsx_warns_when_blank_gap_stop_hides_later_rows(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    for i in range(25):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    for k in range(3):
        ws.cell(row=3000 + k, column=1, value=dt.datetime(2020, 1, 1, 1, 0, k))
        ws.cell(row=3000 + k, column=2, value=4000 - k)
    td = io_load.load_xlsx(_save(wb, tmp_path / "gap_hides.xlsx"))
    assert td.n == 25
    assert any(">=1000 blank rows" in w and "3 later rows" in w for w in td.load_warnings)


def test_load_xlsx_no_gap_warning_for_styled_only_far_cell(tmp_path):
    from openpyxl.styles import PatternFill
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    for i in range(25):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    ws["A1048000"].fill = PatternFill("solid", fgColor="FFFF00")
    td = io_load.load_xlsx(_save(wb, tmp_path / "styled_only.xlsx"))
    assert not any("blank rows" in w for w in td.load_warnings)


def test_xlsx_zip_count_matches_openpyxl_count(tmp_path):
    import zipfile
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    for i in range(50):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    ws["A3000"] = "stray"  # beyond the 1000-blank stop: not counted by either
    path = _save(wb, tmp_path / "count.xlsx")
    wb2 = io_load._xlsx_open(path)
    try:
        n_opx = io_load._xlsx_count_data_rows(wb2.worksheets[0], 1)
    finally:
        wb2.close()
    with zipfile.ZipFile(path) as z:
        part = io_load._xlsx_worksheet_parts(z)[0]
        n_zip = io_load._xlsx_zip_count_data_rows(z, part, 1)
    assert n_zip == n_opx == 50


def test_xlsx_zip_count_handles_gaps_blanks_and_inline_blocks(tmp_path):
    import zipfile
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    r = 2
    for i in range(30):
        ws.cell(row=r, column=1, value=dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i))
        ws.cell(row=r, column=2, value=5000 - i)
        r += 1
        if i in (7, 8, 20):  # skipped (omitted) rows in the middle
            r += 2
    ws.cell(row=r + 1, column=2).fill = openpyxl.styles.PatternFill("solid", fgColor="FFFF00")
    path = _save(wb, tmp_path / "gaps.xlsx")
    wb2 = io_load._xlsx_open(path)
    try:
        n_opx = io_load._xlsx_count_data_rows(wb2.worksheets[0], 1)
    finally:
        wb2.close()
    with zipfile.ZipFile(path) as z:
        part = io_load._xlsx_worksheet_parts(z)[0]
        n_zip = io_load._xlsx_zip_count_data_rows(z, part, 1)
        n_et = io_load._xlsx_zip_count_data_rows_et(z, part, 1)
    assert n_zip == n_opx == n_et == 30


def test_xlsx_zip_count_fast_path_across_small_blocks(tmp_path, monkeypatch):
    import zipfile
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Date", "Pressure"])
    for i in range(1500):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    for i in range(1500, 1510):  # blank rows then more data, inside later blocks
        ws.append([None, None])
    for i in range(1510, 1600):
        ws.append([dt.datetime(2020, 1, 1) + dt.timedelta(seconds=i), 5000 - i])
    path = _save(wb, tmp_path / "blocks.xlsx")
    monkeypatch.setattr(io_load, "_XLSX_COUNT_BLOCK_BYTES", 4096)
    fast_calls = []
    orig = io_load._xlsx_count_region_fast
    monkeypatch.setattr(io_load, "_xlsx_count_region_fast",
                        lambda region, st: fast_calls.append(orig(region, st)) or fast_calls[-1])
    with zipfile.ZipFile(path) as z:
        part = io_load._xlsx_worksheet_parts(z)[0]
        n_zip = io_load._xlsx_zip_count_data_rows(z, part, 1)
        n_et = io_load._xlsx_zip_count_data_rows_et(z, part, 1)
    assert n_zip == n_et == 1590
    assert any(fast_calls) and not all(fast_calls)  # both paths exercised
