"""Tests for the CSV loader (``load_csv``, ``parse_datetime``, ``suggest_channels``) in
dfit_tool/io_load.py.

All fixtures are synthetic and written under ``tmp_path`` -- no test here reads anything under
``C:\\DFIT Data``. Each fixture reproduces, in miniature, the exact column-name and value shapes
of a real corpus file (named in the fix's docstring/comment) so a regression in the loader shows
up here rather than only in a real load.
"""

import numpy as np
import pandas as pd
import pytest

from dfit_tool import io_load

# --------------------------------------------------------------------------------------------------
# FIX A -- Date + Time columns, day-first dates
# --------------------------------------------------------------------------------------------------
def test_suggest_channels_finds_companion_time_column():
    # Strathcona shape: "Date,Time,serialNumber,sample,CASING Pressure (KPAg) ,CASING Temp ..."
    cols = ["Date", "Time", "serialNumber", "sample", "CASING Pressure (KPAg)", "CASING Temp (Celsius)"]
    guess = io_load.suggest_channels(cols)
    assert guess["datetime"] == "Date"
    assert guess["time"] == "Time"


def test_suggest_channels_time_is_none_for_date_slash_time():
    # DEFECT 5: the plain 3-column shape ["Date/Time", "TZ", "Pressure(psia)"] this test used to
    # use is vacuous -- it has no separate column whose bare name is "time" at all, so it can't
    # actually exercise the "and 'time' not in lc[datetime_col]" guard (deleting that clause left
    # this test green). A companion "Time" column is the discriminating input: "Date/Time"
    # already contains "time" in its own name, so it must win over the companion regardless.
    cols = ["Date/Time", "Time", "PRESS"]
    guess = io_load.suggest_channels(cols)
    assert guess["datetime"] == "Date/Time"
    assert guess["time"] is None


def test_suggest_channels_time_is_none_for_timestamp_mst():
    # DEFECT 5: same fix as above -- add a companion "Time" column so the guard is actually
    # exercised ("Timestamp (MST)" contains "time" in its own name too).
    cols = ["Timestamp (MST)", "Time", "PRESS"]
    guess = io_load.suggest_channels(cols)
    assert guess["datetime"] == "Timestamp (MST)"
    assert guess["time"] is None


def test_suggest_channels_time_is_none_for_datetime():
    # DEFECT 5: the third discriminating case -- "DateTime" also contains "time" in its own
    # name, so a companion "Time" column must not be adopted.
    cols = ["DateTime", "Time", "PRESS"]
    guess = io_load.suggest_channels(cols)
    assert guess["datetime"] == "DateTime"
    assert guess["time"] is None


def test_suggest_channels_time_is_none_without_a_time_column():
    cols = ["Date", "serialNumber", "sample", "Pressure (psi)"]
    guess = io_load.suggest_channels(cols)
    assert guess["datetime"] == "Date"
    assert guess["time"] is None


def test_suggest_channels_time_column_with_unit_suffix_still_matches():
    cols = ["Date", "Time (s)", "Pressure (psi)"]
    guess = io_load.suggest_channels(cols)
    assert guess["time"] == "Time (s)"


def test_dayfirst_hint_true_on_day_over_12():
    s = pd.Series(["9/8/2022", "13/8/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is True


# DEFECT 3, rule 1: max_first > 12 and max_second <= 12 -> day-first. A separate case from
# test_dayfirst_hint_true_on_day_over_12 above -- that fixture's second component happens to be
# constant, which also (coincidentally) satisfies rule 4, so deleting rule 1 alone wouldn't be
# caught there. This one varies the second component too, isolating rule 1.
def test_dayfirst_hint_rule1_day_over_12_with_varying_second(tmp_path):
    s = pd.Series(["13/8/2022", "20/9/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is True


# DEFECT 3, rule 2: max_second > 12 and max_first <= 12 -> month-first (the symmetric proof).
# The second component here is constant at 13 (> 12, impossible as a month) while the first
# varies 8/9/10 -- the same shape rule 4 looks for, but rule 2 has PROOF (month 13 doesn't exist)
# that must win: this is what catches a rule 2 that's deleted or never checked, since without it
# rule 4 would wrongly fire True (day=first varies, "month"=13 constant -- nonsense).
def test_dayfirst_hint_rule2_month_over_12_overrides_rule4_shape():
    s = pd.Series(["8/13/2022", "9/13/2022", "10/13/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


# DEFECT 3, rule 3: year constant, first component constant, second varies (>=2) -> month-first.
# A US-style file inside one month: month=5 constant, day increments 1/2/3.
#
# GAP 4 (nit): this pins rule 3's OUTCOME (month-first for a constant-first/varying-second
# fixture inside one month) but does NOT independently distinguish rule 3 from the rule 5
# default -- both return False for this input, so deleting rule 3 entirely leaves this test
# green (it falls through to rule 5, which happens to agree). Rule 3 exists for spec parity
# with rule 4's symmetric day-first case and is not otherwise observable from outside
# _dayfirst_hint; that is a known, accepted gap, not something this test can close.
def test_dayfirst_hint_rule3_constant_month_varying_day():
    s = pd.Series(["5/1/2022", "5/2/2022", "5/3/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


# DEFECT 3, rule 4: year constant, second component constant, first varies (>=2) -> day-first.
# The Strathcona `-rt.csv` case itself: second (month) constant at 8, first (day) increments
# 9/10/11/12 -- no component exceeds 12, so rules 1-2 give no evidence either way.
def test_dayfirst_hint_rule4_constant_month_varying_day_strathcona_shape():
    s = pd.Series(["9/8/2022", "10/8/2022", "11/8/2022", "12/8/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is True


# DEFECT 3: rules 3-4 require the year to be CONSTANT. The same "second component constant,
# first varies" shape as rule 4 above, but with the year also varying, must NOT trigger rule 4 --
# it falls through to the rule 5 default (month-first) instead.
def test_dayfirst_hint_rule4_does_not_apply_when_year_varies():
    s = pd.Series(["9/8/2022", "10/8/2023", "11/8/2024", "12/8/2025"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


# DEFECT 3, rule 5: genuinely no evidence either way -- neither component is constant, neither
# exceeds 12 -- so today's month-first default is kept (a record crossing a month boundary).
# This is also the renamed/updated version of the old "no evidence" test: under the new rule 4,
# a *constant* second component with a *varying* first (this fixture's old value) now returns
# True (see test_dayfirst_hint_rule4_constant_month_varying_day_strathcona_shape above), so this
# case is deliberately reshaped to have both components vary instead.
def test_dayfirst_hint_rule5_default_when_both_components_vary():
    s = pd.Series(["8/9/2022", "9/10/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


def test_dayfirst_hint_false_for_iso_dates():
    s = pd.Series(["2024-12-06 12:39:10", "2024-12-05 08:00:00"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


def test_dayfirst_hint_false_when_both_components_exceed_12():
    s = pd.Series(["13/13/2022", "20/25/2022"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


def test_dayfirst_hint_false_with_no_matching_values():
    s = pd.Series(["not a date", "2024-12-06"], dtype="string")
    assert io_load._dayfirst_hint(s) is False


def test_normalize_ms_colon_rewrites_colon_milliseconds():
    # DEFECT 1b/d: pins _normalize_ms_colon directly. Only the "HH:MM:SS:mmm" shape (a colon
    # where a decimal point belongs) is rewritten; anything else -- an ordinary "HH:MM:SS", an
    # already-correct decimal form, a coarser single-digit-hour form, and (mutation guard for
    # the trailing $ anchor) a colon followed by MORE than 3 digits -- passes through untouched.
    s = pd.Series(
        ["15:58:17:647", "15:58:17", "15:58:17.647", "8:23:17", "15:58:17:6478"],
        dtype="string",
    )
    result = io_load._normalize_ms_colon(s)
    assert result.tolist() == [
        "15:58:17.647",
        "15:58:17",
        "15:58:17.647",
        "8:23:17",
        "15:58:17:6478",
    ]


def test_load_csv_colon_milliseconds_end_to_end(tmp_path):
    # DEFECT 1a: the Lucero Tahu shape -- "Date,Time,Marker,Combined Flow Rate,Combined Flow
    # Total,Max Pressure" with Time shaped "HH:MM:SS:mmm". Every row must parse (not just the
    # date-only fallback), and t_s must be strictly increasing sub-second, not the degenerate
    # all-zero result a same-day date-only parse would give.
    rows = [
        "6/30/2023,15:58:17:647,,0,0,278",
        "6/30/2023,15:58:17:897,,0,0,277",
        "6/30/2023,15:58:18:147,,0,0,276",
        "6/30/2023,15:58:18:397,,0,0,275",
    ]
    p = tmp_path / "lucero.csv"
    p.write_text(
        "Date,Time,Marker,Combined Flow Rate,Combined Flow Total,Max Pressure\n"
        + "\n".join(rows) + "\n"
    )

    td = io_load.load_csv(str(p))
    assert td.n == 4
    dt = td.df["Date"]
    assert dt.notna().sum() == 4
    assert np.all(np.diff(td.t_s) > 0)
    np.testing.assert_allclose(td.t_s, [0.0, 0.25, 0.5, 0.75])


def test_load_csv_garbage_time_falls_back_to_date_only(tmp_path):
    # DEFECT 1a: a companion Time column that is unparseable junk (not even the colon-ms shape
    # _normalize_ms_colon can recover) must not regress a file that was openable on the Date
    # column alone -- joined parses 0 valid, date-only parses 4 (collapsed to one day), so the
    # date-only fallback must be kept and the file must still load rather than raising.
    rows = [
        "6/30/2023,banana,,0,0,278",
        "6/30/2023,not-a-time,,0,0,277",
        "6/30/2023,xyz,,0,0,276",
        "6/30/2023,N/A,,0,0,275",
    ]
    p = tmp_path / "lucero_garbage.csv"
    p.write_text(
        "Date,Time,Marker,Combined Flow Rate,Combined Flow Total,Max Pressure\n"
        + "\n".join(rows) + "\n"
    )

    td = io_load.load_csv(str(p))
    assert td.n == 4
    dt = td.df["Date"]
    assert dt.notna().sum() == 4
    # Date-only resolution: all 4 rows collapse onto the same day.
    np.testing.assert_allclose(td.t_s, [0.0, 0.0, 0.0, 0.0])


def test_load_csv_keeps_joined_when_it_parses_more_than_date_only(tmp_path):
    # DEFECT 1a, the other direction: joined must be kept (not wrongly abandoned for date-only)
    # when it parses MORE valid timestamps. A literal single-space Date field (not a truly empty
    # CSV field, which pandas would read as NaN and which then propagates through the join as a
    # missing value) survives read_csv as the string " " -- non-empty, so it is not dropped --
    # and joins with a bare time-of-day into a parseable "<time>"-only string (dateutil defaults
    # the missing date to today), while the Date column alone is blank and unparseable on those
    # rows. 4 of 5 rows only parse when joined; only 1 parses date-only.
    lines = [
        "Date,Time,Pressure(psi)",
        " ,8:23:17,100",
        " ,8:23:29,101",
        " ,8:23:40,102",
        " ,8:23:50,103",
        "9/8/2022,8:24:00,104",
    ]
    p = tmp_path / "mostly_blank_date.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.n == 5
    dt = td.df["Date"]
    assert dt.notna().sum() == 5
    assert dt.iloc[-1] == pd.Timestamp("2022-09-08 08:24:00")


def test_load_csv_date_time_dayfirst_end_to_end(tmp_path):
    # Reproduces the Strathcona shape: separate Date + Time columns, day-first dates, unpadded
    # time-of-day. Real file: 100-01-28-061-03W6-rt Aug15.csv.
    rows = [
        "9/8/2022,8:23:17,63473,1,-14.565991,12.896053",
        "9/8/2022,8:23:29,63473,1,-14.434333,12.902447",
        "15/8/2022,8:41:38,63473,2,19128.900000,18.244000",
        "15/8/2022,8:41:50,63473,2,19128.951172,18.244686",
    ]
    p = tmp_path / "strathcona.csv"
    p.write_text(
        "Date,Time,serialNumber,sample,CASING Pressure (KPAg) ,CASING Temp (Celsius) \n"
        + "\n".join(rows) + "\n"
    )

    td = io_load.load_csv(str(p))
    assert td.n == 4
    assert td.datetime_col == "Date"
    dt = td.df["Date"]
    assert dt.notna().sum() == 4
    assert dt.iloc[0] == pd.Timestamp("2022-08-09 08:23:17")
    assert dt.iloc[-1] == pd.Timestamp("2022-08-15 08:41:50")
    # Span is time-of-day-accurate (~6 days), not the whole-days-only span a Date-only parse
    # would give (which would report exactly 6 days with the intra-day spacing all lost).
    span_s = td.t_s[-1] - td.t_s[0]
    expected_span_s = (pd.Timestamp("2022-08-15 08:41:50") - pd.Timestamp("2022-08-09 08:23:17")).total_seconds()
    assert span_s == pytest.approx(expected_span_s)


def test_load_csv_date_time_dayfirst_rt_shape_short_test_end_to_end(tmp_path):
    # DEFECT 3: the direct sibling of the fixture above that the OLD hint missed -- Real file:
    # 100-01-28-061-03W6-rt.csv. Its Date values are exactly {9/8/2022, 10/8/2022, 11/8/2022,
    # 12/8/2022} -- a 3-day test, no day-of-month evidence above 12 anywhere, but the constant
    # second component (month=8) and varying first (day 9-12) is rule 4's Strathcona case. Old
    # behavior parsed this as Sep 8 -> Dec 8 (~91 days); it must now read as August 9-12 (~3 days).
    rows = [
        "9/8/2022,8:23:17,63473,1,100.0",
        "10/8/2022,8:23:17,63473,1,100.0",
        "11/8/2022,8:23:17,63473,1,100.0",
        "12/8/2022,8:23:17,63473,1,100.0",
    ]
    p = tmp_path / "strathcona_rt.csv"
    p.write_text(
        "Date,Time,serialNumber,sample,CASING Pressure (KPAg) \n" + "\n".join(rows) + "\n"
    )

    td = io_load.load_csv(str(p))
    dt = td.df["Date"]
    assert dt.notna().sum() == 4
    assert list(dt.dt.month) == [8, 8, 8, 8]
    assert list(dt.dt.day) == [9, 10, 11, 12]
    span_days = (td.t_s[-1] - td.t_s[0]) / 86400.0
    assert span_days == pytest.approx(3.0, abs=0.01)


# --------------------------------------------------------------------------------------------------
# FIX B -- elapsed-time-column fallback
# --------------------------------------------------------------------------------------------------
def test_find_elapsed_column_accepts_delta_hrs():
    df = pd.DataFrame({"Delta(Hrs)": [0.0, 1.0, 2.0]})
    found = io_load._find_elapsed_column(df)
    assert found is not None
    name, secs = found
    assert name == "Delta(Hrs)"
    np.testing.assert_allclose(secs, [0.0, 3600.0, 7200.0])


def test_find_elapsed_column_accepts_minutes_and_seconds():
    df_min = pd.DataFrame({"Delta(min)": [0.0, 1.0]})
    name, secs = io_load._find_elapsed_column(df_min)
    assert name == "Delta(min)"
    np.testing.assert_allclose(secs, [0.0, 60.0])

    df_sec = pd.DataFrame({"Elapsed(sec)": [0.0, 30.0]})
    name, secs = io_load._find_elapsed_column(df_sec)
    assert name == "Elapsed(sec)"
    np.testing.assert_allclose(secs, [0.0, 30.0])


def test_find_elapsed_column_rejects_unitless_delta():
    df = pd.DataFrame({"Delta": [0.0, 1.0, 2.0]})
    assert io_load._find_elapsed_column(df) is None


def test_find_elapsed_column_rejects_physical_channel_names():
    df = pd.DataFrame(
        {
            "Flow Rate(m3/min)": [0.0, 1.0, 2.0],
            "Delta Pressure(psi)": [0.0, 1.0, 2.0],
        }
    )
    assert io_load._find_elapsed_column(df) is None


def test_find_elapsed_column_rejects_non_monotonic():
    df = pd.DataFrame({"Delta(Hrs)": [0.0, 2.0, 1.0, 3.0]})
    assert io_load._find_elapsed_column(df) is None


def test_find_elapsed_column_rejects_zero_span():
    df = pd.DataFrame({"Delta(Hrs)": [1.0, 1.0, 1.0]})
    assert io_load._find_elapsed_column(df) is None


def test_load_csv_falls_back_to_elapsed_column(tmp_path):
    # Reproduces the Goodnight shape: Excel-mangled "MM:SS.0" Date/Time, but a clean Delta(Hrs).
    lines = [
        "Date/Time, TZ,Delta(Hrs), Casing 1 pressure (psi)",
        "26:13.0,CDT,0,11.297",
        "26:14.0,CDT,0.0002778,11.313",
        "42:27.0,CDT,291.2705556,1393.0",
    ]
    p = tmp_path / "goodnight.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "DateTime"
    assert td.n == 3
    assert np.all(np.diff(td.t_s) >= 0)
    assert td.t_s[0] == 0.0
    assert td.t_s[-1] / 3600 == pytest.approx(291.2705556, abs=1e-3)


def test_load_csv_elapsed_column_rebases_to_zero_when_it_does_not_start_there(tmp_path):
    # DEFECT 4: True Oil\Abra Data's spotter files measure t_s[0] = 10.00008 / 1.00008 -- the
    # elapsed column itself doesn't start at 0 (logging started before the elapsed counter did).
    # TestData.t_s is documented as "elapsed seconds from first sample", same as the datetime
    # path's own elapsed_seconds() guarantees, so this fallback must rebase too.
    lines = [
        "Date/Time, TZ,Delta(Hrs), Casing 1 pressure (psi)",
        "26:13.0,CDT,5.0,11.297",
        "26:14.0,CDT,5.5,11.313",
        "42:27.0,CDT,6.0,1393.0",
    ]
    p = tmp_path / "goodnight_offset.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "DateTime"
    assert td.t_s[0] == 0.0
    np.testing.assert_allclose(td.t_s, [0.0, 1800.0, 3600.0])


def test_load_csv_elapsed_column_rebase_leaves_interior_nan_alone(tmp_path):
    # DEFECT 4: an interior blank cell must stay NaN (missing elapsed data mirrors a NaT in the
    # datetime path -- neither is invented a value), while the surrounding finite values still
    # rebase relative to the first FINITE one, not index 0.
    lines = [
        "Date/Time, TZ,Delta(Hrs), Casing 1 pressure (psi)",
        "26:13.0,CDT,5.0,11.297",
        "26:14.0,CDT,,11.313",
        "26:15.0,CDT,5.5,11.320",
        "42:27.0,CDT,6.0,1393.0",
    ]
    p = tmp_path / "goodnight_gap.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "DateTime"
    assert td.t_s[0] == 0.0
    assert np.isnan(td.t_s[1])
    np.testing.assert_allclose(
        [td.t_s[0], td.t_s[2], td.t_s[3]], [0.0, 1800.0, 3600.0]
    )


def test_load_csv_elapsed_column_rebase_uses_first_finite_value_not_index_zero(tmp_path):
    # GAP 3 (should-fix): "the rebase uses the first FINITE elapsed value" is unpinned -- both
    # existing D4 fixtures above (offset and interior-gap) start finite at index 0, so
    # `secs - finite[0]` and the wrong `secs - secs[0]` agree on them. A LEADING blank in
    # Delta(Hrs) is the discriminating shape: secs[0] is NaN, so `secs - secs[0]` would poison
    # every value to NaN, while the correct rebase against finite[0] leaves the later values
    # intact and only the leading gap itself as NaN.
    lines = [
        "Date/Time, TZ,Delta(Hrs), Casing 1 pressure (psi)",
        "26:13.0,CDT,,11.297",
        "26:14.0,CDT,5.0,11.313",
        "26:15.0,CDT,5.5,11.320",
        "42:27.0,CDT,6.0,1393.0",
    ]
    p = tmp_path / "goodnight_leading_gap.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "DateTime"
    assert np.isnan(td.t_s[0])
    assert td.t_s[1] == 0.0
    assert td.t_s[2] == 1800.0
    assert td.t_s[3] == 3600.0


def test_load_csv_synthetic_datetime_column_avoids_collision_with_real_column(tmp_path):
    # GAP 5 (nit): the `while synth_col in df.columns` collision loop in FIX B's synthetic-
    # column naming is unpinned. When the real (unusable) datetime column is itself literally
    # named "DateTime" -- not "Date/Time" or "Datetime" -- the naive first guess ("DateTime")
    # would collide with it, so the loop must fall through to "DateTime (2)".
    lines = [
        "DateTime, TZ,Delta(Hrs), Casing 1 pressure (psi)",
        "26:13.0,CDT,0,11.297",
        "26:14.0,CDT,0.0002778,11.313",
        "42:27.0,CDT,291.2705556,1393.0",
    ]
    p = tmp_path / "goodnight_datetime_named.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "DateTime (2)"
    # The original "DateTime" column (the unusable one) must be left untouched, not overwritten
    # by the synthetic one.
    assert list(td.df["DateTime"]) == ["26:13.0", "26:14.0", "42:27.0"]
    assert td.t_s[0] == 0.0
    assert td.t_s[-1] / 3600 == pytest.approx(291.2705556, abs=1e-3)


def test_load_csv_keeps_good_datetime_col_even_with_delta_hrs_present(tmp_path):
    # Pins the guard that stops FIX B from firing on a file like the Vesta one, which has BOTH
    # a good Datetime column and a Delta(Hrs) column.
    lines = [
        "Datetime,TZ,Delta(Hrs),Flow Rate(m3/min)",
        "2022-10-31 10:35:43.000,MDT,0.0000000,0.000",
        "2022-10-31 10:36:43.000,MDT,0.0166667,0.010",
        "2022-10-31 10:37:43.000,MDT,0.0333333,0.020",
    ]
    p = tmp_path / "vesta.csv"
    p.write_text("\n".join(lines) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Datetime"
    assert "DateTime" not in td.df.columns


# --------------------------------------------------------------------------------------------------
# FIX C -- preamble skipping
# --------------------------------------------------------------------------------------------------
def test_detect_header_skiprows_three_column_shape(tmp_path):
    p = tmp_path / "preamble3.csv"
    p.write_text(
        "Job ID: 9942,Spotter: 1115093\n"
        "Row(s): 4\n"
        "Date/Time,TZ,Pressure(psia)\n"
        "2018-03-29 20:58:24,CDT,16.9\n"
        "2018-03-29 20:58:25,CDT,16.9\n"
    )
    assert io_load._detect_header_skiprows(str(p)) == 2


def test_detect_header_skiprows_five_column_shape(tmp_path):
    p = tmp_path / "preamble5.csv"
    p.write_text(
        "Job ID: 13751,Spotter: 1115471\n"
        "Row(s): 3\n"
        "Datetime,TZ,Delta(Hrs),Flow Rate(m3/min),Totaliser(m3)\n"
        "2022-10-31 10:35:43.000,MDT,0.0000000,0.000,0.044\n"
        "2022-10-31 10:36:43.000,MDT,0.0166667,0.010,0.045\n"
    )
    assert io_load._detect_header_skiprows(str(p)) == 2


def test_detect_header_skiprows_zero_for_ordinary_csv(tmp_path):
    p = tmp_path / "ordinary.csv"
    p.write_text("Date/Time,TZ,Pressure(psia)\n2018-03-29 20:58:24,CDT,16.9\n")
    assert io_load._detect_header_skiprows(str(p)) == 0


def test_detect_header_skiprows_counts_quoted_comma_as_one_field(tmp_path):
    # The header and data rows each have a quoted field containing a literal comma. A naive
    # comma-split (not the stdlib csv module) would see that as a 4th field and never find a
    # consistent modal width of 3, so this pins that the csv module's quoting is honored.
    p = tmp_path / "quoted.csv"
    p.write_text(
        "Job ID: 9999\n"
        "Row(s): 2\n"
        '"Name","Value, with comma","Other"\n'
        '"a","b, c","d"\n'
        '"e","f, g","h"\n'
    )
    assert io_load._detect_header_skiprows(str(p)) == 2


def test_detect_header_skiprows_skips_all_empty_modal_width_line(tmp_path):
    # GAP 2 (should-fix): the "at least one non-empty field" clause is unpinned. The ",,\n" line
    # below is 3 fields wide -- the modal width, same as the header and data rows -- but every
    # field is empty, so it must be skipped rather than mistaken for the header (an all-empty
    # row is never numeric, so DEFECT 2's numeric-field check alone would not reject it).
    p = tmp_path / "empty_modal_line.csv"
    p.write_text(
        "Job ID: 9942,Spotter: 1115093\n"
        "Row(s): 4\n"
        ",,\n"
        "Date/Time,TZ,Pressure(psia)\n"
        "2018-03-29 20:58:24,CDT,16.9\n"
        "2018-03-29 20:58:25,CDT,17.0\n"
    )
    assert io_load._detect_header_skiprows(str(p)) == 3
    td = io_load.load_csv(str(p))
    assert list(td.df.columns) == ["Date/Time", "TZ", "Pressure(psia)"]


def test_load_csv_preamble_end_to_end(tmp_path):
    p = tmp_path / "caprito_shape.csv"
    p.write_text(
        "Job ID: 9942,Spotter: 1115093\n"
        "Row(s): 3\n"
        "Date/Time,TZ,Pressure(psia)\n"
        "2018-03-29 20:58:24,CDT,16.9\n"
        "2018-03-29 20:58:25,CDT,17.0\n"
        "2018-03-29 20:58:26,CDT,17.1\n"
    )
    td = io_load.load_csv(str(p))
    assert td.n == 3
    assert list(td.df.columns) == ["Date/Time", "TZ", "Pressure(psia)"]


def test_load_csv_preamble_does_not_collapse_to_row_count_column(tmp_path):
    # Anti-regression: a naive "retry skiprows=1,2,3... take the first that parses" loop would
    # succeed at skiprows=1 here, producing a single-column frame named "Row(s): 5". Pin that
    # this does NOT happen -- the full 5-column table must load instead.
    p = tmp_path / "vesta_shape.csv"
    p.write_text(
        "Job ID: 13751,Spotter: 1115471\n"
        "Row(s): 5\n"
        "Datetime,TZ,Delta(Hrs),Flow Rate(m3/min),Totaliser(m3)\n"
        "2022-10-31 10:35:43.000,MDT,0.0000000,0.000,0.044\n"
        "2022-10-31 10:36:43.000,MDT,0.0166667,0.010,0.045\n"
        "2022-10-31 10:37:43.000,MDT,0.0333333,0.020,0.046\n"
    )
    td = io_load.load_csv(str(p))
    assert list(td.df.columns) == ["Datetime", "TZ", "Delta(Hrs)", "Flow Rate(m3/min)", "Totaliser(m3)"]
    assert td.n == 3


def test_load_csv_ragged_unrecoverable_still_raises(tmp_path):
    # A genuinely malformed file: every line has a different field count (no repeats at all),
    # so there's no real modal width to recover -- detection falls back to the header's own
    # (tied, first-inserted) count, resolves to skiprows=0, and the original ParserError from
    # the first read_csv attempt must propagate unchanged.
    p = tmp_path / "ragged.csv"
    p.write_text(
        "a,b,c\n"
        "1,2,3,4\n"
        "1,2,3,4,5\n"
        "1,2,3,4,5,6\n"
        "1,2,3,4,5,6,7\n"
    )
    with pytest.raises(pd.errors.ParserError):
        io_load.load_csv(str(p))


# --------------------------------------------------------------------------------------------------
# DEFECT 2 -- a narrow real header vs. a wider ragged data row at the same modal width
# --------------------------------------------------------------------------------------------------
def test_detect_header_skiprows_rejects_numeric_data_row_razor_shape(tmp_path):
    # The Razor shape: a real 2-field header, a 2-field units row, then 3-field data rows (a
    # trailing comma). The 3-field width is modal (most data rows share it), but every 3-field
    # line is a data row with a numeric first field -- none qualifies as a header, so detection
    # must return 0 and let the original ParserError propagate rather than silently adopting the
    # first data row as the header.
    p = tmp_path / "razor.csv"
    p.write_text(
        "Time,Job Time\n"
        "(min) ,(date time)\n"
        "400.00000,12/16/2016 12:40:01 PM,\n"
        "420.00000,12/16/2016 12:45:01 PM,\n"
        "440.00000,12/16/2016 12:50:01 PM,\n"
        "460.00000,12/16/2016 12:55:01 PM,\n"
        "480.00000,12/16/2016 12:59:01 PM,\n"
    )
    assert io_load._detect_header_skiprows(str(p)) == 0
    with pytest.raises(pd.errors.ParserError):
        io_load.load_csv(str(p))


def test_detect_header_skiprows_skips_past_all_numeric_metadata_row(tmp_path):
    # An all-numeric metadata row (e.g. an INSITE-style "0,0,0" row) happens to land at the modal
    # width, same as the real header and the data rows that follow it. It must be rejected (a
    # numeric first field) so detection continues to the real, non-numeric header just after it.
    # "Row(s): 4" (1 field) is kept from the plain FIX C fixture so the initial pd.read_csv still
    # raises ParserError (a clean 2-field-preamble-then-3-field-table would instead let pandas
    # silently promote the extra column to an index, never reaching this fallback at all).
    p = tmp_path / "numeric_metadata.csv"
    p.write_text(
        "Job ID: 9942,Spotter: 1115093\n"
        "Row(s): 4\n"
        "0,0,0\n"
        "Date/Time,TZ,Pressure(psia)\n"
        "2018-03-29 20:58:24,CDT,16.9\n"
        "2018-03-29 20:58:25,CDT,16.9\n"
    )
    assert io_load._detect_header_skiprows(str(p)) == 3
    td = io_load.load_csv(str(p))
    assert list(td.df.columns) == ["Date/Time", "TZ", "Pressure(psia)"]
    assert td.n == 2


def test_detect_header_skiprows_never_returns_a_numeric_first_field_line(tmp_path):
    # Anti-regression across every FIX C / DEFECT 2 fixture pinned above: whatever line index is
    # returned, that line's first field must never itself parse as a bare number.
    fixtures = [
        (
            "Job ID: 9942,Spotter: 1115093\n"
            "Row(s): 4\n"
            "Date/Time,TZ,Pressure(psia)\n"
            "2018-03-29 20:58:24,CDT,16.9\n"
            "2018-03-29 20:58:25,CDT,16.9\n"
        ),
        (
            "Job ID: 13751,Spotter: 1115471\n"
            "Row(s): 3\n"
            "Datetime,TZ,Delta(Hrs),Flow Rate(m3/min),Totaliser(m3)\n"
            "2022-10-31 10:35:43.000,MDT,0.0000000,0.000,0.044\n"
            "2022-10-31 10:36:43.000,MDT,0.0166667,0.010,0.045\n"
        ),
        (
            "Job ID: 9942,Spotter: 1115093\n"
            "Row(s): 4\n"
            "0,0,0\n"
            "Date/Time,TZ,Pressure(psia)\n"
            "2018-03-29 20:58:24,CDT,16.9\n"
            "2018-03-29 20:58:25,CDT,16.9\n"
        ),
    ]
    for i, text in enumerate(fixtures):
        p = tmp_path / f"nonnumeric_{i}.csv"
        p.write_text(text)
        skiprows = io_load._detect_header_skiprows(str(p))
        header_line = text.splitlines()[skiprows]
        first_field = header_line.split(",")[0].strip()
        with pytest.raises(ValueError):
            float(first_field)


# --------------------------------------------------------------------------------------------------
# DEFECT 6 -- the header detector must never read past 20 lines or let a reader error escape
# --------------------------------------------------------------------------------------------------
def test_detect_header_skiprows_oversized_field_past_line_20_is_never_read(tmp_path):
    # A >131072-char quoted field appearing after line 20 must never be read at all -- islice(20)
    # stops before it -- so detection completes normally instead of raising _csv.Error.
    p = tmp_path / "oversized_after_20.csv"
    lines = [
        "Job ID: 9942,Spotter: 1115093",
        "Row(s): 4",
        "Date/Time,TZ,Pressure(psia)",
        "2018-03-29 20:58:24,CDT,16.9",
        "2018-03-29 20:58:25,CDT,16.9",
    ]
    # Pad to well past line 20 before the oversized field.
    while len(lines) < 25:
        lines.append("2018-03-29 20:58:26,CDT,16.9")
    huge_field = "x" * 200_000
    lines.append(f'2018-03-29 20:58:27,CDT,"{huge_field}"')
    p.write_text("\n".join(lines) + "\n")

    assert io_load._detect_header_skiprows(str(p)) == 2


def test_detect_header_skiprows_oversized_field_within_20_lines_returns_zero(tmp_path):
    # The same oversized quoted field WITHIN the first 20 lines would raise _csv.Error inside the
    # csv reader; that must be caught and turned into a safe "no preamble found" (0), not escape.
    p = tmp_path / "oversized_within_20.csv"
    huge_field = "x" * 200_000
    lines = [
        "Job ID: 9942,Spotter: 1115093",
        f'Row(s): 4,"{huge_field}"',
        "Date/Time,TZ,Pressure(psia)",
        "2018-03-29 20:58:24,CDT,16.9",
    ]
    p.write_text("\n".join(lines) + "\n")

    assert io_load._detect_header_skiprows(str(p)) == 0


def test_detect_header_skiprows_non_utf8_byte_past_line_20_does_not_raise(tmp_path):
    # A non-UTF-8 byte appearing after line 20 must never be decoded at all -- islice(20) stops
    # pulling more rows from the reader once it has 20, and the padding below is sized well past
    # Python's text-mode read-ahead buffer (io.DEFAULT_BUFFER_SIZE) so that buffer's first chunk
    # never reaches the bad byte either. Even if it somehow were reached, the detector must not
    # let UnicodeDecodeError escape -- it must return a plain int, not raise.
    lines = [
        "Job ID: 9942,Spotter: 1115093",
        "Row(s): 4",
        "Date/Time,TZ,Pressure(psia)",
        "2018-03-29 20:58:24,CDT,16.9",
        "2018-03-29 20:58:25,CDT,16.9",
    ]
    # Padding rows long enough that 20 of them alone exceed io.DEFAULT_BUFFER_SIZE (131072
    # bytes), so the read-ahead buffer's first chunk cannot reach the bad byte placed after them.
    filler = "2018-03-29 20:58:26,CDT," + ("9" * 7000)
    while len(lines) < 25:
        lines.append(filler)
    text = "\n".join(lines) + "\n"
    p = tmp_path / "non_utf8_past_20.csv"
    # Write as UTF-8-with-BOM (matching load_csv's own encoding), then append a raw non-UTF-8
    # byte sequence past line 20.
    with open(p, "wb") as f:
        f.write(text.encode("utf-8-sig"))
        f.write(b"2018-03-29 20:58:27,CDT,\xff\xfe\n")

    assert io_load._detect_header_skiprows(str(p)) == 2


def test_detect_header_skiprows_non_utf8_byte_within_20_lines_does_not_raise(tmp_path):
    # GAP 1 (should-fix): the `UnicodeDecodeError` arm of the except tuple is unpinned by the
    # test above, which places its bad byte past the 20-line read-ahead window so it never
    # actually gets decoded. This fixture puts a non-UTF-8 byte on line 2, well within the
    # 20-line sample, of a file whose shape would otherwise raise ParserError (a preamble before
    # the real header) -- the csv reader must actually hit the bad byte and raise
    # UnicodeDecodeError while decoding, and that must be caught and turned into a safe 0, not
    # escape.
    p = tmp_path / "non_utf8_within_20.csv"
    with open(p, "wb") as f:
        f.write(b"Job ID: 9942,Spotter: 1115093\n")
        f.write(b"Row(s): 4\xff\n")
        f.write(b"Date/Time,TZ,Pressure(psia)\n")
        f.write(b"2018-03-29 20:58:24,CDT,16.9\n")
        f.write(b"2018-03-29 20:58:25,CDT,16.9\n")

    assert io_load._detect_header_skiprows(str(p)) == 0


# --------------------------------------------------------------------------------------------------
# FIX D -- reverse-chronological export
# --------------------------------------------------------------------------------------------------
def test_load_csv_reverses_fully_descending_export(tmp_path):
    # Reproduces the Civitas Bijou shape: clean ISO timestamps, newest row first.
    p = tmp_path / "civitas.csv"
    p.write_text(
        '"Timestamp (MST)","PRESS"\n'
        '"2024-12-06 12:39:10","300.0"\n'
        '"2024-12-06 12:39:05","200.0"\n'
        '"2024-12-06 12:39:00","100.0"\n'
    )
    td = io_load.load_csv(str(p))
    assert td.n == 3
    assert td.t_s[0] == 0.0
    assert np.all(np.diff(td.t_s) >= 0)
    # Re-association check: pressure values must travel with their own row, not get scrambled.
    pressures = td.column("PRESS")
    assert pressures[0] == 100.0
    assert pressures[-1] == 300.0


def test_load_csv_does_not_reorder_a_few_out_of_order_rows(tmp_path):
    # Note: this is *not* a "mostly ascending, one blip" file -- 2 of its 3 consecutive steps go
    # backwards (12:39:00 -> 12:38:50 -> 12:39:20 -> 12:38:40). It still must NOT be reordered,
    # because it is not exactly monotonic decreasing end to end (a "mostly decreasing" fraction
    # heuristic would wrongly reverse this).
    p = tmp_path / "not_wholly_ordered.csv"
    p.write_text(
        '"Timestamp (MST)","PRESS"\n'
        '"2024-12-06 12:39:00","0.0"\n'
        '"2024-12-06 12:38:50","100.0"\n'
        '"2024-12-06 12:39:20","200.0"\n'
        '"2024-12-06 12:38:40","300.0"\n'
    )
    td = io_load.load_csv(str(p))
    pressures = td.column("PRESS")
    # Unreordered: first row is still the first row written in the file.
    assert pressures[0] == 0.0
    assert pressures[-1] == 300.0


def test_load_csv_does_not_reorder_all_identical_timestamps(tmp_path):
    p = tmp_path / "all_same.csv"
    p.write_text(
        '"Timestamp (MST)","PRESS"\n'
        '"2024-12-06 12:39:00","100.0"\n'
        '"2024-12-06 12:39:00","200.0"\n'
        '"2024-12-06 12:39:00","300.0"\n'
    )
    td = io_load.load_csv(str(p))
    pressures = td.column("PRESS")
    assert pressures[0] == 100.0
    assert pressures[-1] == 300.0


# --------------------------------------------------------------------------------------------------
# regression guard
# --------------------------------------------------------------------------------------------------
def test_load_csv_ordinary_month_first_csv_unchanged(tmp_path):
    p = tmp_path / "ordinary.csv"
    p.write_text(
        "Date/Time,Pressure (psi)\n"
        "8/9/2022 08:23:17,5000.0\n"
        "8/9/2022 08:23:29,4995.0\n"
        "8/10/2022 08:23:40,4990.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Date/Time"
    assert td.n == 3
    dt = td.df["Date/Time"]
    # Month-first: 8/9/2022 -> August 9, not September 8.
    assert dt.iloc[0] == pd.Timestamp("2022-08-09 08:23:17")
    assert dt.iloc[-1] == pd.Timestamp("2022-08-10 08:23:40")
    np.testing.assert_allclose(td.t_s, [0.0, 12.0, 86423.0])
