"""Tests for the CSV loader (``load_csv``, ``parse_datetime``, ``suggest_channels``) in
dfit_tool/io_load.py.

All fixtures are synthetic and written under ``tmp_path`` -- no test here reads anything under
``C:\\DFIT Data``. Each fixture reproduces, in miniature, the exact column-name and value shapes
of a real corpus file (named in the fix's docstring/comment) so a regression in the loader shows
up here rather than only in a real load.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from dfit_tool import io_load, units

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


# --------------------------------------------------------------------------------------------------
# multi-candidate datetime column choice (load_csv, io_load._best_datetime_column)
#
# Measured case: Badger 25N-3HZ DFIT.csv has both "Time" (elapsed minutes, e.g. 85.00000) and
# "Job Time" (wall-clock "12/21/2017 9:22:20 AM" strings). suggest_channels' own find() picks
# "Time" by column order (the wrong one) -- load_csv must instead evaluate every datetime-name
# candidate and keep whichever parses the most valid timestamps.
# --------------------------------------------------------------------------------------------------
def test_load_csv_picks_job_time_over_elapsed_time_column(tmp_path):
    rows = [
        "85.00000,12/21/2017 9:22:20 AM,-14.91,246.86",
        "85.01667,12/21/2017 9:22:21 AM,-13.96,246.34",
        "85.03334,12/21/2017 9:22:22 AM,-13.48,247.00",
        "85.05001,12/21/2017 9:22:23 AM,-13.10,247.40",
    ]
    p = tmp_path / "badger_shape.csv"
    p.write_text(
        "Time,Job Time,Treating Pressure,Surface Pressure\n"
        + "\n".join(rows) + "\n"
    )

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Job Time"
    # ~3 s of real wall-clock span, not ~0.05 min misread as an Excel serial number.
    span_s = td.t_s[-1] - td.t_s[0]
    assert span_s == pytest.approx(3.0, abs=0.01)


def test_load_csv_single_datetime_candidate_unchanged(tmp_path):
    # Only one column matches the datetime needles at all -- the multi-candidate evaluation must
    # not run (and must not change the outcome if it did).
    p = tmp_path / "single_candidate.csv"
    p.write_text(
        "Date/Time,Pressure (psi)\n"
        "8/9/2022 08:23:17,5000.0\n"
        "8/9/2022 08:23:29,4995.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Date/Time"


def test_load_csv_multi_candidate_tie_keeps_suggest_channels_choice(tmp_path):
    # "Time" and "Job Time" both hold the same valid wall-clock strings -- an exact tie in valid-
    # parse count -- so the winner must stay suggest_channels' own column-order choice ("Time")
    # rather than switch to "Job Time".
    p = tmp_path / "tie_shape.csv"
    p.write_text(
        "Time,Job Time,Pressure (psi)\n"
        "8/9/2022 08:23:17,8/9/2022 08:23:17,5000.0\n"
        "8/9/2022 08:23:29,8/9/2022 08:23:29,4995.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Time"


def test_load_csv_multi_candidate_recomputes_companion_time_col(tmp_path):
    # The winning datetime column can differ from suggest_channels' own guess, so its companion
    # "Time" column (FIX A) must be re-derived against the WINNER, not carried over from the
    # original guess. "Elapsed Date" (bad numbers) loses to "Real Date" (real dates); "Real Date"
    # is date-like-not-time-like, so its own companion "Time" column must be picked up and joined.
    rows = [
        "9999.0,8/9/2022,8:23:17,100.0",
        "9998.0,8/9/2022,8:23:29,101.0",
    ]
    p = tmp_path / "companion_shape.csv"
    p.write_text(
        "Elapsed Date,Real Date,Time,Pressure (psi)\n"
        + "\n".join(rows) + "\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Real Date"
    dt = td.df["Real Date"]
    assert dt.iloc[0] == pd.Timestamp("2022-08-09 08:23:17")
    assert dt.iloc[-1] == pd.Timestamp("2022-08-09 08:23:29")


def test_companion_time_col_excludes_elapsed_prefers_real(tmp_path):
    # FIX 6: "Date, Elapsed Time, Real Time" must pair Date with Real Time, not the
    # elapsed-minutes column column order alone would otherwise pick first.
    cols = ["Date", "Elapsed Time", "Real Time", "Pressure"]
    assert io_load._companion_time_col("Date", cols) == "Real Time"


def test_companion_time_col_excludes_test_delta_duration_tokens():
    assert io_load._companion_time_col("Date", ["Date", "Test Time", "Pressure"]) is None
    assert io_load._companion_time_col("Date", ["Date", "Delta Time", "Pressure"]) is None
    assert io_load._companion_time_col("Date", ["Date", "Duration Time", "Pressure"]) is None


def test_companion_time_col_preferred_wins_over_plain_order():
    # "Clock Time" sits AFTER "Job Time" in column order but must still win as the preferred
    # wall-clock-shaped name.
    cols = ["Date", "Job Time", "Clock Time", "Pressure"]
    assert io_load._companion_time_col("Date", cols) == "Clock Time"


def test_companion_time_col_exact_bare_time_still_wins_outright():
    # The narrower, original rule is unaffected by the exclude/preferred logic: an exact bare
    # "Time" column always wins, even over a "Real Time" that would otherwise be preferred.
    cols = ["Date", "Real Time", "Time", "Pressure"]
    assert io_load._companion_time_col("Date", cols) == "Time"


def test_load_csv_date_elapsed_real_time_end_to_end(tmp_path):
    # Full end-to-end regression for the "Date, Elapsed Time, Real Time" shape: the join must use
    # Real Time (full sub-second resolution), not Elapsed Time.
    rows = [
        "04/10/2016,0.0,12:00:00,5000.0",
        "04/10/2016,0.08333333333333333,12:00:05,4999.0",
        "04/10/2016,0.16666666666666666,12:00:10,4998.0",
    ]
    p = tmp_path / "date_elapsed_real.csv"
    p.write_text("Date,Elapsed Time,Real Time,Pressure\n" + "\n".join(rows) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Date"
    span_s = td.t_s[-1] - td.t_s[0]
    assert span_s == pytest.approx(10.0)


def test_load_csv_job_time_full_datetime_kept_when_it_leads(tmp_path):
    # FIX 6 regression: a "Job Time" column of full datetimes, sitting BEFORE a bare "Date"
    # column, must stay the datetime column at full resolution -- not get reserved away as
    # "Date"'s (bogus) time-of-day companion, which would force the wrong column and collapse
    # every sample onto one midnight timestamp.
    rows = [
        "04/10/2016 12:00:00,04/10/2016,5000.0",
        "04/10/2016 12:00:05,04/10/2016,4999.0",
        "04/10/2016 12:00:10,04/10/2016,4998.0",
    ]
    p = tmp_path / "jobtime_leads.csv"
    p.write_text("Job Time,Date,Pressure\n" + "\n".join(rows) + "\n")

    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Job Time"
    span_s = td.t_s[-1] - td.t_s[0]
    assert span_s == pytest.approx(10.0)


def test_companion_is_time_of_day_vetoes_full_datetime_companion(tmp_path):
    # A "Job Time" column matches "Date"'s companion NAME rule, but its own VALUES are full
    # datetimes, not bare time-of-day -- _companion_is_time_of_day must veto it.
    df = pd.DataFrame({
        "Date": ["04/10/2016", "04/10/2016"],
        "Job Time": ["04/10/2016 12:00:00", "04/10/2016 12:00:05"],
    })
    assert io_load._companion_is_time_of_day("Job Time", df) is False


def test_companion_is_time_of_day_tolerates_leading_units_row():
    # A real corpus companion column routinely carries one leading units-declaration row (e.g.
    # "(hh:mm:ss)") among thousands of genuine time-of-day rows -- that must not veto it.
    values = ["(hh:mm:ss)"] + [f"14:0{i}:00" for i in range(9)]
    df = pd.DataFrame({"Real Time": values})
    assert io_load._companion_is_time_of_day("Real Time", df) is True


def test_companion_is_time_of_day_normalizes_ms_colon():
    df = pd.DataFrame({"Time": ["15:58:17:647", "15:58:17:897"]})
    assert io_load._companion_is_time_of_day("Time", df) is True


# --------------------------------------------------------------------------------------------------
# Caprito "%m-%d-%Y_%H:%M:%S" fast path (FIX 9) and sub-second fast path (FIX 7)
# --------------------------------------------------------------------------------------------------
def test_parse_datetime_caprito_underscore_format():
    s = pd.Series(["03-26-2018_16:44:05", "03-26-2018_16:44:06"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2018-03-26 16:44:05")
    assert result.iloc[1] == pd.Timestamp("2018-03-26 16:44:06")


def test_load_csv_caprito_timestamp_shape_end_to_end(tmp_path):
    rows = [
        "03-26-2018_16:44:05,0.95",
        "03-26-2018_16:44:06,5.85",
        "03-26-2018_16:44:07,3.95",
    ]
    p = tmp_path / "caprito_shape.csv"
    p.write_text("Time Stamp,DischPress\n" + "\n".join(rows) + "\n")
    td = io_load.load_csv(str(p))
    assert td.n == 3
    assert td.t_s[-1] - td.t_s[0] == pytest.approx(2.0)


def test_parse_datetime_fractional_seconds_fast_path():
    s = pd.Series(["01/02/2020 03:04:05.123456", "01/02/2020 03:04:06.654321"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2020-01-02 03:04:05.123456")
    assert result.iloc[1] == pd.Timestamp("2020-01-02 03:04:06.654321")


# --------------------------------------------------------------------------------------------------
# parse_datetime -- Excel serial fallback plausible-range guard
#
# Fallback 1 (bare Excel serial numbers) must not treat every bare number as a date: only serials
# within the plausible 1990-2100 range (io_load._EXCEL_SERIAL_MIN/MAX) are accepted.
# --------------------------------------------------------------------------------------------------
def test_parse_datetime_accepts_real_excel_serial():
    # Fallback 1 rounds to whole seconds (~1 Hz data): 43508.34097 days is 08:10:59.808, which
    # rounds up to 08:11:00.
    s = pd.Series(["43508.34097"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2019-02-12 08:11:00")


def test_parse_datetime_rejects_small_bare_number_as_serial():
    # 85.0 sits nowhere near the plausible Excel-serial range -- it's an elapsed-minutes value,
    # not a date -- and Fallback 2's generic dateutil parse also has no date-like shape to read
    # out of a bare decimal, so the result must be NaT, not 1900-03-25.
    s = pd.Series(["85.00000"], dtype="string")
    result = io_load.parse_datetime(s)
    assert pd.isna(result.iloc[0])


def test_load_csv_bare_numeric_elapsed_column_falls_back_to_fix_b(tmp_path):
    # With the serial-range guard in place, a file whose only time-ish column is a bare numeric
    # elapsed-minutes column no longer misparses as ~1900 dates (valid_frac drops to ~0) -- it now
    # correctly drops through to the FIX B elapsed-column fallback (_find_elapsed_column) instead,
    # provided that column is named the way FIX B recognizes ("delta"/"elapsed", see
    # _find_elapsed_column). Before the guard, Fallback 1 accepted every bare number as a serial,
    # so valid_frac read 1.0 and FIX B never got a chance to run -- this file loaded, silently,
    # with the same class of bogus ~1900 dates as the Badger bug.
    p = tmp_path / "elapsed_only.csv"
    p.write_text(
        "Elapsed (min),Pressure (psi)\n"
        "0.00000,5000.0\n"
        "0.01667,4995.0\n"
        "0.03334,4990.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.n == 3
    assert td.t_s[0] == 0.0
    np.testing.assert_allclose(td.t_s, [0.0, 1.0002, 2.0004], atol=1e-6)


# --------------------------------------------------------------------------------------------------
# parse_datetime -- Fallback 2 must never resolve a value pd.to_numeric accepts
#
# Review finding: rejecting an out-of-range value in Fallback 1 (above) isn't enough on its own --
# Fallback 2's unrestricted dateutil parse was still happy to read a bare number as a date by a
# different route ("2024" -> 2024-01-01, "1500.5" -> 1500-05-01, "20171221" -> 2017-12-21). A
# purely numeric string gets exactly one route to a date (Fallback 1's Excel-serial parse); if
# that fallback already rejected it, Fallback 2 must not get a second, unguarded attempt.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("value", ["2024", "1500.5", "20171221", "1999.99"])
def test_parse_datetime_fallback2_never_parses_bare_numbers(value):
    s = pd.Series([value], dtype="string")
    result = io_load.parse_datetime(s)
    assert pd.isna(result.iloc[0])


# --------------------------------------------------------------------------------------------------
# parse_datetime -- 12-hour AM/PM fast path
#
# A whole-file AM/PM-formatted column (e.g. Badger 25N-3HZ's "12/21/2017 9:22:20 AM") must resolve
# through the second exact, vectorized fast path (_PRIMARY_DT_FORMAT_AMPM) rather than falling
# through to the much slower per-element dateutil parse -- verified here by asserting no
# "Could not infer format" UserWarning fires (that warning only ever comes from the dateutil
# fallback), not just by checking the parsed value.
# --------------------------------------------------------------------------------------------------
def test_parse_datetime_ampm_fast_path_no_dateutil_warning():
    s = pd.Series(["12/21/2017 9:22:20 AM", "12/21/2017 9:22:21 AM"], dtype="string")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2017-12-21 09:22:20")
    assert result.iloc[1] == pd.Timestamp("2017-12-21 09:22:21")


def test_parse_datetime_ampm_fast_path_dayfirst_variant():
    # Day-first hint (day component exceeds 12 in one row) must select the day-first AM/PM format,
    # not just the month-first one.
    s = pd.Series(["15/8/2022 9:22:20 AM", "20/8/2022 1:05:00 PM"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2022-08-15 09:22:20")
    assert result.iloc[1] == pd.Timestamp("2022-08-20 13:05:00")


def test_parse_datetime_tz_aware_string_does_not_raise():
    # Measured case: a SCADA-style "Datetime(UTC)" column with an explicit UTC marker (e.g.
    # "2019-08-29 14:13:35Z") makes Fallback 2's dateutil parse infer a tz-AWARE result, which
    # `.astype("datetime64[us]")` refuses outright (a TypeError, not a parse failure) -- this
    # column doesn't even have to be the one `load_csv` ends up using: _best_datetime_column
    # scores every multi-candidate column, including ones it ultimately rejects, so a tz-aware
    # sibling must not crash the load just for being evaluated. The offset is dropped (local
    # wall-clock time kept), not converted -- elapsed-time math never needs absolute UTC.
    s = pd.Series(["2019-08-29 14:13:35Z", "2019-08-29 14:13:39Z"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2019-08-29 14:13:35")
    assert result.iloc[1] == pd.Timestamp("2019-08-29 14:13:39")


def test_parse_datetime_fallback2_rejects_implausible_dateutil_guess():
    # Measured case: Hodges 2CH DFIT.csv's ~1.05M-row "Date Time" column has two genuinely
    # corrupted cells ("23 23:43:18" -- missing its date entirely, "4/19/227" -- a typo'd year),
    # and dateutil's generic parser happily guesses year-1 and year-227 timestamps for them
    # rather than failing. Unguarded, either one poisons a min/max computed over the column with
    # a bogus multi-century span. Both must stay NaT, same as an out-of-window Excel serial.
    s = pd.Series(["04/11/2023 14:33:12", "23 23:43:18", "4/19/227"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2023-04-11 14:33:12")
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_parse_datetime_fast_path_rejects_implausible_year():
    # M2 (round 3): an exact-format strptime match has no range check of its own -- a 4-digit
    # year field accepts ANY 4 digits -- so a corrupted-but-well-shaped date (e.g. a typo'd
    # year) previously sailed through the fast path untouched by any guard at all. Measured
    # case: Civitas Allred's "6 - 21011210.DTF.csv" has several implausible years (1941, 4221,
    # 7127) that match "%m/%d/%Y %H:%M:%S" perfectly.
    s = pd.Series(
        ["04/11/2023 14:33:12", "02/04/7127 22:00:00", "04/16/4221 22:00:00"], dtype="string"
    )
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2023-04-11 14:33:12")
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])


def test_load_csv_tz_aware_sibling_column_does_not_crash(tmp_path):
    # End-to-end version of the above: a good, primary-format "Datetime" column (no tz) alongside
    # a tz-aware "Datetime(UTC)" sibling -- both are datetime-name-matching candidates, and the
    # tz-aware one must not crash multi-candidate scoring even though "Datetime" wins outright.
    p = tmp_path / "tz_sibling.csv"
    p.write_text(
        "Datetime,Datetime(UTC),Pressure (psi)\n"
        "08/29/2019 08:13:35,2019-08-29 14:13:35Z,75.0\n"
        "08/29/2019 08:13:39,2019-08-29 14:13:39Z,0.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Datetime"


def test_parse_datetime_numeric_utc_offset_does_not_raise():
    # Same TypeError as the "Z"-suffixed case above, but with a numeric UTC offset instead of a
    # bare "Z" marker (e.g. Edge Energy Simpson 36-1H / Synergy Resources Sanford's
    # "HistReport_..." files, whose SCADA export uses "-07:00"-style offsets). The offset is
    # dropped (local wall-clock kept), matching the "Z" case.
    s = pd.Series(
        ["2019-11-08 07:00:00-07:00", "2019-11-08 07:00:01-07:00"], dtype="string"
    )
    result = io_load.parse_datetime(s)
    assert result.iloc[0] == pd.Timestamp("2019-11-08 07:00:00")
    assert result.iloc[1] == pd.Timestamp("2019-11-08 07:00:01")


def test_parse_datetime_rejects_bare_clock_strings():
    # Measured case: Crescent Point's Dressler "...1secdata.csv" has an unnamed first column
    # (an INSITE treatment log under an undetected preamble) whose real values are elapsed
    # "MM:SS.f" clock strings with no hours and no date at all -- e.g. "09:07.0". Read alone,
    # dateutil parses "09:07.0" as 09:07:00 defaulted to TODAY's date (a "valid" timestamp that
    # is really just an artifact of when the code runs), while a minutes value >= 24 (an invalid
    # hour) correctly fails as NaT already -- both must end up NaT, not a mix of NaT and
    # spuriously-dated rows.
    s = pd.Series(["09:07.0", "09:08.0", "27:32.7", "13:05:01"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.isna().all()


def test_load_csv_bare_clock_column_fails_instead_of_garbage_duration(tmp_path):
    # End-to-end Dressler repro: an unnamed, numeric-named first column (no header row detected
    # under a uniformly-7-field preamble, so a job-header line and a units line both land as
    # "data") whose real values are "MM:SS.f" elapsed clock strings. Before the guard above, one
    # genuine date elsewhere in the column (a job-header line, "11-Aug-14") anchored the elapsed-
    # time math against thousands of today-dated "MM:SS.f" rows years apart -- a ~12-YEAR
    # "duration" for what is actually a same-day 1-second-cadence pump job. With the guard, only
    # that one genuine date survives, which is not enough to define a usable timeframe, so the
    # load must fail outright rather than report a plausible-looking but meaningless result.
    lines = [
        "2,3,4",
        "Job Data Listing 1secdata,,",
        "INSITE for Stimulation v4.5.1,,",
        "11-Aug-14,,",
        "Time,Slurry Rate,Treating Pressure",
        "(hh:mm:ss.ttt),(bpm),(psi)",
        "09:07.0,10.0,5000.0",
        "09:08.0,10.0,4995.0",
        "09:09.0,10.0,4990.0",
    ]
    p = tmp_path / "dressler_shape.csv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        io_load.load_csv(str(p))


def test_parse_datetime_mixed_utc_offsets_across_dst_does_not_raise():
    # L2: plain pd.to_datetime raises "Mixed timezones detected" outright (even with
    # errors="coerce") when an array mixes tz-aware values with DIFFERENT UTC offsets -- e.g. a
    # SCADA export spanning a DST transition, "-05:00" before it and "-06:00" after. This must be
    # caught and retried with utc=True (a common reference instant), not allowed to propagate --
    # including when the column carrying it merely LOSES multi-candidate scoring, so this must
    # never raise out of _score_datetime_candidate either.
    s = pd.Series(
        ["2019-11-03 01:00:00-05:00", "2019-11-03 01:30:00-05:00",
         "2019-11-03 01:00:00-06:00", "2019-11-03 01:30:00-06:00"],
        dtype="string",
    )
    result = io_load.parse_datetime(s)
    assert result.notna().all()
    # Real elapsed time across the fall-back transition: 90 minutes, not the 30 minutes naive
    # local-wall-clock arithmetic would show (each pair 30 min apart, but the second pair is a
    # further hour later in absolute time once the clocks fall back).
    assert (result.iloc[-1] - result.iloc[0]).total_seconds() == 90 * 60


def test_score_datetime_candidate_mixed_tz_losing_candidate_does_not_raise(tmp_path):
    # End-to-end version of the above: a good "Datetime" column wins outright, but a losing
    # sibling candidate carries the DST-mixed offsets. Scoring it must not crash the whole load.
    p = tmp_path / "mixed_tz_sibling.csv"
    p.write_text(
        "Datetime,Datetime(local),Pressure (psi)\n"
        "11/03/2019 01:00:00,2019-11-03 01:00:00-05:00,75.0\n"
        "11/03/2019 01:05:00,2019-11-03 01:00:00-06:00,74.0\n"
    )
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Datetime"


# --------------------------------------------------------------------------------------------------
# M2 -- Fallback 2 runs at full length (not reindexed to the still-NaT subset), so a malformed
# row keeps its real neighbors for format-inference context.
# --------------------------------------------------------------------------------------------------
def test_parse_datetime_corrupted_row_stays_nat_with_well_formed_neighbors():
    # Measured case: Strathcona's 100-09-14-062-04W6-rt.csv has a row whose Date cell lost its
    # leading day digits ("/10/2022" instead of "31/10/2022"), joined with a real Time into
    # "/10/2022 22:11:25", sitting among thousands of well-formed "31/10/2022 22:11:25"-shaped
    # neighbors. Parsed WITH those neighbors, pandas correctly guesses the shared dayfirst format
    # and the corrupted row fails to match it (NaT) -- parsed in isolation (or reindexed into a
    # subset without those neighbors), dateutil's own single-value guess reads it as a
    # plausible-but-wrong 2022-10-01, which this must not do.
    # Enough total rows, and enough OTHER unparseable ones, that the fast-path-resolved fraction
    # stays below _FAST_PATH_SKIP_FRACTION (0.90) -- otherwise Fallback 2 would be skipped
    # entirely and this test would pass for the wrong reason (the fast path alone already fails
    # the corrupted row, without ever exercising Fallback 2's context-aware rejection).
    good = ["31/10/2022 22:11:{:02d}".format(s) for s in range(30)]
    values = (
        good[:5] + ["/10/2022 22:11:25"] + good[5:15]
        + ["/10/2022 22:12:10", "/10/2022 22:12:45", "/10/2022 22:13:10"] + good[15:]
    )
    s = pd.Series(values, dtype="string")
    result = io_load.parse_datetime(s)
    assert pd.isna(result.iloc[5])
    assert result.iloc[4] == pd.Timestamp("2022-10-31 22:11:04")
    assert result.iloc[6] == pd.Timestamp("2022-10-31 22:11:05")


def test_parse_datetime_no_digit_value_does_not_break_whole_column_format_guess():
    # A value with no digit at all (e.g. a units-declaration row misread as data, "(date time)")
    # must be masked out BEFORE the generic parse, not just discarded after -- left in, it can
    # make pandas fail to guess ANY shared format for the array at all (a real perf cliff on a
    # large column, see the fix's own comment; here just checked for correctness) and everything
    # else in the array must still parse normally around it.
    # Two more no-digit junk rows beyond the measured "(date time)" one, purely to keep the
    # fast-path-resolved fraction below _FAST_PATH_SKIP_FRACTION (0.90) -- otherwise Fallback 2
    # would be skipped entirely and this test would pass without ever exercising the masking it
    # claims to check.
    good = ["12/14/2016 10:25:{:02d} AM".format(s) for s in range(20)]
    values = ["(date time)"] + good + ["N/A", "---"]
    s = pd.Series(values, dtype="string")
    result = io_load.parse_datetime(s)
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == pd.Timestamp("2016-12-14 10:25:00")
    assert result.notna().sum() == 20


# --------------------------------------------------------------------------------------------------
# Isolated-timestamp-outlier guard (_mask_isolated_timestamp_outliers): a single corrupted cell
# surrounded on both sides by mutually consistent neighbors is masked, not left to poison the
# reported span with a huge spurious swing.
# --------------------------------------------------------------------------------------------------
def test_load_csv_isolated_timestamp_outlier_masked(tmp_path):
    # Measured case: Strathcona's 100-09-14-062-04W6-rt.csv, row 5563 -- a Date cell reads
    # "8/10/2022" where every neighbor reads "28/10/2022" (a dropped leading digit), landing
    # that one sample ~20 days before its neighbors and the next sample ~20 days back again.
    lines = ["Date,Time,Pressure(psi)"]
    for i in range(20):
        day = 8 if i == 10 else 28
        lines.append(f"{day}/10/2022,17:23:{i:02d},{5000 - i}")
    p = tmp_path / "isolated_outlier.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    t = np.asarray(td.t_s, dtype=float)
    assert np.isnan(t[10])
    assert np.isfinite(t).sum() == 19
    # The surviving samples still increase smoothly at 1s/row across the masked gap.
    np.testing.assert_allclose(t[9], 9.0)
    np.testing.assert_allclose(t[11], 11.0)
    assert any("outlier" in w for w in td.load_warnings)


def test_load_csv_outlier_warning_not_attached_when_column_abandoned(tmp_path):
    # L3: an outlier gets masked on dt_col's own parse, but the column is BELOW the trust
    # fraction overall and load_csv falls back to a separate elapsed-time column instead --
    # dt_col is not the column actually used, so the outlier warning (which describes ITS
    # parse) must not appear in the final load_warnings.
    lines = ["Date,Delta (sec),Pressure(psi)"]
    for i in range(20):
        # Only every third row has a real date (8 of 20 -- below the 50% trust threshold), one
        # of which (i == 9) is an isolated outlier surrounded by otherwise-consistent neighbors.
        if i % 3 == 0:
            day = 8 if i == 9 else 28
            date_s = f"{day}/10/2022"
        else:
            date_s = "not-a-date"
        lines.append(f"{date_s},{i},{5000 - i}")
    p = tmp_path / "outlier_then_abandoned.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col != "Date"
    assert not any("outlier" in w for w in td.load_warnings)


def test_load_csv_genuine_large_gap_not_masked_as_outlier(tmp_path):
    # A REAL discontinuity -- a genuine gap in logging -- must not be masked: unlike a corrupted
    # single cell, the samples on either side of a real gap are close to EACH OTHER (both near
    # the new, shifted level), not mutually far apart with the outlier sandwiched between two
    # otherwise-adjacent-looking neighbors.
    lines = ["Date,Time,Pressure(psi)"]
    for i in range(10):
        lines.append(f"28/10/2022,17:23:{i:02d},{5000 - i}")
    # A genuine multi-day gap in logging, then readings resume and continue normally.
    for i in range(10):
        lines.append(f"5/11/2022,9:00:{i:02d},{4000 - i}")
    p = tmp_path / "genuine_gap.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    t = np.asarray(td.t_s, dtype=float)
    assert np.isfinite(t).sum() == 20
    assert not any("outlier" in w for w in td.load_warnings)


# --------------------------------------------------------------------------------------------------
# H1 -- clock-only elapsed-time fallback (_find_clock_column): a wall-clock-only column with no
# date anywhere in the file.
# --------------------------------------------------------------------------------------------------
def test_load_csv_clock_only_ampm_column(tmp_path):
    # Great Western Marcus Pad shape: "Time Stamp" is a bare 12-hour clock, no date column at all.
    lines = ["Time Stamp,Rate,Pressure"]
    lines += [f"9:13:{s:02d} AM,0,100" for s in range(32, 40)]
    p = tmp_path / "clock_ampm.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, list(range(8)))
    assert any("clock times only" in w for w in td.load_warnings)


def test_load_csv_clock_only_24h_column(tmp_path):
    # Tap Rock Enron shape: bare "Time", 24-hour, no AM/PM, no date.
    lines = ["Time,Rate,Pressure"]
    lines += [f"16:20:{s:02d},0,100" for s in range(0, 8)]
    p = tmp_path / "clock_24h.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, list(range(8)))


def test_load_csv_clock_only_under_units_row(tmp_path):
    # Fifth Creek / WPX shape: a "(datetime)"/"(hh:mm:ss)" units-declaration row right after the
    # header, misread as an ordinary data row -- must not stop the clock-only column from being
    # recognized (it's just one more non-clock-shaped value the fraction check tolerates).
    lines = ["JobTime,Rate,Pressure", "(datetime),(bpm),(psi)"]
    lines += [f"11:50:{s:02d},0,100" for s in range(56, 60)]
    lines += [f"11:51:{s:02d},0,100" for s in range(0, 4)]
    p = tmp_path / "clock_units_row.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.n == 9
    assert np.isnan(td.t_s[0])
    np.testing.assert_allclose(td.t_s[1:], list(range(8)))


def test_load_csv_clock_only_unwraps_midnight_rollover(tmp_path):
    # A job that runs past midnight: the elapsed time must keep increasing across the rollover,
    # not jump backward.
    lines = ["Time,Rate,Pressure"]
    lines += ["23:59:58,0,100", "23:59:59,0,100", "0:00:00,0,100", "0:00:01,0,100"]
    p = tmp_path / "clock_rollover.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, [0.0, 1.0, 2.0, 3.0])


def _fmt_hms(total_s):
    total_s = int(total_s)
    return f"{total_s // 3600}:{(total_s % 3600) // 60:02d}:{total_s % 60:02d}"


def test_load_csv_clock_overnight_gap_unwraps_correctly(tmp_path):
    # L4: a sparsely-sampled overnight gap (evening readings, then the next morning's, nothing
    # in between) must still unwrap -- the backward step from the last evening sample to the
    # first morning one is well under the OLD 12h threshold's proof, but well over the new 1h
    # one. 20:00-21:00 then 10:00-11:00 the next day: 1h + a 13h gap + 1h = 15h total.
    lines = ["Time,Rate,Pressure"]
    lines += [f"{_fmt_hms(s)},0,100" for s in range(20 * 3600, 21 * 3600, 60)]
    lines += [f"{_fmt_hms(s)},0,100" for s in range(10 * 3600, 11 * 3600, 60)]
    p = tmp_path / "clock_overnight_gap.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    # Last sample is 10:59:00 (range() stops one step short of 11:00:00) -- 14h59m after the
    # first sample at 20:00:00.
    np.testing.assert_allclose(td.t_s[-1], 14 * 3600 + 59 * 60, rtol=0, atol=1)
    assert np.all(np.diff(td.t_s) >= 0)


def test_load_csv_clock_reverses_newest_first_log(tmp_path):
    # L4: a reverse-chronological (newest-first) clock log, crossing midnight, must be reversed
    # (mirroring FIX D) before unwrapping -- not left descending. Chronological 22:00 -> 26:00
    # (02:00 the next day), stored newest-first: a real 4h span.
    times = [_fmt_hms(s % 86400) for s in range(22 * 3600, 26 * 3600, 60)][::-1]
    lines = ["Time,Rate,Pressure"] + [f"{t},0,100" for t in times]
    p = tmp_path / "clock_reverse_midnight.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    # Last (chronologically) sample is 01:59:00 (range() stops one step short of 02:00:00) --
    # 3h59m after the first sample at 22:00:00.
    np.testing.assert_allclose(td.t_s[-1], 3 * 3600 + 59 * 60, rtol=0, atol=1)
    assert np.all(np.diff(td.t_s) >= 0)
    assert any("reversed" in w for w in td.load_warnings)


def test_load_csv_clock_elapsed_hms_past_24h_not_truncated(tmp_path):
    # Measured (synthetic) bug: an H:MM:SS elapsed-DURATION column (not a wall clock) whose hour
    # field genuinely runs past 23 (e.g. "39:59:00") was previously read as a wall clock, which
    # can never represent hour>=24 -- those rows became NaT, truncating a real 40h record down
    # to the ~24h a naive reading would show. A 3-field value with any hour>23 must be read as
    # an elapsed duration instead (no midnight unwrap -- it's already monotonic by construction).
    lines = ["Time,Rate,Pressure"]
    lines += [f"{_fmt_hms(s)},0,100" for s in range(0, 40 * 3600, 60)]
    p = tmp_path / "clock_elapsed_hms_40h.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == len(td.t_s)
    np.testing.assert_allclose(td.t_s[-1], (40 * 3600) - 60, rtol=0, atol=1)
    assert any("elapsed duration" in w for w in td.load_warnings)


def test_load_csv_clock_elapsed_mmss_past_60min_not_misread_as_hours(tmp_path):
    # Measured (synthetic) bug: a bare "MM:SS" elapsed-duration column (e.g. "39:59", 39 minutes
    # 59 seconds) whose first field exceeds 23 at some point can never be a valid 24-hour clock
    # hour either -- it must be elapsed minutes, not misread as an "H:MM" wall clock (which would
    # turn a 40-minute record into a bogus ~40-HOUR span).
    lines = ["Time,Rate,Pressure"]
    lines += [f"{s // 60:02d}:{s % 60:02d},0,100" for s in range(0, 40 * 60)]
    p = tmp_path / "clock_elapsed_mmss_40min.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == len(td.t_s)
    np.testing.assert_allclose(td.t_s[-1], 40 * 60 - 1, rtol=0, atol=0.01)
    assert any("elapsed duration" in w for w in td.load_warnings)


def test_load_csv_clock_ambiguous_two_field_raises(tmp_path):
    # A bare 2-field, no-AM/PM reading that fits none of _classify_clock_mode's proofs -- the
    # first field never exceeds 23 (not elapsed_mmss), never wraps from >=23 down to 0 (not a
    # proven clock), and a fresh (first, second) pair shows up almost every row (median run
    # length < 2, so its own resolution isn't coarser than the sample rate either) -- is
    # genuinely ambiguous between "H:MM" and "MM:SS" and must raise a clear, specific error
    # naming that ambiguity, not silently guess either way.
    lines = ["Time,Rate,Pressure"]
    lines += [f"{s // 60:02d}:{s % 60:02d},0,100" for s in range(0, 10 * 60)]
    p = tmp_path / "clock_ambiguous.csv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="H:MM vs MM:SS ambiguous"):
        io_load.load_csv(str(p))
    with pytest.raises(ValueError) as exc_info:
        io_load.load_csv(str(p))
    assert "override" not in str(exc_info.value)


def test_load_csv_clock_two_field_midnight_wrap_is_clock(tmp_path):
    # A backward step in the first field from >=23 down to 0 is conclusive proof of a real
    # wall-clock hour rolling over -- elapsed minutes never reset to 0 after climbing past 23,
    # they just keep counting. H:MM, one sample per minute, crossing midnight: 20:00 -> 03:59
    # the next "day" is a real ~8h span.
    lines = ["Time,Rate,Pressure"]
    lines += [f"{(m // 60) % 24}:{m % 60:02d},0,100" for m in range(20 * 60, 28 * 60)]
    p = tmp_path / "clock_two_field_midnight_wrap.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s[-1] / 3600, 7.983, atol=0.01)
    assert not any("elapsed duration" in w for w in td.load_warnings)


def test_load_csv_clock_two_field_repeated_values_is_clock(tmp_path):
    # A first, second pair that repeats across multiple consecutive rows (median run length
    # >= 2) means the field's own resolution is coarser than the sample rate -- consistent with
    # a real hour:minute clock sampled faster than once a minute (here, once every 10s, so each
    # minute value repeats ~6 times) -- proof of "clock" even with no AM/PM and no wrap.
    lines = ["Time,Rate,Pressure"]
    lines += [
        f"{s // 3600}:{(s % 3600) // 60:02d},0,100" for s in range(14 * 3600, 16 * 3600, 10)
    ]
    p = tmp_path / "clock_two_field_repeated_minute.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert not any("elapsed duration" in w for w in td.load_warnings)
    assert np.isfinite(td.t_s).sum() == len(td.t_s)
    # Run length is a heuristic (a >1 Hz MM:SS log repeats values too), so it is flagged.
    assert any("low confidence" in w for w in td.load_warnings)


def test_load_csv_clock_midnight_wrap_is_not_low_confidence(tmp_path):
    # A midnight wrap is a proof of "clock", so no low-confidence note.
    mins = list(range(22 * 60, 24 * 60)) + list(range(0, 60))
    lines = ["Time,Rate,Pressure"] + [f"{m // 60}:{m % 60:02d},0,100" for m in mins]
    p = tmp_path / "clock_wrap.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert not any("low confidence" in w for w in td.load_warnings)


def test_clock_seconds_of_day_excludes_dressler_ambiguous_mm_ss_shape():
    # H1's own carve-out: Dressler's "MM:SS.f" shape (exactly two colon-separated fields plus a
    # fraction, no seconds group) is ambiguous with an "H:MM" reading of a fractional minute, so
    # _clock_seconds_of_day must never treat it as a usable clock value.
    s = pd.Series(["27:32.7", "09:07.0"], dtype="string")
    result = io_load._clock_seconds_of_day(s)
    assert result.isna().all()


def test_load_csv_dressler_shape_still_fails_not_rescued_by_clock_fallback(tmp_path):
    # H1 must not rescue Crescent Point's Dressler shape: its "MM:SS.f" values are excluded from
    # the clock fallback on principle (see the test above), so this file must keep failing to
    # load exactly as it did before H1 was added.
    lines = [
        "2,3,4",
        "Job Data Listing 1secdata,,",
        "INSITE for Stimulation v4.5.1,,",
        "11-Aug-14,,",
        "Time,Slurry Rate,Treating Pressure",
        "(hh:mm:ss.ttt),(bpm),(psi)",
        "09:07.0,10.0,5000.0",
        "09:08.0,10.0,4995.0",
        "09:09.0,10.0,4990.0",
    ]
    p = tmp_path / "dressler_shape2.csv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        io_load.load_csv(str(p))


# --------------------------------------------------------------------------------------------------
# H2a -- the clock regex accepts 1- or 2-digit seconds (e.g. "15:10:9"), not just 2-digit.
# --------------------------------------------------------------------------------------------------
def test_parse_datetime_rejects_single_digit_second_clock_string():
    # Companion check to the clock-fallback test below: a single-digit-second clock string is
    # still recognized as bare-clock-shaped (and so still rejected by parse_datetime, not handed
    # to dateutil's same-day default).
    s = pd.Series(["15:10:9"], dtype="string")
    result = io_load.parse_datetime(s)
    assert result.isna().all()


def test_load_csv_clock_only_single_digit_seconds(tmp_path):
    # Measured case: Sandpoint's "Sandpoint Resources LLC_07-09-20_15-10-08.csv" opens with
    # "15:10:9" (single-digit seconds) before the rest of the column zero-pads normally -- with
    # only a 2-digit-seconds regex, this one row silently fails the clock-fallback's own
    # predominantly-clock check on a small file, or reads as garbage elsewhere; widened to 1-2
    # digits, the whole column loads.
    lines = ["Time,Rate,Pressure", "15:10:9,0,100", "15:10:10,0,100", "15:10:11,0,100"]
    p = tmp_path / "clock_single_digit_sec.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.n == 3
    np.testing.assert_allclose(td.t_s, [0.0, 1.0, 2.0])


# --------------------------------------------------------------------------------------------------
# H2b -- the trust-fraction check divides by non-empty cells, not every row (blank trailing
# padding rows don't count against it), and raises unconditionally (no ">=2 valid" exception)
# when nothing survives the fraction check and no elapsed/clock fallback applies either.
# --------------------------------------------------------------------------------------------------
def test_load_csv_blank_trailing_rows_do_not_fail_valid_fraction_check(tmp_path):
    # A file with real, fully-parseable data followed by blank padding rows (no values in any
    # column at all) must not have its trust fraction dragged down by those padding rows -- they
    # carry no "invalid" datetime cell, just no cell at all.
    lines = ["Job Time,Pressure(psi)"]
    lines += [f"12/21/2017 9:22:{s:02d} AM,5000" for s in range(20, 40)]
    lines += [",", ",", ","]
    p = tmp_path / "trailing_blanks.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Job Time"
    assert td.n == 23


def test_load_csv_low_fraction_with_no_fallback_raises_even_with_many_valid_rows(tmp_path):
    # H2b's replacement for the old ">=2 valid" exception: a column with a low VALID FRACTION
    # (most non-empty cells fail to parse) must raise even when the absolute valid COUNT is well
    # above 2, as long as no elapsed or clock fallback rescues it. Measured case: Liberty's
    # Anderson "DFIT-FINAL.csv" -- a plain sample-index column where ~9% of the integers happen
    # to fall inside the plausible Excel-serial range and misparse as real (but wrong) dates.
    lines = ["Index,Pressure(psi)"]
    # 9 non-date-shaped small integers (never in Excel-serial range) for every 1 that happens to
    # fall in range (32874-73051) -- an ~10% valid fraction, comfortably below the 50% threshold,
    # with more than 2 "valid" rows in absolute terms.
    for i in range(1, 41):
        val = 32900 + i if i % 10 == 0 else i
        lines.append(f"{val},{5000 - i}")
    p = tmp_path / "low_fraction_index.csv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        io_load.load_csv(str(p))


def test_load_csv_trailing_excel_ref_error_rows_do_not_fail_valid_fraction(tmp_path):
    # H2b's own fraction check must not punish a file for something that isn't really data at
    # all: a broken Excel formula reference ("#REF!") filling every cell of a large trailing
    # block. Measured case: Crestone Peak's "21011234 raw data.csv" has 233,470 genuinely valid,
    # contiguous "Date Time" rows followed by ~713,000 trailing "#REF!" rows -- but the real
    # Pressure/Temp columns keep reading real, continuing values through that trailing block
    # (this is real data missing a timestamp, not blank padding) -- "#REF!" (and the other
    # common Excel error sentinels) must be excluded from the non-empty-cell denominator the
    # same way a blank cell is, and the missing timestamps extrapolated at the good prefix's own
    # regular step (see the companion extrapolation test below).
    lines = ["Date Time,Pressure(psi)"]
    lines += [f"1/1/24 00:00:{s:02d},{5000 - s}" for s in range(20)]
    lines += [f"#REF!,{4980 - s}" for s in range(200)]
    p = tmp_path / "trailing_ref_error.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Date Time"
    # Extrapolated: all 220 rows end up with a usable timestamp, not just the 20 real ones.
    assert np.isfinite(td.t_s).sum() == 220
    np.testing.assert_allclose(td.t_s, np.arange(220, dtype=float))
    assert any("extrapolated" in w for w in td.load_warnings)


def test_load_csv_scattered_error_tokens_still_count_against_fraction(tmp_path):
    # The other half of the edge-run rule: a "#N/A"/"#REF!" cell SCATTERED through an otherwise-
    # unparseable column (not confined to one contiguous run at either end) must still count as
    # a non-empty, INVALID cell -- exempting it unconditionally would let a column that's mostly
    # error tokens pass the trust-fraction check outright, which is exactly backwards.
    lines = ["Index,Pressure(psi)"]
    for i in range(1, 41):
        if i % 10 == 0:
            lines.append(f"{32900 + i},{5000 - i}")  # the occasional genuine-looking value
        elif i % 3 == 0:
            lines.append(f"#N/A,{5000 - i}")  # scattered, not an edge run
        else:
            lines.append(f"{i},{5000 - i}")
    p = tmp_path / "scattered_na.csv"
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError):
        io_load.load_csv(str(p))


def test_load_csv_trailing_blank_time_with_real_data_extrapolates(tmp_path):
    # The extrapolation rule isn't specific to Excel error tokens -- a genuinely BLANK time cell
    # sitting in a trailing run that still carries real Pressure data gets the same treatment.
    lines = ["Date Time,Pressure(psi)"]
    lines += [f"1/1/24 00:00:{s:02d},{5000 - s}" for s in range(10)]
    lines += [f",{4990 - s}" for s in range(30)]
    p = tmp_path / "trailing_blank_time.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 40
    np.testing.assert_allclose(td.t_s, np.arange(40, dtype=float))


def test_load_csv_trailing_irregular_block_left_nan_with_warning(tmp_path):
    # When the surviving good run's own spacing is too irregular to trust, the trailing
    # data-bearing-but-timestamp-less rows are left as NaN (never guessed at a made-up spacing),
    # and a warning says so.
    lines = ["Date Time,Pressure(psi)"]
    # Irregular steps: 1s, 5s, 1s, 5s, ... -- median step exists but >1% of steps deviate from it.
    t = 0
    times = []
    for i in range(20):
        times.append(t)
        t += 1 if i % 2 == 0 else 5
    lines += [f"1/1/24 00:{s // 60:02d}:{s % 60:02d},{5000 - i}" for i, s in enumerate(times)]
    lines += [f"#REF!,{4980 - s}" for s in range(50)]
    p = tmp_path / "trailing_irregular.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 20
    assert any("no usable timestamp" in w for w in td.load_warnings)


def test_load_csv_trailing_gap_breaks_extrapolation_contiguity(tmp_path):
    # M3(a): a run of fully-empty rows (no value in ANY column, not just the timestamp) between
    # the last valid timestamp and a LATER stretch of real, data-bearing-but-timestamp-less rows
    # breaks contiguity -- only rows in an unbroken data-bearing run starting right at the last
    # valid timestamp are extrapolatable. Measured case: Great Western's "Seltzer Pump -
    # 036HN.csv" -- the row immediately after its last valid timestamp is already fully empty,
    # so its own real trailing data (60 rows, much later) is never reachable.
    lines = ["Time Stamp,Pressure(psi)"]
    lines += [f"1/1/24 00:00:{s:02d},{5000 - s}" for s in range(10)]
    lines += [",", ",", ","]  # a fully-empty gap: no value in ANY column
    lines += [f",{4990 - s}" for s in range(5)]  # real data, but past the gap
    p = tmp_path / "trailing_gap_breaks_contiguity.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 10
    np.testing.assert_allclose(td.t_s[:10], np.arange(10, dtype=float))
    assert any("5 data-bearing rows had no usable timestamp" in w for w in td.load_warnings)


def test_load_csv_edge_block_never_overwrites_a_readable_unparsed_cell(tmp_path):
    # M3(b): a cell that has real, readable (if wrong) text and simply failed to parse must
    # never be overwritten with a fabricated extrapolated value -- only a genuinely blank or
    # error-token cell is. Measured case: Extraction Oil & Gas's "Wake 33-20-13-...PRESSURE.csv"
    # opens with rows dated 1987 (a stuck default clock before the gauge was synced, now
    # rejected by the plausible-range guard) ahead of its real data -- readable, not blank, so
    # they must stay NaT with a "no usable timestamp" warning, not get a fabricated timestamp
    # extrapolated backward from the first good sample.
    lines = ["Date Time,Pressure(psi)"]
    lines += [f"7/25/1987 10:14:{s:02d},{14.0 + s * 0.01:.2f}" for s in range(5)]
    lines += [f"1/1/2024 00:00:{s:02d},{5000 - s}" for s in range(10)]
    p = tmp_path / "leading_readable_unparsed.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 10
    assert any("5 data-bearing rows had no usable timestamp" in w for w in td.load_warnings)
    assert not any("extrapolated" in w for w in td.load_warnings)


def test_load_csv_leading_units_declaration_row_not_data_bearing(tmp_path):
    # Bug fix: a units-declaration row right after the header (non-null text in every column,
    # but none of it numeric) must not be counted as "data-bearing" -- a plain notna() check
    # would wrongly count it and warn "1 data-bearing rows had no usable timestamp." Measured
    # case: Ballard's "Dilts 31-24 TH DFIT CSV Data.csv", row 2:
    # "(min) ,(date time), (psi), (bpm), (psi), (bbls)".
    lines = ["Elapsed (min),Date Time,Pressure (psi),Rate (bpm)"]
    lines.append("(min) ,(date time), (psi), (bpm)")
    lines += [f"{s},1/1/24 00:00:{s:02d},{5000 - s},2.0" for s in range(10)]
    p = tmp_path / "leading_units_row.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 10
    assert not any("no usable timestamp" in w for w in td.load_warnings)
    assert not any("extrapolated" in w for w in td.load_warnings)


def test_load_csv_leading_bare_unit_string_row_not_extrapolated(tmp_path):
    # Bug fix, the other shape: a units row that's blank cells plus a single bare unit string
    # (non-null in that one column, but not numeric) must not get a fabricated timestamp
    # extrapolated one median step before the first real sample. Measured case: Black Hills'
    # "Cope 107 -108 16HS BHP Fracpro.CSV", whose units row is
    # "                 ,               psi ".
    lines = ["Date Time,Pressure(psi)"]
    lines.append("           ,          psi ")
    lines += [f"1/1/24 00:00:{s:02d},{5000 - s}" for s in range(10)]
    p = tmp_path / "leading_bare_unit_string_row.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == 10
    assert not any("extrapolated" in w for w in td.load_warnings)
    assert not any("no usable timestamp" in w for w in td.load_warnings)


# --------------------------------------------------------------------------------------------------
# L1 -- date/companion pairing is independent of column order (a bare "Time" column before its
# real "Date" pair must not win just by coming first).
# --------------------------------------------------------------------------------------------------
def test_load_csv_time_before_date_column_order_independent(tmp_path):
    # suggest_channels' find() takes the first column-order match -- "Time" before "Date" -- but
    # _datetime_column_candidates correctly identifies "Date" as the one real candidate (with
    # "Time" reserved as its FIX-A companion). load_csv must switch to "Date" even though there's
    # only one candidate to consider (previously only multi-candidate cases triggered a switch).
    lines = [
        "Time,Date,Pressure(psi)",
        "8:23:17,9/8/2022,100",
        "8:23:29,9/8/2022,101",
        "8:23:40,9/8/2022,102",
        "8:23:50,9/8/2022,103",
        "8:24:00,9/8/2022,104",
    ]
    p = tmp_path / "time_before_date.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Date"
    np.testing.assert_allclose(td.t_s, [0.0, 12.0, 23.0, 33.0, 43.0])


# --------------------------------------------------------------------------------------------------
# FIX B -- broader numeric-time-column fallback (_find_numeric_time_column)
#
# Measured case: a Fracpro ASCII export whose only time-ish column is bare elapsed time with no
# "delta"/"elapsed" in its name at all (_find_elapsed_column's stricter rule never matches it).
# Before the Fallback-1 serial-range guard, a file like this silently misparsed as ~1900 dates;
# guarded, it correctly reads as no date at all and used to hard-fail with "Could not parse any
# datetimes" -- this fallback resolves the column's unit (header suffix -> units row -> name
# token -> bare-"Time"-assume-minutes) and loads it as elapsed time instead.
# --------------------------------------------------------------------------------------------------
def test_load_csv_time_header_suffix_seconds(tmp_path):
    p = tmp_path / "time_sec.csv"
    p.write_text("Time (sec),Pressure (psi)\n0.0,5000.0\n1.0,4995.0\n2.0,4990.0\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, [0.0, 1.0, 2.0])
    assert td.load_warnings == []


def test_load_csv_time_name_token_seconds(tmp_path):
    p = tmp_path / "time_us_sec.csv"
    p.write_text("Time_sec,Pressure (psi)\n0.0,5000.0\n1.0,4995.0\n2.0,4990.0\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, [0.0, 1.0, 2.0])


def test_load_csv_bare_minutes_name_no_needle_match(tmp_path):
    # "Minutes" contains none of suggest_channels' own datetime needles ("time"/"date"/...), so
    # it's never a multi-candidate; load_csv falls back to it as the first column, and the unit
    # resolves from the name token alone.
    p = tmp_path / "minutes.csv"
    p.write_text("Minutes,Pressure (psi)\n0.0,5000.0\n1.0,4995.0\n2.0,4990.0\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, [0.0, 60.0, 120.0])
    assert td.load_warnings == []


def test_load_csv_units_declaration_row(tmp_path):
    # Badger 25N-3HZ shape: a literal "(min) , " row right after the header, read as an ordinary
    # (garbage) first data row rather than a header. No header suffix, no name token -- the unit
    # comes only from this row.
    p = tmp_path / "unit_row.csv"
    p.write_text("Time,Pressure (psi)\n(min) , \n0.0,5000.0\n1.0,4995.0\n2.0,4990.0\n")
    td = io_load.load_csv(str(p))
    assert np.isnan(td.t_s[0])
    np.testing.assert_allclose(td.t_s[1:], [0.0, 60.0, 120.0])
    assert td.load_warnings == []


def test_load_csv_bare_time_assumes_minutes_with_warning(tmp_path):
    # No header suffix, no units row, no name token anywhere -- a bare "Time" header falls back to
    # assuming minutes (the plain Fracpro ASCII convention), but only with an explicit
    # load_warnings entry, since it's a guess.
    p = tmp_path / "bare_time.csv"
    p.write_text("Time,Pressure (psi)\n0.0,5000.0\n1.0,4995.0\n2.0,4990.0\n")
    td = io_load.load_csv(str(p))
    np.testing.assert_allclose(td.t_s, [0.0, 60.0, 120.0])
    assert len(td.load_warnings) == 1
    assert "minute" in td.load_warnings[0].lower()


def test_load_csv_delta_hrs_path_still_preferred_over_broader_fallback(tmp_path):
    # Keep the existing, stricter delta/elapsed behavior intact: when _find_elapsed_column
    # already finds a usable column, the broader numeric-time-column fallback is never consulted.
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
    assert td.load_warnings == []
    assert td.t_s[-1] / 3600 == pytest.approx(291.2705556, abs=1e-3)


# --------------------------------------------------------------------------------------------------
# _best_datetime_column -- always score the default; exclude a bare time-of-day-only candidate
# --------------------------------------------------------------------------------------------------
def test_best_datetime_column_scores_default_even_when_not_a_candidate(tmp_path):
    # "Clock Time" (pure time-of-day) is suggest_channels' own first-match default; "Timestamp"
    # (real dates, with a couple of blanks) must still win outright, even though the default
    # itself is never excluded from being scored (issue: an unscored default used to read as 0.0
    # and lose to literally any candidate).
    rows = []
    for i in range(10):
        ts = f"1/1/2024 08:{i:02d}:00" if i not in (3, 7) else ""
        clock = f"8:{i:02d}:00 AM"
        rows.append(f"{clock},{ts}")
    guess = io_load.suggest_channels(["Clock Time", "Timestamp"])
    assert guess["datetime"] == "Clock Time"
    p = tmp_path / "tod.csv"
    p.write_text("Clock Time,Timestamp\n" + "\n".join(rows) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Timestamp"


def test_best_datetime_column_excludes_pure_time_of_day_candidate(tmp_path):
    # Same shape, opposite column order: the real datetime column ("Timestamp") is
    # suggest_channels' own default here, and the bare time-of-day column ("Clock Time") must not
    # be able to steal the win just because dateutil silently defaults its missing date to today.
    rows = []
    for i in range(10):
        ts = f"1/1/2024 08:{i:02d}:00" if i not in (3, 7) else ""
        clock = f"8:{i:02d}:00 AM"
        rows.append(f"{ts},{clock}")
    p = tmp_path / "tod2.csv"
    p.write_text("Timestamp,Clock Time\n" + "\n".join(rows) + "\n")
    td = io_load.load_csv(str(p))
    assert td.datetime_col == "Timestamp"


# --------------------------------------------------------------------------------------------------
# pressure-channel ranking (header scan of 2,476 corpus files, see io_load._pressure_tier)
# --------------------------------------------------------------------------------------------------
def test_pressure_ranking_surf_beats_pump():
    cols = ["DateTime", "Pump Press", "Surf Press [Csg]"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surf Press [Csg]"


def test_pressure_ranking_surf_beats_add():
    cols = ["DateTime", "Add Pressure Chan1", "Surf Press [Csg]"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surf Press [Csg]"


def test_pressure_ranking_surface_beats_treating():
    cols = ["DateTime", "Treating Pressure", "Surface Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surface Pressure"


def test_pressure_ranking_treating_beats_generic():
    cols = ["DateTime", "Pressure", "Treating Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Treating Pressure"


def test_pressure_ranking_ann_demoted_below_tbg():
    # "Surf Press [Ann]" carries the "ann" demote token alongside "surf" -- demotion overrides,
    # so the [Tbg] column (no demote token) wins even though both are otherwise tier 0.
    cols = ["DateTime", "Surf Press [Ann]", "Surf Press [Tbg]"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surf Press [Tbg]"


def test_pressure_ranking_generic_beats_two_demoted_add_columns():
    cols = ["DateTime", "Add Pressure Chan1", "Add Pressure Channel 2", "Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Pressure"


def test_pressure_ranking_bottomhole_temp_not_picked_as_pressure():
    # BUG fix: the old find("bhp", "bottom") substring rule matched "Bottomhole Temp" and picked
    # a temperature channel as pressure. It must never be a candidate at all now.
    cols = ["DateTime", "Bottomhole Temp", "Treating Pressure", "Surf Press [Csg]"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surf Press [Csg]"
    assert guess["pressure_is_bhp"] is False


def test_pressure_ranking_bhp_named_column_wins_outright():
    cols = ["DateTime", "Surf Press [Csg]", "Bottomhole Press"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Bottomhole Press"
    assert guess["pressure_is_bhp"] is True


def test_pressure_ranking_bare_treating_and_surface_columns():
    # Bare columns named exactly "Treating"/"Surface" (no "press"/"psi" substring) are still
    # pressure candidates.
    cols = ["Time Stamp", "Total Rate", "Treating", "Surface"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surface"


def test_pressure_ranking_all_demoted_falls_back_to_column_order():
    cols = ["DateTime", "Max PSI", "Average Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Max PSI"


def test_pressure_ranking_channel_token_does_not_substring_match_ann():
    # "Channel" contains the substring "ann" but must not token-match the "ann" demote rule.
    cols = ["DateTime", "Pressure Channel 1", "Surf Press"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Surf Press"

    cols2 = ["DateTime", "Pressure Channel 1"]
    guess2 = io_load.suggest_channels(cols2)
    assert guess2["pressure"] == "Pressure Channel 1"


def test_pressure_ranking_no_candidates_is_none():
    cols = ["DateTime", "Rate", "Volume"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] is None
    assert guess["pressure_is_bhp"] is False


def test_pressure_ranking_bhp_wins_even_when_demoted_by_calc():
    # BUG fix: "Calc BHP" is the sole candidate and is demoted (tier 3, "calc"), so it never
    # reaches the bhp-outright branch -- but pressure_is_bhp must still read True, since the
    # chosen column IS named BHP. It no longer tracks which branch picked it.
    cols = ["DateTime", "Calc BHP"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Calc BHP"
    assert guess["pressure_is_bhp"] is True


def test_pressure_ranking_bhp_prefix_token_wins_outright():
    # "bhp" matches as a token PREFIX now, not just an exact token, so "BHP1" counts too.
    cols = ["DateTime", "Surf Press", "BHP1"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "BHP1"
    assert guess["pressure_is_bhp"] is True


def test_pressure_ranking_btmh_no_longer_bhp_named():
    # The "btmh" rule is removed: a Fracpro "Meas'd Btmh Press" export reads as a computed name,
    # not a direct BHP label, so it ranks as generic (tier 2) and loses to Treating (tier 1).
    cols = ["DateTime", "Meas'd Btmh Press", "Treating Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Treating Pressure"
    assert guess["pressure_is_bhp"] is False


def test_pressure_ranking_down_hole_not_demoted():
    # "down" was removed from the demote list ("Down Hole Pressure" should not be demoted, only
    # pump-down via "pump" is) -- both columns are tier 2, so column order decides.
    cols = ["DateTime", "Down Hole Pressure", "Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Down Hole Pressure"


def test_pressure_ranking_maximum_is_demoted():
    cols = ["DateTime", "Maximum Pressure", "Pressure"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "Pressure"


def test_pressure_ranking_whp_and_well_head_are_tier0():
    # A "whp" token PREFIX (not just an exact token) counts, so "WHP1" qualifies; and "well" +
    # "head" as two adjacent tokens (from a space) is recognized as wellhead too.
    cols = ["DateTime", "Treating Pressure", "WHP1"]
    guess = io_load.suggest_channels(cols)
    assert guess["pressure"] == "WHP1"

    cols2 = ["DateTime", "Treating Pressure", "Well Head Pressure"]
    guess2 = io_load.suggest_channels(cols2)
    assert guess2["pressure"] == "Well Head Pressure"


# --------------------------------------------------------------------------------------------------
# pressure liveness filter (io_load._live_pressure_candidates)
#
# Corpus check of 109 Treating->Surface name-ranking switches: 97 actually picked a DEAD
# "Surface Pressure" channel over a live "Treating Pressure" one.
# --------------------------------------------------------------------------------------------------
def _column_fn(data: dict):
    """A dict-backed stand-in for TestData.column, for suggest_channels' `column` kwarg."""
    def column(name):
        return data[name]
    return column


def test_pressure_liveness_dead_surface_loses_to_live_treating():
    cols = ["DateTime", "Surface Pressure", "Treating Pressure"]
    data = {
        "Surface Pressure": np.full(50, 240.0),
        "Treating Pressure": np.concatenate(
            [np.linspace(0.0, 8000.0, 20), np.linspace(8000.0, 3000.0, 30)]
        ),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Treating Pressure"


def test_pressure_liveness_all_zero_wellhead_loses_to_live_gauge():
    cols = ["DateTime", "Wellhead Pressure", "Pressure Gauge 245"]
    data = {
        "Wellhead Pressure": np.zeros(50),
        "Pressure Gauge 245": np.linspace(500.0, 5000.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Pressure Gauge 245"


def test_pressure_liveness_constant_max_loses_to_live_average():
    cols = ["DateTime", "Max PSI", "Average Pressure"]
    data = {
        "Max PSI": np.full(50, 9191.0),
        "Average Pressure": np.concatenate(
            [np.linspace(0.0, 6000.0, 25), np.linspace(6000.0, 2000.0, 25)]
        ),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Average Pressure"


def test_pressure_liveness_both_live_ranking_unchanged():
    cols = ["DateTime", "Pump Press", "Surf Press [Csg]"]
    data = {
        "Pump Press": np.linspace(1000.0, 9000.0, 50),
        "Surf Press [Csg]": np.linspace(500.0, 8500.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Surf Press [Csg]"


def test_pressure_liveness_all_dead_falls_back_to_name_ranking():
    cols = ["DateTime", "Pump Press", "Surf Press [Csg]"]
    data = {
        "Pump Press": np.full(50, 15.0),
        "Surf Press [Csg]": np.full(50, 20.0),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Surf Press [Csg]"


def test_pressure_liveness_raising_column_treated_dead():
    cols = ["DateTime", "Surf Press [Csg]", "Treating Pressure"]

    def column(name):
        if name == "Surf Press [Csg]":
            raise KeyError(name)
        return np.linspace(0.0, 8000.0, 50)

    guess = io_load.suggest_channels(cols, column=column)
    assert guess["pressure"] == "Treating Pressure"


# --------------------------------------------------------------------------------------------------
# pressure liveness filter -- header-unit scaling, outlier resistance, subsampling
# --------------------------------------------------------------------------------------------------
def test_pressure_liveness_kpa_header_factor_prevents_false_disqualify():
    # Same underlying signal, once in a psi-headed column and once in a kPa-headed one. Without
    # applying each column's own header-suffix factor first, the kPa column's raw magnitude
    # (~6.9x the psi column's) would dwarf the psi column's, failing the psi column's relative-
    # range check and excluding it from "live" -- even though it's the exact same real signal.
    psi_vals = np.linspace(4550.0, 9550.0, 50)
    kpa_vals = psi_vals / units.KPA_TO_PSI
    cols = ["DateTime", "Surf Press [Csg] (psi)", "Surf Press [Tbg] (kPa)"]
    data = {
        "Surf Press [Csg] (psi)": psi_vals,
        "Surf Press [Tbg] (kPa)": kpa_vals,
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Surf Press [Csg] (psi)"


def test_pressure_liveness_mpa_header_factor_fixes_absolute_floor():
    # An all-MPa file: raw MPa magnitudes (~0-60) sit far below the 100-psi absolute floor, so
    # without scaling, EVERY candidate (dead or live) fails it and the filter silently falls back
    # to name-only ranking, which would pick the dead "Surface Pressure" (tier 0) over the live,
    # demoted "Pump Pressure" (tier 3). With the header factor applied, the live channel's real
    # ~8700 psi range clears the floor and wins.
    cols = ["DateTime", "Surface Pressure (MPa)", "Pump Pressure (MPa)"]
    data = {
        "Surface Pressure (MPa)": np.full(50, 1.6),
        "Pump Pressure (MPa)": np.linspace(0.0, 60.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Pump Pressure (MPa)"


def test_pressure_liveness_relative_range_only_failure():
    # B's p99 matches A's (passes the relative-p99 rule) but its range (200) is well under 25%
    # of A's (8000) -- isolates the relative-RANGE rule as the sole reason B loses.
    cols = ["DateTime", "Pressure A", "Pressure B"]
    data = {
        "Pressure A": np.linspace(0.0, 8000.0, 50),
        "Pressure B": np.linspace(7800.0, 8000.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Pressure A"


def test_pressure_liveness_relative_p99_only_failure():
    # B's range (~3100) clears 25% of A's (~8800, passes the relative-range rule) because it
    # spans mostly-negative values, but its p99 (~170) is well under 30% of A's (~8900) --
    # isolates the relative-P99 rule as the sole reason B loses.
    cols = ["DateTime", "Pressure A", "Pressure B"]
    data = {
        "Pressure A": np.linspace(0.0, 9000.0, 50),
        "Pressure B": np.linspace(-3000.0, 200.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Pressure A"


def test_pressure_liveness_mostly_nan_sentinel_does_not_disable_filter():
    # "Sensor Pressure X" is 90% NaN with a handful of huge sentinel values -- it fails the
    # absolute finite-fraction check, so per the fix it must NOT contribute to max_range/max_p99;
    # if it did, its huge nominal range would fail Treating's relative checks and let dead
    # Surface (or nothing) win instead.
    cols = ["DateTime", "Surface Pressure", "Treating Pressure", "Sensor Pressure X"]
    treating = np.concatenate([np.linspace(0.0, 8000.0, 20), np.linspace(8000.0, 3000.0, 30)])
    sentinel = np.full(50, np.nan)
    sentinel[:5] = 999999.0
    data = {
        "Surface Pressure": np.full(50, 240.0),
        "Treating Pressure": treating,
        "Sensor Pressure X": sentinel,
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Treating Pressure"


def test_pressure_liveness_live_bhp_wins_outright_over_live_surf():
    cols = ["DateTime", "Surf Press [Csg]", "Bottomhole Press"]
    data = {
        "Surf Press [Csg]": np.linspace(4000.0, 9000.0, 50),
        "Bottomhole Press": np.linspace(4200.0, 9200.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Bottomhole Press"
    assert guess["pressure_is_bhp"] is True


def test_pressure_liveness_dead_bhp_loses_to_live_surf():
    cols = ["DateTime", "Surf Press [Csg]", "Bottomhole Press"]
    data = {
        "Surf Press [Csg]": np.linspace(4000.0, 9000.0, 50),
        "Bottomhole Press": np.full(50, 15.0),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Surf Press [Csg]"
    assert guess["pressure_is_bhp"] is False


def test_pressure_liveness_subsample_path_large_array():
    # Exercises the >LIVENESS_SUBSAMPLE_CAP stride path (250,000 samples -> stride 2).
    n = 250_000
    cols = ["DateTime", "Surface Pressure", "Treating Pressure"]
    data = {
        "Surface Pressure": np.full(n, 240.0),
        "Treating Pressure": np.linspace(0.0, 8000.0, n),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["pressure"] == "Treating Pressure"


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


# --------------------------------------------------------------------------------------------------
# unit detection (io_load.classify_pressure_magnitude / detect_channel_unit /
# refresh_unit_detection) -- see units.py and ../CLAUDE.md's Approach section.
# --------------------------------------------------------------------------------------------------
def _kpa_csv(tmp_path):
    """Strathcona shape: "CASING Pressure (KPAg)" header, no rate/volume."""
    p = tmp_path / "kpa.csv"
    p.write_text(
        "Date,CASING Pressure (KPAg)\n"
        "1/1/2024 00:00:00,90050\n"
        "1/1/2024 00:00:01,90040\n"
        "1/1/2024 00:00:02,90030\n"
    )
    return io_load.load_csv(str(p))


def test_header_suffix_kpag_detected_with_source_header(tmp_path):
    td = _kpa_csv(tmp_path)
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    det = td.unit_detections["CASING Pressure (KPAg)"]
    assert det.unit == "kpa"
    assert det.factor == pytest.approx(0.1450377377)
    assert det.source == "header"


def test_header_suffix_conversion_applied_lazily_via_column(tmp_path):
    td = _kpa_csv(tmp_path)
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    converted = td.column("CASING Pressure (KPAg)")
    np.testing.assert_allclose(converted, np.array([90050, 90040, 90030]) * 0.1450377377)
    # The raw df itself is never mutated -- lazy conversion, not eager.
    np.testing.assert_allclose(
        pd.to_numeric(td.df["CASING Pressure (KPAg)"]).to_numpy(dtype=float),
        [90050.0, 90040.0, 90030.0],
    )


@pytest.mark.parametrize("p99,expected_unit,expected_confidence", [
    (5000.0, "psi", "high"),
    (17000.0, "psi", "low"),
    (90000.0, "kpa", "high"),
])
def test_classify_pressure_magnitude_thresholds(p99, expected_unit, expected_confidence):
    values = np.full(20, p99)
    unit, confidence = io_load.classify_pressure_magnitude(values)
    assert unit == expected_unit
    assert confidence == expected_confidence


def test_classify_pressure_magnitude_too_few_finite_samples_is_psi_low():
    unit, confidence = io_load.classify_pressure_magnitude(np.array([np.nan, 5000.0]))
    assert unit == "psi"
    assert confidence == "low"


def test_detect_channel_unit_pressure_heuristic_no_header(tmp_path):
    p = tmp_path / "nohdr.csv"
    p.write_text("Date,Pressure\n1/1/2024 00:00:00,90050\n1/1/2024 00:00:01,90040\n")
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, "Pressure")
    det = td.unit_detections["Pressure"]
    assert det.unit == "kpa"
    assert det.source == "heuristic"
    assert det.confidence == "high"


def test_rate_inherits_metric_pressure_verdict(tmp_path):
    p = tmp_path / "metric.csv"
    p.write_text(
        "Date,Pressure,Rate\n"
        "1/1/2024 00:00:00,90050,6.0\n"
        "1/1/2024 00:00:01,90040,6.0\n"
    )
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, "Pressure", rate_col="Rate")
    rate_det = td.unit_detections["Rate"]
    assert rate_det.unit == "m3/min"
    assert rate_det.source == "inherited"
    assert rate_det.confidence == "low"
    np.testing.assert_allclose(td.column("Rate"), np.array([6.0, 6.0]) * units.M3_TO_BBL)


def test_rate_inherits_field_pressure_verdict(tmp_path):
    p = tmp_path / "field.csv"
    p.write_text(
        "Date,Pressure,Rate\n"
        "1/1/2024 00:00:00,5000,6.0\n"
        "1/1/2024 00:00:01,4995,6.0\n"
    )
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, "Pressure", rate_col="Rate")
    rate_det = td.unit_detections["Rate"]
    assert rate_det.unit == "bpm"
    assert rate_det.source == "inherited"
    np.testing.assert_allclose(td.column("Rate"), [6.0, 6.0])


def test_refresh_unit_detection_idempotent(tmp_path):
    td = _kpa_csv(tmp_path)
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    first = dict(td.unit_factors)
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    second = dict(td.unit_factors)
    assert first == second
    first_values = td.column("CASING Pressure (KPAg)")
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    second_values = td.column("CASING Pressure (KPAg)")
    np.testing.assert_allclose(first_values, second_values)


def test_override_bypasses_header_and_heuristic(tmp_path):
    td = _kpa_csv(tmp_path)
    io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)", pressure_unit="psi")
    det = td.unit_detections["CASING Pressure (KPAg)"]
    assert det.unit == "psi"
    assert det.factor == 1.0
    assert det.source == "override"


def test_low_confidence_heuristic_warns_even_at_factor_one(tmp_path):
    p = tmp_path / "ambiguous.csv"
    p.write_text("Date,Pressure\n1/1/2024 00:00:00,17000\n1/1/2024 00:00:01,17000\n")
    td = io_load.load_csv(str(p))
    warnings = io_load.refresh_unit_detection(td, "Pressure")
    assert any("ambiguous" in w.lower() for w in warnings)


def test_header_conversion_warns_with_factor_and_source(tmp_path):
    td = _kpa_csv(tmp_path)
    warnings = io_load.refresh_unit_detection(td, "CASING Pressure (KPAg)")
    assert len(warnings) == 1
    assert "kpa" in warnings[0].lower()
    assert "header" in warnings[0].lower()


def test_field_units_load_is_silent(tmp_path):
    p = tmp_path / "field.csv"
    p.write_text("Date,Pressure (psi)\n1/1/2024 00:00:00,5000\n1/1/2024 00:00:01,4995\n")
    td = io_load.load_csv(str(p))
    warnings = io_load.refresh_unit_detection(td, "Pressure (psi)")
    assert warnings == []


# --------------------------------------------------------------------------------------------------
# finding 3 -- MPa/bar pressure must also trigger metric inheritance for rate/volume (not just
# kPa); inherited confidence is always "low"; a channel's own header suffix still wins.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("header,peak", [("MPa", 90.05), ("bar", 900.5)])
def test_rate_and_volume_inherit_mpa_and_bar_pressure(tmp_path, header, peak):
    p = tmp_path / "metric.csv"
    p.write_text(
        f"Date,Pressure ({header}),Rate,Volume\n"
        f"1/1/2024 00:00:00,{peak},6.0,100.0\n"
        f"1/1/2024 00:00:01,{peak - 0.01},6.0,100.0\n"
    )
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, f"Pressure ({header})", rate_col="Rate", volume_col="Volume")
    rate_det = td.unit_detections["Rate"]
    vol_det = td.unit_detections["Volume"]
    assert rate_det.unit == "m3/min"
    assert rate_det.source == "inherited"
    assert rate_det.confidence == "low"
    assert vol_det.unit == "m3"
    assert vol_det.confidence == "low"
    np.testing.assert_allclose(td.column("Rate"), np.array([6.0, 6.0]) * units.M3_TO_BBL)
    np.testing.assert_allclose(td.column("Volume"), np.array([100.0, 100.0]) * units.M3_TO_BBL)


def test_rate_inherited_field_confidence_is_low_not_high(tmp_path):
    # Finding 3: inherited confidence is always "low" (an inference from another channel's
    # verdict), even when it lands on field units.
    p = tmp_path / "field.csv"
    p.write_text(
        "Date,Pressure,Rate\n"
        "1/1/2024 00:00:00,5000,6.0\n"
        "1/1/2024 00:00:01,4995,6.0\n"
    )
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, "Pressure", rate_col="Rate")
    assert td.unit_detections["Rate"].confidence == "low"


def test_rate_own_header_suffix_beats_kpa_pressure_inheritance(tmp_path):
    # A rate column whose own header names bpm must stay bpm even under a metric pressure verdict
    # -- a channel's own header suffix always wins over inheritance.
    p = tmp_path / "mixed.csv"
    p.write_text(
        "Date,Pressure (KPAg),Rate (bpm)\n"
        "1/1/2024 00:00:00,90050,6.0\n"
        "1/1/2024 00:00:01,90040,6.0\n"
    )
    td = io_load.load_csv(str(p))
    io_load.refresh_unit_detection(td, "Pressure (KPAg)", rate_col="Rate (bpm)")
    rate_det = td.unit_detections["Rate (bpm)"]
    assert rate_det.unit == "bpm"
    assert rate_det.source == "header"
    np.testing.assert_allclose(td.column("Rate (bpm)"), [6.0, 6.0])


# --------------------------------------------------------------------------------------------------
# finding 4 -- refresh_unit_detection must not KeyError on a column name absent from td.df (a
# foreign picks JSON, or a folder-mode source switch that leaves a stale column name behind).
# --------------------------------------------------------------------------------------------------
def test_refresh_unit_detection_missing_column_no_raise_no_entry(tmp_path):
    td = _kpa_csv(tmp_path)
    warnings = io_load.refresh_unit_detection(
        td, "Pressure Not A Real Column", rate_col="Rate Also Missing",
        volume_col="Volume Also Missing")
    assert warnings == []
    assert td.unit_factors == {}
    assert td.unit_detections == {}


def test_refresh_unit_detection_missing_pressure_col_still_skips_rate_inheritance(tmp_path):
    # A missing pressure column means no pressure_detection to inherit from -- the rate/volume
    # branch itself is unaffected as long as it too is a real column, defaulting to field units.
    p = tmp_path / "onlyrate.csv"
    p.write_text("Date,Rate\n1/1/2024 00:00:00,6.0\n1/1/2024 00:00:01,6.0\n")
    td = io_load.load_csv(str(p))
    warnings = io_load.refresh_unit_detection(td, "Missing Pressure Col", rate_col="Rate")
    assert warnings == []
    assert "Missing Pressure Col" not in td.unit_detections
    rate_det = td.unit_detections["Rate"]
    assert rate_det.unit == "bpm"
    assert rate_det.source == "inherited"


def test_load_csv_trailing_block_extrapolates_through_subsecond_jitter(tmp_path):
    # Millisecond stamps wandering 0.87-1.0 s around a 1 s cadence are NOT irregular: the
    # drift-based regularity test (small least-squares residual, steps within 50% of the median)
    # accepts them and extrapolates the timestamp-less trailing block at the fitted slope.
    rng = np.random.default_rng(0)
    n_good, n_bad = 400, 60
    t = np.arange(n_good) + rng.uniform(-0.06, 0.06, n_good)
    t[0] = 0.0
    lines = ["Date Time,Pressure(psi)"]
    for i, s in enumerate(t):
        ms = int(round(s * 1000))
        lines.append(f"1/1/24 00:{ms // 60000:02d}:{(ms // 1000) % 60:02d}.{ms % 1000:03d},{5000 - i}")
    lines += [f"#REF!,{4000 - s}" for s in range(n_bad)]
    p = tmp_path / "jitter_trailing.csv"
    p.write_text("\n".join(lines) + "\n")
    td = io_load.load_csv(str(p))
    assert np.isfinite(td.t_s).sum() == n_good + n_bad
    assert td.t_s[-1] == pytest.approx(n_good + n_bad - 1, abs=1.0)
    assert not any("no usable timestamp" in w for w in td.load_warnings)


def test_load_csv_companion_time_with_padded_whitespace_still_joins(tmp_path):
    # The companion veto's anchored time-of-day match must ignore cell padding.
    p = tmp_path / "padded_time.csv"
    p.write_text(
        "Date, Time, Pressure\n"
        "04/10/2016, 12:00:00, 5000\n"
        "04/10/2016, 12:30:00, 4900\n"
        "04/10/2016, 14:46:35, 4800\n"
    )
    td = io_load.load_csv(str(p))
    assert td.t_s[-1] - td.t_s[0] == pytest.approx(9995.0)
