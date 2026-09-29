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
