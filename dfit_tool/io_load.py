"""CSV loading, datetime parsing, channel mapping, and surface->BHP conversion.

The loader is deliberately format-tolerant: DFIT exports carry a datetime column that is usually
``MM/DD/YYYY HH:MM:SS`` but occasionally leaks raw Excel serial numbers (e.g. ``43508.34097``) in
the long falloff tail. Both are parsed onto one elapsed-seconds time base.

Several other real-corpus CSV shapes are also handled in ``load_csv``: a separate ``Date`` +
``Time`` column pair (joined before parsing), day-first (Canadian) dates detected from the
column's own values rather than assumed, a two-line preamble before the real header row, a
fully reverse-chronological export, and a datetime column that is unusable (e.g. Excel-mangled)
but has a clean elapsed-time column to fall back on.

Two formats are handled: CSV (``load_csv``) and Fracpro's binary ``.DBS`` format (``load_dbs``).
``load`` dispatches on the file extension.
"""

from __future__ import annotations

import csv
import itertools
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import units

# Excel's day-zero (the epoch that already accounts for the 1900 leap-year bug).
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_EXCEL_EPOCH_NP = np.datetime64(_EXCEL_EPOCH.to_datetime64(), "us")
_UNIX_EPOCH_NP = np.datetime64("1970-01-01T00:00:00", "us")
_PRIMARY_DT_FORMAT = "%m/%d/%Y %H:%M:%S"
_PRIMARY_DT_FORMAT_DAYFIRST = "%d/%m/%Y %H:%M:%S"
# A second exact, vectorized fast path for the 12-hour AM/PM layout (e.g. Badger 25N-3HZ's
# "12/21/2017 9:22:20 AM"). Without this, a whole-file AM/PM column falls through both primary
# fast paths to Fallback 2's per-element dateutil inference below -- correct, but 10-100x slower
# on a multi-million-row column, and this is exactly the shape a multi-datetime-candidate file
# (see _best_datetime_column) can end up parsing twice.
_PRIMARY_DT_FORMAT_AMPM = "%m/%d/%Y %I:%M:%S %p"
_PRIMARY_DT_FORMAT_AMPM_DAYFIRST = "%d/%m/%Y %I:%M:%S %p"

# psi of hydrostatic head per (ppg * ft): the standard field-units constant.
PSI_PER_PPG_FT = 0.052


def _epoch_plus_seconds(epoch: np.datetime64, secs: np.ndarray) -> np.ndarray:
    """Vectorized ``epoch + secs`` (float seconds, NaN/inf allowed) as a ``datetime64[us]`` array.

    Replaces this module's three hot-path uses of
    ``pd.Timestamp(epoch) + pd.to_timedelta(secs, unit="s")`` (each followed by
    ``.astype("datetime64[us]")``): plain numpy datetime64 arithmetic over a large array is far
    cheaper than pandas' per-element ``Timedelta`` construction (profiled at ~0.7-2.0 s on
    200k-1.5M-sample files). This is not a naive ``round(secs * 1e6)`` -- that loses precision
    once ``secs`` exceeds a few million, since float64 only carries ~15-17 significant digits and
    the product can run past 1e15. Instead it mirrors pandas' own float->timedelta conversion
    (``cast_from_unit``) bit for bit: truncate ``secs`` into a whole-seconds ``base`` (exact, since
    ``base`` alone is always well within float64's exact-integer range for any real elapsed time)
    and a ``frac`` remainder in (-1, 1), round `frac` to 9 decimal digits (nanoseconds) the same
    way pandas does before combining, then assemble the nanosecond count from the two pieces
    separately -- so the precision loss that a single ``secs * 1e9`` multiplication would incur
    never happens. Verified bit-identical to the old pandas expression across >1e6 randomized
    values (uniform magnitudes from small fractional seconds up to ~1e9 s, plus NaN) in scratch
    testing. Non-finite entries become NaT, matching ``pd.to_timedelta(nan, unit="s")``.
    """
    secs = np.asarray(secs, dtype=np.float64)
    finite = np.isfinite(secs)
    # The int64 nanosecond assembly below wraps silently past ~9.22e9 s, and pandas raises on
    # +-inf. Any such input takes the original pandas expression, so it keeps HEAD's exact
    # behavior (correct far-out dates, or the same OverflowError) instead of a plausible wrong date.
    if np.isinf(secs).any() or (np.abs(secs[finite]) > 9.0e9).any():
        return (pd.Timestamp(epoch) + pd.to_timedelta(secs, unit="s")).astype(
            "datetime64[us]").to_numpy()
    base = np.trunc(secs)
    frac = np.round(secs - base, 9)
    us = np.zeros(secs.shape, dtype=np.int64)
    base_i = base[finite].astype(np.int64)
    frac_ns = (frac[finite] * 1e9).astype(np.int64)
    us[finite] = (base_i * 1_000_000_000 + frac_ns) // 1000
    out = np.empty(secs.shape, dtype="datetime64[us]")
    out[finite] = epoch + us[finite].astype("timedelta64[us]")
    out[~finite] = np.datetime64("NaT", "us")
    return out


# Below this parsed-valid fraction, the datetime column is treated as unusable and load_csv looks
# for an elapsed-time column to fall back on instead (FIX B). Measured: Goodnight_DFIT_data.csv's
# Excel-mangled "Date/Time" column parses 0.40 valid (419,040 / 1,048,575 -- dateutil silently
# misreads "MM:SS.0" as a time-of-day on today's date), while a genuinely good column measures
# 1.00. 0.5 sits between the two with margin on both sides.
_MIN_VALID_DT_FRACTION = 0.5

# Excel's own literal error-formula sentinels -- treated the same as a blank cell when deciding
# how much of a column is "real" data (see _non_empty_mask below), not as a value that ought to
# have parsed. Measured case: Crestone Peak's "21011234 raw data.csv" has 233,470 genuinely
# valid, contiguous "Date Time" rows followed by ~713,000 trailing "#REF!" rows (a broken Excel
# formula reference, not a reading of anything) -- a real, good ~64.85h DFIT record padded by
# spreadsheet corruption, not the ~9%-valid, no-real-data-at-all shape _MIN_VALID_DT_FRACTION's
# unconditional raise (see load_csv) exists to catch. Counting "#REF!" as non-empty would count
# every one of those 713,000 rows against the file, sinking a genuinely good record to the same
# ~25% fraction as Anderson's actual sample-index column -- excluding it (same as a blank cell)
# correctly reads this file as the ~100%-valid one it actually is.
_EMPTY_CELL_TOKENS = frozenset({
    "#ref!", "#n/a", "#value!", "#div/0!", "#name?", "#null!", "#num!",
})


def _non_empty_mask(s: pd.Series) -> pd.Series:
    """True where `s` (already read as a raw column) holds something other than a blank cell or
    an Excel literal error-formula sentinel (see _EMPTY_CELL_TOKENS) -- i.e. something that
    OUGHT to be a real value, whether or not it actually parses as a date.
    """
    stripped = s.astype("string").str.strip()
    return (
        stripped.notna()
        & (stripped.str.len() > 0)
        & (~stripped.str.lower().isin(_EMPTY_CELL_TOKENS))
    )

# Plausible-range bounds for parse_datetime's Fallback 1 (bare Excel serial numbers), in whole
# days since the Excel epoch. A DFIT is never logged before 1990 or after 2100, but a small bare
# number is frequently something else entirely -- elapsed minutes/hours, a sample index -- and
# without a range guard Fallback 1 happily maps it to a bogus date in 1900 (measured case: the
# Badger 25N-3HZ file's elapsed-minutes "Time" column, e.g. 85.00000 -> 1900-03-25). Values outside
# this range are left as NaT for Fallback 1 (Fallback 2's generic string parse still gets a chance
# at them, and correctly leaves a bare number like that as NaT too -- it has no date-like shape).
_EXCEL_SERIAL_MIN = 32874  # Excel serial for 1990-01-01
_EXCEL_SERIAL_MAX = 73051  # Excel serial for 2100-01-01

# The same plausible-date window, as Timestamps -- also applied to Fallback 2's dateutil result
# below (not just Fallback 1's serial parse). Measured case: Hodges 2CH DFIT.csv's ~1.05M-row
# "Date Time" column has two genuinely corrupted cells ("23 23:43:18", "4/19/227" -- a truncated
# date and a typo'd year), and dateutil's generic parser, asked to make its best guess, happily
# returns year-1 and year-227 timestamps for them rather than failing outright. Left unguarded,
# either one poisons any min/max computed over the column with a bogus multi-century span. A
# value dateutil returns outside this window is exactly as implausible for a DFIT as an
# out-of-window Excel serial, so it is rejected the same way -- left NaT, not a wrong date.
_PLAUSIBLE_DT_MIN = pd.Timestamp("1990-01-01")
_PLAUSIBLE_DT_MAX = pd.Timestamp("2100-01-01")


# --------------------------------------------------------------------------------------------------
# datetime parsing
# --------------------------------------------------------------------------------------------------
_LEADING_DM_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")


def _dayfirst_hint(s: pd.Series) -> bool:
    """Sniff day-first (D/M/Y) vs. the default month-first (M/D/Y) from a string date series.

    Looks only at values with a leading ``D/D/Y`` or ``D-D-Y`` triple where the first two
    components are 1-2 digits and the year is 2-4 digits, so a 4-digit ISO leading component
    (``"2024-12-06 ..."``) is never considered (its first component alone would need to match
    ``\\d{1,2}`` followed immediately by a separator, which "2024" never does). Decides in order,
    first rule to fire wins:

    1. Some value's first component exceeds 12 (impossible as a month) while the largest second
       component is still <=12 (still possible as a month) -- day-first, proven. Measured case:
       the Strathcona files mix unambiguous day-first dates like "15/8/2022" (month 15 doesn't
       exist) with ones like "9/8/2022" that parse silently *wrong* as month-first without this
       check.
    2. The symmetric proof: second component's max exceeds 12, first's does not -- month-first.
    3. No proof either way, but the year is constant across every matched value, the first
       component is *also* constant, and the second component varies over >=2 distinct values --
       month-first (a US-style file inside one month, e.g. "5/1", "5/2", "5/3": month constant,
       day incrementing).
    4. Same, but with first and second swapped -- day-first. This is the Strathcona `-rt.csv`
       case itself: a record short enough to sit inside one month (a DFIT always is) where the
       *day* increments (9, 10, 11, 12) and the *month* sits constant at 8 -- no component ever
       exceeds 12, so rules 1-2 give no evidence, but the constant/varying split does. If the
       year itself varies, rules 3-4 do not apply (there's no "inside one month" evidence to
       read), and this falls through to rule 5.
    5. Otherwise month-first -- today's default, kept when the record genuinely crosses a month
       boundary (or there's no matching evidence at all) and gives no signal either way.

    Runs the regex over deduplicated leading prefixes rather than the whole column: every rule
    below reads only ``max``/``nunique``/"any match" over the matched components, all of which
    are invariant under deduplication (a value repeated a million times contributes nothing a
    single copy doesn't), and the pattern only ever looks at the first 10 characters anyway. On a
    ~1 Hz multi-day record this collapses the regex extract from ~10^5-10^6 rows to a few hundred
    distinct day-stamps.
    """
    u = s.str.slice(0, 10).dropna().unique()
    m = pd.Series(u, dtype="string").str.extract(_LEADING_DM_RE)
    first = pd.to_numeric(m[0], errors="coerce")
    second = pd.to_numeric(m[1], errors="coerce")
    year = pd.to_numeric(m[2], errors="coerce")
    if first.notna().sum() == 0:
        return False
    max_first = first.max()
    max_second = second.max()

    # Rules 1-2: positive proof from an out-of-range component, independent of the year.
    if max_first > 12 and max_second <= 12:
        return True
    if max_second > 12 and max_first <= 12:
        return False

    # Rules 3-4: no proof, but a constant year plus a constant/varying split between the other
    # two components. Only the matched rows participate (year, first, and second are captured by
    # one shared regex match per row, so their notna sets already agree).
    first_valid = first.dropna()
    second_valid = second.dropna()
    year_valid = year.dropna()
    if year_valid.nunique() == 1:
        if first_valid.nunique() == 1 and second_valid.nunique() >= 2:
            return False
        if second_valid.nunique() == 1 and first_valid.nunique() >= 2:
            return True

    # Rule 5: no evidence either way.
    return False


def parse_datetime(series: pd.Series, reject_bare_clock: bool = True) -> pd.Series:
    """Parse a datetime column that may mix formatted strings and Excel serial numbers.

    Returns a tz-naive datetime64 Series. Any value that cannot be parsed becomes NaT.

    ``reject_bare_clock`` (default True) forces a bare time-of-day/clock string (see
    _TIME_OF_DAY_RE below) to NaT rather than trusting dateutil's silent default-to-today. The
    one caller that passes False is FIX A's companion-Time join in ``load_csv``: there, a
    date-less joined string comes from a genuinely blank Date field next to a real Time value
    (DEFECT 1a), and treating it as a valid same-day timestamp is the documented, intentional
    behavior for that specific join -- unlike a bare clock STANDING ALONE as its own datetime
    column (the Dressler-shaped case below), which this guard exists to catch.
    """
    s = series.astype("string").str.strip()
    dayfirst = _dayfirst_hint(s)
    fmt = _PRIMARY_DT_FORMAT_DAYFIRST if dayfirst else _PRIMARY_DT_FORMAT

    # Fast path: the primary formatted-string layout. Normalize to microsecond resolution so the
    # fallbacks below can be merged without lossy-cast errors (pandas 3.0 is unit-strict).
    dt = pd.to_datetime(s, format=fmt, errors="coerce").astype("datetime64[us]")

    # Fast path 2: the 12-hour AM/PM layout (see _PRIMARY_DT_FORMAT_AMPM above) -- still an exact,
    # vectorized strptime match, so a whole-file AM/PM column never has to fall through to the
    # much slower per-element dateutil inference in Fallback 2 below.
    still_na = dt.isna()
    if still_na.any():
        ampm_fmt = _PRIMARY_DT_FORMAT_AMPM_DAYFIRST if dayfirst else _PRIMARY_DT_FORMAT_AMPM
        ampm = pd.to_datetime(s, format=ampm_fmt, errors="coerce").astype("datetime64[us]")
        dt = dt.combine_first(ampm)

    # Fallback 1: bare Excel serial numbers (rounded to whole seconds; data is ~1 Hz). Restricted
    # to the still-NaT rows: pd.to_numeric is purely elementwise (one row's parse can never
    # depend on another's), so subsetting first and combining back is exactly equivalent to
    # running it over the whole column, just cheaper once most rows already parsed on the fast
    # path.
    still_na = dt.isna()
    if still_na.any():
        # Filled positionally on the full index (not the subset's), so combine_first needs no
        # label alignment -- a duplicate index label would otherwise raise.
        na_mask = still_na.to_numpy()
        as_num = pd.to_numeric(s[na_mask], errors="coerce")
        # Reject anything outside the plausible Excel-serial range (see _EXCEL_SERIAL_MIN/MAX
        # above) before it ever becomes a date -- an out-of-range value is left NaN here so
        # Fallback 2 gets a chance at it instead of a bogus 1900-ish (or far-future) timestamp.
        as_num = as_num.where(as_num.between(_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX))
        secs = np.round(as_num.to_numpy(dtype=float) * 86400.0)
        excel_vals = np.full(len(s), np.datetime64("NaT", "us"))
        excel_vals[na_mask] = _epoch_plus_seconds(_EXCEL_EPOCH_NP, secs)
        excel = pd.Series(excel_vals, index=s.index)
        dt = dt.combine_first(excel)

    # Fallback 2: flexible parse for anything still missing (other string layouts). Run at FULL
    # LENGTH, not reindexed to the still-NaT subset -- pd.to_datetime with no explicit `format`
    # guesses a shared strptime format from the array's first non-null value and, when that guess
    # succeeds, applies it via a fast compiled loop (a mismatching value simply becomes NaT, no
    # further cost); reindexing to a smaller, already-more-homogeneous-garbage subset can change
    # what gets guessed, and therefore change a row's OWN result even though dateutil parses each
    # row independently once the guess is fixed. Measured case: Strathcona's
    # 100-09-14-062-04W6-rt.csv has a corrupted row (a Date cell's leading day digits are simply
    # gone, joined with a real Time into "/10/2022 22:11:25") sitting among thousands of
    # well-formed "31/10/2022 22:11:25"-shaped neighbors -- parsed WITH those neighbors visible,
    # pandas guesses the neighbors' shared format and the corrupted row simply fails to match it
    # (NaT, correct); parsed ALONE, or reindexed into a subset dominated by other unresolved
    # oddities, dateutil's own single-value guess reads it as a plausible-but-wrong 2022-10-01.
    #
    # What must still happen BEFORE this call, not after, is blanking (to NA, not removing --
    # length must stay full so every remaining value keeps its real neighbors) every value that
    # could never legitimately become a NEW date through this fallback anyway:
    #
    # - A bare number: dateutil's generic inference happily reads "2024" as 2024-01-01, "1500.5"
    #   as 1500-05-01, or "20171221" as 2017-12-21 -- exactly the class of bug Fallback 1's
    #   plausible-serial-range guard exists to prevent, just reached by a different path. A
    #   purely numeric string gets exactly one sanctioned route to a date -- Fallback 1's
    #   Excel-serial parse above -- and anything that fallback already rejected (out of the
    #   plausible 1990-2100 range) must stay NaT here too, not get a second, unguarded attempt.
    # - A bare time-of-day/clock string (see _TIME_OF_DAY_RE above, e.g. "9:22:20 AM" or the
    #   Dressler-shaped "09:07.0") -- it has no date component at all, so whatever date dateutil
    #   defaulted it to (today, always) is not a reading of the file, just an artifact of when
    #   this code happened to run. Skippable (see the docstring's ``reject_bare_clock`` note) for
    #   FIX A's join, which relies on this exact defaulting for a blank Date value beside a real
    #   Time one.
    # - A value with no digit in it at all -- it cannot be a date or time under ANY
    #   interpretation, and dateutil's format-GUESS itself (not just the eventual per-row parse)
    #   can fail outright on one, which is worse than merely wasting time on it: pandas then gives
    #   up guessing ANY format and falls back to the slow, per-element dateutil.parser.parse() for
    #   EVERY remaining non-null value in the array, not just this one. Measured case: Horsetail
    #   07E-0636's 631k-row Job Time column carries a literal "(date time)" units-declaration row
    #   (misread as data, see FIX B) at the very start of the column -- left unmasked, the whole-
    #   column call above takes ~26s (the slow path, for one single bad row); masked, ~1.4s.
    #
    # Not blanking every row that's ALREADY resolved (dt.notna()) is deliberate too, and is
    # exactly what preserves Strathcona's good-neighbor context above -- blanking them would
    # reproduce the same "malformed row parsed with no real context" problem subsetting caused.
    # Blanking (rather than removing) is also free: a null entry is skipped immediately by
    # pd.to_datetime, with no format-guess or per-row parse attempt charged against it, so a
    # mostly-null full-length array costs the same as a compacted one of just its non-null values.
    still_na = dt.isna()
    if still_na.any():
        for_parse = s.copy()
        is_numeric = pd.to_numeric(s, errors="coerce").notna().to_numpy()
        for_parse = for_parse.where(~is_numeric, pd.NA)
        if reject_bare_clock:
            is_clock = s.str.match(_TIME_OF_DAY_RE).fillna(False).to_numpy(dtype=bool)
            for_parse = for_parse.where(~is_clock, pd.NA)
        has_digit = s.str.contains(r"\d", regex=True, na=False).to_numpy()
        for_parse = for_parse.where(has_digit, pd.NA)
        # A mix of tz-aware values with DIFFERENT UTC offsets (e.g. "-05:00" and "-06:00" either
        # side of a DST transition in the same SCADA export) makes plain pd.to_datetime raise
        # "Mixed timezones detected" outright -- even with errors="coerce" -- rather than coerce
        # per element, since it cannot represent non-uniform offsets in one array without a
        # common timezone. Retried with utc=True (which normalizes every tz-aware value to a
        # common UTC instant first) on that specific error only -- not tried first, since mixing
        # utc=True with naive/heterogeneous-shaped values elsewhere in the same array has been
        # observed to change what gets guessed for THOSE values too. This also has to survive a
        # candidate that ultimately LOSES the multi-candidate comparison in _best_datetime_column
        # (every candidate gets scored, including ones this call rejects) and must not crash the
        # whole load, or scoring, just for being evaluated.
        try:
            generic = pd.to_datetime(for_parse, errors="coerce", dayfirst=dayfirst)
        except ValueError as exc:
            if "Mixed timezones" not in str(exc):
                raise
            generic = pd.to_datetime(for_parse, errors="coerce", dayfirst=dayfirst, utc=True)
        # A value carrying an explicit UTC/offset marker (e.g. "2019-08-29 14:13:35Z") infers as
        # tz-AWARE, which `.astype("datetime64[us]")` refuses outright (pandas 3.0 raises rather
        # than silently dropping the offset). Every other value here (and everywhere else in this
        # module) is tz-naive by construction. Converting to a common UTC instant FIRST (the
        # utc=True retry above, when it ran) rather than just dropping each value's own offset in
        # place is what makes the resulting elapsed-time math correct across a DST transition --
        # a fixed, uniform offset cancels out in any later subtraction either way, but a
        # transition's real ~1-hour jump only shows up correctly once every value shares one
        # reference instant.
        if getattr(generic.dtype, "tz", None) is not None:
            generic = generic.dt.tz_localize(None)
        generic = generic.astype("datetime64[us]")
        # Reject anything dateutil returned outside the plausible 1990-2100 window too -- see
        # _PLAUSIBLE_DT_MIN/MAX above.
        in_range = (generic >= _PLAUSIBLE_DT_MIN) & (generic <= _PLAUSIBLE_DT_MAX)
        generic = generic.where(in_range.to_numpy(), np.datetime64("NaT", "us"))
        dt = dt.combine_first(generic)

    return dt


def elapsed_seconds(dt: pd.Series) -> np.ndarray:
    """Seconds elapsed from the first valid timestamp."""
    t0 = dt.dropna().iloc[0]
    return (dt - t0).dt.total_seconds().to_numpy(dtype=float)


# A companion Time-of-day column occasionally uses a colon, not a decimal point, before the
# milliseconds (e.g. "15:58:17:647"). Anchored full-match only -- a normal "15:58:17", an
# already-correct "15:58:17.647", and a coarser "8:23:17" must all pass through untouched.
_MS_COLON_RE = re.compile(r"^(\d{1,2}:\d{2}:\d{2}):(\d{1,3})$")


def _normalize_ms_colon(time_s: pd.Series) -> pd.Series:
    """Rewrite a ``HH:MM:SS:mmm`` time-of-day column to ``HH:MM:SS.mmm``.

    Measured case (DEFECT 1b): the Lucero Tahu files' companion ``Time`` column is shaped like
    ``15:58:17:647`` -- a colon where a decimal point belongs. Joined with the date column
    unmodified, that string matches neither the primary format, the Excel-serial fallback, nor
    dateutil, and every row becomes NaT. Only the LAST colon is swapped for a period (the regex
    is anchored on both ends), so this never touches a value of any other shape.
    """
    return time_s.str.replace(_MS_COLON_RE, r"\1.\2", regex=True)


# --------------------------------------------------------------------------------------------------
# channel detection
# --------------------------------------------------------------------------------------------------
_UNIT_RE = re.compile(r"\(([^)]*)\)")


def _unit_of(colname: str) -> Optional[str]:
    m = _UNIT_RE.search(colname)
    return m.group(1).strip().lower() if m else None


def _bare_name(colname: str) -> str:
    """Column name with any parenthesized unit suffix and surrounding whitespace stripped."""
    return _UNIT_RE.sub("", colname).strip().lower()


# --------------------------------------------------------------------------------------------------
# pressure-channel ranking
#
# A header scan of 2,476 corpus files showed the old "first substring match in column order"
# pressure pick losing to an auxiliary/aggregate channel in ~200 files -- "Pump Press" or "Add
# Pressure Chan1" picked over "Surf Press [Csg]", "Treating Pressure" over "Surface Pressure",
# "PS Triplex Pressure" over "Mainline Pressure" -- and, separately, the old BHP-guess substring
# "bottom" matching "Bottomhole Temp" and picking a temperature channel as pressure (15 files).
# The ranking below fixes both: token-based tier matching (so e.g. "ann" doesn't substring-match
# "channel") ranks a surface/wellhead-named channel above a generic one above a demoted
# aux/pump/annulus/max/avg/calc one, and a BHP-named channel (excluding one that is itself
# demoted) still wins outright, minus the temperature bug.
# --------------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[^a-z0-9]+")


def _tokens(colname: str) -> list[str]:
    """Lowercased alnum tokens, split on any run of non-alnum characters."""
    return [t for t in _TOKEN_RE.split(colname.lower()) if t]


# A bare (no "press"/"bhp"/"whp"/"psi" substring) column name is still a pressure candidate when
# every token is one of these -- the corpus's bare "Treating"/"Surface" columns.
_BARE_PRESSURE_WORDS = {"treating", "surface", "surf", "wellhead"}

# Tier 3: demoted aux/aggregate/derived pressure channels -- checked first, overrides tiers 0/1.
# "down" is deliberately absent: pump-down is already demoted via "pump", and a genuine
# "Down Hole Pressure" channel (a bare BHP-like reading, distinct from surface/wellhead) should
# not be.
_PRESSURE_DEMOTE_EXACT = {
    "add", "backside", "pump", "discharge", "max", "min", "avg", "average",
    "maximum", "minimum", "averaged", "boost", "triplex", "jet", "iron",
}
_PRESSURE_DEMOTE_PREFIXES = ("addtl", "addl", "addit", "aux", "ann", "hydr", "calc")


def _is_pressure_candidate(colname: str) -> bool:
    """A column is a pressure candidate if its name contains "press"/"bhp"/"whp"/"psi"
    (substring, the original rule) or its bare name is made up entirely of surface/treating/
    wellhead tokens (the bare-"Treating"/"Surface" corpus case). A temperature channel (a token
    starting with "temp", e.g. "Bottomhole Temp") is never a candidate, regardless of either
    rule.
    """
    tokens = _tokens(colname)
    if any(t.startswith("temp") for t in tokens):
        return False
    name = colname.lower()
    if any(n in name for n in ("press", "bhp", "whp", "psi")):
        return True
    bare_tokens = _tokens(_bare_name(colname))
    return bool(bare_tokens) and all(t in _BARE_PRESSURE_WORDS for t in bare_tokens)


def _is_bhp_named(colname: str) -> bool:
    """True for a column whose tokens name it explicitly as bottomhole pressure: a token starting
    with "bhp" (so "BHP", "BHP1" both count) or "bottom" ("Bottomhole Press"). No "btmh" rule --
    a Fracpro export's "Meas'd Btmh Press" reads as a computed/derived name, not a direct BHP
    label, so it is left to rank as a generic (tier 2) pressure channel instead.
    """
    return any(t.startswith("bhp") or t.startswith("bottom") for t in _tokens(colname))


def _is_wellhead_named(tokens: list[str]) -> bool:
    """True for a surface/WHP/wellhead token: "surf"/"surface" (prefix), "whp"/"whp1" (prefix),
    a bare "wellhead" token, or the two tokens "well" and "head" split adjacent by the
    tokenizer's own separator handling (e.g. "Well Head Pressure" -> ["well", "head",
    "pressure"])."""
    if any(t.startswith("surf") or t.startswith("whp") or t == "wellhead" for t in tokens):
        return True
    return any(tokens[i] == "well" and tokens[i + 1] == "head" for i in range(len(tokens) - 1))


def _pressure_tier(colname: str) -> int:
    """Rank a pressure candidate: 3 = demoted aux/pump/annulus/max/avg/calc (checked first,
    overrides 0/1), 0 = surface/WHP/wellhead, 1 = treating, 2 = everything else (generic
    "Pressure", "Line Pressure", "CASING Pressure (KPAg)", ...). Lower wins.
    """
    tokens = _tokens(colname)
    if any(t in _PRESSURE_DEMOTE_EXACT or t.startswith(_PRESSURE_DEMOTE_PREFIXES)
           for t in tokens):
        return 3
    if _is_wellhead_named(tokens):
        return 0
    if any(t.startswith("treat") for t in tokens):
        return 1
    return 2


# Liveness filter: name ranking alone still lets a dead/backside gauge or a locked-aggregate
# channel outrank a real signal just because it's named "Surface"/"Wellhead"/"Max". Corpus check
# of 109 Treating->Surface name-ranking switches found 97 of them actually picked a DEAD "Surface
# Pressure" channel over a live "Treating Pressure" one -- measured cases: Treating p99 8821 psi
# vs. Surface median 237 psi/p99 265 psi; a "Wellhead Pressure" reading all zero; a "Surface
# Pressure" sitting at roughly -20 psi (a disconnected/backside gauge). A constant channel like
# "Max PSI" = 9191 psi (locked at one aggregate value) is the same failure by shape rather than
# name. None of this is visible from the header alone, hence this optional value-based filter.
LIVENESS_MIN_FINITE_FRAC = 0.5
LIVENESS_MIN_RANGE_PSI = 100.0
LIVENESS_MIN_RANGE_FRAC_OF_MAX = 0.25
LIVENESS_MIN_P99_FRAC_OF_MAX = 0.3

# `column()` is called at load time, before `refresh_unit_detection` has ever run (unit_factors
# is empty), so stats here have to apply their own header-suffix pressure factor -- otherwise a
# kPa channel (raw values ~6.9x psi) either wins the relative-range/p99 bars it shouldn't, or an
# all-MPa file (raw values ~0.0069x psi) fails the absolute 100-psi floor outright and silently
# falls back to name-only ranking. Only the header suffix is consulted, never the magnitude
# heuristic (`classify_pressure_magnitude`) -- that needs its own full-column percentile call and
# would fight this filter's own ranking logic on an unlabeled file.
LIVENESS_SUBSAMPLE_CAP = 100_000


def _pressure_header_factor(colname: str) -> float:
    """The header-suffix pressure conversion factor for one column (e.g. a "(kPa)" suffix ->
    ``units.KPA_TO_PSI``), or ``1.0`` when there's no parenthesized suffix or it isn't a
    recognized pressure unit."""
    unit = _unit_of(colname)
    if unit is None:
        return 1.0
    hit = units.lookup_pressure(unit)
    return hit[1] if hit is not None else 1.0


def _pressure_candidate_stats(colnames: list[str], column) -> dict[str, Optional[tuple]]:
    """(p1, p99, finite_frac) per candidate, in psi, from ``column(name) -> np.ndarray`` scaled
    by that column's own header-suffix pressure factor (see ``_pressure_header_factor`` -- NOT
    the magnitude heuristic, and NOT `TestData.unit_factors`, which is empty this early). ``None``
    for a column where calling ``column`` raises (missing/unparseable) -- treated as dead, never
    live.

    A record can be ~10^6 raw rows and this runs on the UI thread at file-open time, so the
    array is strided down to at most ``LIVENESS_SUBSAMPLE_CAP`` points first (both the finite
    fraction and the percentiles are measured on that subsample, not the full column), and p1/p99
    come from one combined ``np.percentile(..., [1, 99])`` call rather than two.
    """
    stats: dict[str, Optional[tuple]] = {}
    for c in colnames:
        try:
            raw = np.asarray(column(c), dtype=float)
        except Exception:
            stats[c] = None
            continue
        factor = _pressure_header_factor(c)
        values = raw * factor if factor != 1.0 else raw
        stride = max(1, values.size // LIVENESS_SUBSAMPLE_CAP)
        if stride > 1:
            values = values[::stride]
        finite = values[np.isfinite(values)]
        finite_frac = finite.size / values.size if values.size else 0.0
        if finite.size == 0:
            stats[c] = (0.0, 0.0, finite_frac)
        else:
            p1, p99 = np.percentile(finite, [1, 99])
            stats[c] = (float(p1), float(p99), finite_frac)
    return stats


def _live_pressure_candidates(colnames: list[str], column) -> list[str]:
    """Restrict `colnames` to the ones that look like a real, live pressure signal, using
    ``column(name) -> np.ndarray`` (see ``_pressure_candidate_stats`` for the header-unit scaling
    and subsampling applied first). A candidate is "live" iff: its finite fraction is
    >= ``LIVENESS_MIN_FINITE_FRAC``; its (p99 - p1) range is >= ``LIVENESS_MIN_RANGE_PSI`` AND
    >= ``LIVENESS_MIN_RANGE_FRAC_OF_MAX`` of the largest range among candidates; and its p99 is
    >= ``LIVENESS_MIN_P99_FRAC_OF_MAX`` of the largest p99 among candidates (the last two are
    relative, not absolute, so this holds across a wide range of true pressures).

    The "largest range"/"largest p99" comparisons are taken only over candidates that already
    pass the two ABSOLUTE checks (finite fraction, 100-psi floor) -- a mostly-NaN or
    sentinel-heavy channel with a huge nominal range must not get to set the relative bar that
    disables the filter for everyone else.

    Falls back to returning `colnames` unchanged whenever nothing qualifies (a genuinely dead
    file, or every candidate raised) -- callers then get today's name-only ranking rather than an
    empty candidate list.
    """
    stats = _pressure_candidate_stats(colnames, column)
    baseline: dict[str, tuple] = {}
    for c in colnames:
        s = stats[c]
        if s is None:
            continue
        p1, p99, finite_frac = s
        rng = p99 - p1
        if finite_frac >= LIVENESS_MIN_FINITE_FRAC and rng >= LIVENESS_MIN_RANGE_PSI:
            baseline[c] = (p99, rng)
    if not baseline:
        return colnames
    max_range = max(rng for _, rng in baseline.values())
    max_p99 = max(p99 for p99, _ in baseline.values())
    live = [c for c, (p99, rng) in baseline.items()
            if rng >= LIVENESS_MIN_RANGE_FRAC_OF_MAX * max_range
            and p99 >= LIVENESS_MIN_P99_FRAC_OF_MAX * max_p99]
    return live if live else colnames


def suggest_channels(columns: list[str], column=None) -> dict[str, Optional[str]]:
    """Best-guess mapping of column names to roles.

    Returns keys: ``pressure``, ``rate``, ``volume``, ``pressure_is_bhp`` (bool guess),
    ``datetime``, and ``time``. Values are column names (or None). The UI presents these as
    defaults.

    ``time`` is the name of a companion time-of-day column that has to be joined with
    ``datetime`` before parsing (measured case: Strathcona's separate ``Date`` + ``Time``
    columns, where ``Date`` alone collapses ~11-second samples down to daily granularity). It is
    only ever set when ``datetime`` is date-like-but-not-time-like (its name contains "date" and
    not "time" -- so "Date/Time", "DateTime", and "Timestamp (MST)" never trigger it) and there
    is a separate column whose bare name is exactly "time".

    ``pressure`` is chosen from ranked candidates rather than the first substring match in
    column order: a column explicitly named as BHP (a token starting with "bhp" or "bottom")
    wins outright and sets ``pressure_is_bhp`` -- unless that same column is also demoted (see
    below) -- else the highest-ranked candidate wins, ties broken by column order. The rank
    order is surface/WHP/wellhead-named, then treating-named, then generic, then demoted
    aux/pump/annulus/max/avg/calc-type channels last. See ``_pressure_tier`` and
    ``_is_pressure_candidate``. ``pressure_is_bhp`` always matches whether the *chosen* column is
    BHP-named, independent of which branch chose it.

    `column`, when given (a callable `name -> np.ndarray`, e.g. a loaded ``TestData.column``),
    narrows the pressure candidates to the "live" ones first -- see ``_live_pressure_candidates``
    -- since a dead/backside gauge or a locked-constant aggregate channel can otherwise outrank a
    real signal on name alone. Left ``None``, selection is name-only (unchanged from before this
    filter existed).
    """
    lc = {c: c.lower() for c in columns}

    def find(*needles, avoid=()):
        for c in columns:
            name = lc[c]
            if any(n in name for n in needles) and not any(a in name for a in avoid):
                return c
        return None

    datetime_col = find(*_DATETIME_NAME_NEEDLES)
    # A rate/volume column can also contain "time"; don't misassign those as the datetime col.
    # (_DATETIME_NAME_NEEDLES/_DATETIME_NAME_AVOID are shared with load_csv's own
    # _datetime_column_candidates, defined further down this module -- see there.)
    if datetime_col and any(x in lc[datetime_col] for x in _DATETIME_NAME_AVOID):
        datetime_col = None

    time_col = None
    if datetime_col and "date" in lc[datetime_col] and "time" not in lc[datetime_col]:
        for c in columns:
            if c != datetime_col and _bare_name(c) == "time":
                time_col = c
                break

    pressure_candidates = [c for c in columns if _is_pressure_candidate(c)]
    if column is not None and pressure_candidates:
        pressure_candidates = _live_pressure_candidates(pressure_candidates, column)
    bhp_candidates = [c for c in pressure_candidates
                      if _is_bhp_named(c) and _pressure_tier(c) != 3]
    if bhp_candidates:
        pressure = bhp_candidates[0]
    elif pressure_candidates:
        order = {c: i for i, c in enumerate(columns)}
        pressure = min(pressure_candidates, key=lambda c: (_pressure_tier(c), order[c]))
    else:
        pressure = None
    pressure_is_bhp = _is_bhp_named(pressure) if pressure else False

    rate = find("rate", "bpm", "flow", "slurry")
    volume = find("vol", "bbl", avoid=("rate",))

    return {
        "datetime": datetime_col,
        "time": time_col,
        "pressure": pressure,
        "rate": rate,
        "volume": volume,
        "pressure_is_bhp": pressure_is_bhp,
    }


# --------------------------------------------------------------------------------------------------
# unit detection: header suffix -> magnitude heuristic -> per-channel override
# --------------------------------------------------------------------------------------------------
@dataclass
class UnitDetection:
    """One channel's resolved unit and the field-units conversion factor for it.

    ``source`` is one of ``"header"`` (a recognized parenthesized suffix in the column name),
    ``"heuristic"`` (pressure-only magnitude classification, see ``classify_pressure_magnitude``),
    ``"inherited"`` (rate/volume borrowing the file-level pressure verdict -- see
    ``detect_channel_unit``), or ``"override"`` (the analyst's per-channel unit dropdown).
    ``confidence`` is ``"high"`` or ``"low"``; only a low-confidence pressure heuristic call
    triggers its own warning even at factor 1.0 -- see ``_maybe_warn``.
    """
    unit: str
    factor: float
    source: str
    confidence: str


def classify_pressure_magnitude(values: np.ndarray) -> tuple[str, str]:
    """Guess psi vs. kPa purely from magnitude: the 99th percentile of finite samples (not the
    raw max, which the near-G=0 water-hammer spike or a stray outlier can blow out of proportion)
    against thresholds wide enough apart that a real DFIT never straddles them --
    Strathcona's kPa data peaks ~90,050 kPa (~13,060 psi-equivalent), a field-unit test tops out
    in the low thousands of psi.

    ``< 15,000`` -> psi/high; ``>= 20,000`` -> kpa/high; the 15-20k overlap band has no clean
    single-unit reading, so it defaults to psi/low (the historically assumed unit, kept for
    backward compatibility) rather than guessing; fewer than 2 finite samples is the same
    can't-tell case (psi/low)."""
    finite = values[np.isfinite(values)] if len(values) else np.asarray([], dtype=float)
    if finite.size < 2:
        return "psi", "low"
    p99 = float(np.percentile(finite, 99))
    if p99 < 15000.0:
        return "psi", "high"
    if p99 >= 20000.0:
        return "kpa", "high"
    return "psi", "low"


_UNIT_LOOKUP = {"pressure": units.lookup_pressure, "rate": units.lookup_rate,
                "volume": units.lookup_volume}
_UNIT_FACTORS = {"pressure": units.PRESSURE_FACTORS, "rate": units.RATE_FACTORS,
                 "volume": units.VOLUME_FACTORS}


def detect_channel_unit(colname: str, values: np.ndarray, kind: str, override: str,
                        pressure_detection: Optional[UnitDetection] = None) -> UnitDetection:
    """Resolve one channel's unit, in priority order: the analyst's per-channel override (when
    not ``"auto"``) -> the column name's parenthesized header suffix (``_unit_of``) -> a
    kind-specific fallback.

    ``kind`` is ``"pressure"``, ``"rate"``, or ``"volume"``. The pressure fallback is the
    magnitude heuristic (``classify_pressure_magnitude``). Rate and volume get no standalone
    heuristic -- 1 m3/min == 6.29 bpm, and the two unit ranges overlap with no clean threshold --
    so they instead inherit the file-level pressure verdict passed in as `pressure_detection`:
    any non-psi pressure unit (kpa, mpa, or bar) means metric rate/volume too (m3/min / m3),
    anything else means field units (bpm / bbl), always at low confidence (an inference from
    another channel's verdict, not direct evidence, whichever way it lands). A channel's own
    header suffix always wins over that inheritance.
    """
    lookup = _UNIT_LOOKUP[kind]
    if override != "auto":
        hit = lookup(override)
        if hit is not None:
            canon, factor = hit
            return UnitDetection(unit=canon, factor=factor, source="override", confidence="high")
    header_unit = _unit_of(colname)
    if header_unit is not None:
        hit = lookup(header_unit)
        if hit is not None:
            canon, factor = hit
            return UnitDetection(unit=canon, factor=factor, source="header", confidence="high")
    if kind == "pressure":
        unit, confidence = classify_pressure_magnitude(values)
        return UnitDetection(unit=unit, factor=_UNIT_FACTORS["pressure"][unit], source="heuristic",
                             confidence=confidence)
    # rate/volume: inherit the file-level pressure verdict rather than guess independently. Any
    # non-psi pressure unit (kpa, mpa, bar) means metric -- not just kpa, or an MPa- or bar-
    # headed file would leave an unlabeled rate/volume at a field-unit factor of 1.0 with no
    # warning. Always "low" confidence: this is an inference from another channel's verdict, not
    # direct evidence, whichever way it lands.
    metric = pressure_detection is not None and pressure_detection.unit != "psi"
    if kind == "rate":
        unit = "m3/min" if metric else "bpm"
    else:
        unit = "m3" if metric else "bbl"
    return UnitDetection(unit=unit, factor=_UNIT_FACTORS[kind][unit], source="inherited",
                         confidence="low")


def _maybe_warn(kind: str, det: UnitDetection) -> list[str]:
    """Warning policy for one channel's ``UnitDetection``: always warn when the resolved factor
    isn't 1.0 (a real conversion happened, silent would hide it); also warn on a low-confidence
    pressure heuristic call even at factor 1.0 (the magnitude call itself is uncertain, even
    though it landed on the assumed-default unit). Silent for a factor-1.0 inherited/header/
    override detection -- that's the ordinary field-units load, and warning on every load would
    just be noise."""
    label = kind.capitalize()
    if det.factor != 1.0:
        return [f"{label} converted from {det.unit} ×{det.factor:.6g} ({det.source})"]
    if kind == "pressure" and det.source == "heuristic" and det.confidence == "low":
        return [f"{label} magnitude ambiguous between psi and kPa; defaulted to psi — "
                "verify with the unit dropdown"]
    return []


# --------------------------------------------------------------------------------------------------
# data container + config
# --------------------------------------------------------------------------------------------------
@dataclass
class ChannelConfig:
    """How the loaded columns map to physical channels, plus BHP-conversion inputs."""

    pressure_col: str
    pressure_is_bhp: bool = False
    rate_col: Optional[str] = None
    volume_col: Optional[str] = None
    # Required only when pressure_is_bhp is False (surface pressure -> compute BHP):
    mw_ppg: Optional[float] = None
    tvd_ft: Optional[float] = None

    def needs_bhp_inputs(self) -> bool:
        return not self.pressure_is_bhp

    def bhp_inputs_ready(self) -> bool:
        return self.pressure_is_bhp or (self.mw_ppg is not None and self.tvd_ft is not None)


@dataclass
class TestData:
    """A loaded test: the raw frame plus an elapsed-seconds time base."""

    path: str
    df: pd.DataFrame
    datetime_col: str
    t_s: np.ndarray = field(repr=False)  # elapsed seconds from first sample
    columns: list[str] = field(default_factory=list)
    # Unit-conversion cache, rebuilt by refresh_unit_detection: {column name -> multiply-by-this
    # factor to reach field units}. Defaults empty -> factor 1.0 everywhere -> zero behavior
    # change for any caller that never invokes detection (see units.py's design note / ../CLAUDE.md).
    unit_factors: dict[str, float] = field(default_factory=dict, repr=False)
    unit_detections: dict[str, "UnitDetection"] = field(default_factory=dict, repr=False)
    unit_warnings: list[str] = field(default_factory=list, repr=False)
    # Load-time anomalies unrelated to unit detection (e.g. a .DBS file with trailing padding
    # records truncated by load_dbs -- see its docstring). Set once at load and never rebuilt,
    # unlike unit_warnings; model.compute_all folds these into DerivedResults.warnings every
    # recompute so they surface in the same panel.
    load_warnings: list[str] = field(default_factory=list, repr=False)

    @property
    def n(self) -> int:
        return len(self.df)

    def column(self, name: str) -> np.ndarray:
        """Raw numeric values for `name`, scaled by its cached unit factor (see unit_factors).

        Lazy, not eager: the underlying df is never mutated for unit conversion, so re-running
        refresh_unit_detection (an override change, a remap) always recomputes from the untouched
        raw data -- there is structurally no double-conversion risk.
        """
        raw = pd.to_numeric(self.df[name], errors="coerce").to_numpy(dtype=float)
        factor = self.unit_factors.get(name, 1.0)
        return raw * factor if factor != 1.0 else raw

    def pressure_surface(self, cfg: ChannelConfig) -> np.ndarray:
        return self.column(cfg.pressure_col)

    def bhp(self, cfg: ChannelConfig) -> np.ndarray:
        """Bottomhole pressure over all samples.

        If the mapped pressure channel is already BHP, it is returned unchanged. Otherwise it is
        treated as surface pressure and a constant hydrostatic head is added:

            BHP = WHP + 0.052 * mw_ppg * tvd_ft

        Hydrostatic only (no friction) -- valid post-shut-in, which is where every pick is made.
        """
        p = self.column(cfg.pressure_col)
        if cfg.pressure_is_bhp:
            return p
        if cfg.mw_ppg is None or cfg.tvd_ft is None:
            raise ValueError("Surface pressure selected but mw_ppg/tvd_ft not set for BHP conversion")
        return p + hydrostatic_head(cfg.mw_ppg, cfg.tvd_ft)


def hydrostatic_head(mw_ppg: float, tvd_ft: float) -> float:
    """Hydrostatic head in psi for a static fluid column (field units)."""
    return PSI_PER_PPG_FT * mw_ppg * tvd_ft


def refresh_unit_detection(td: TestData, pressure_col: Optional[str],
                           rate_col: Optional[str] = None, volume_col: Optional[str] = None,
                           pressure_unit: str = "auto", rate_unit: str = "auto",
                           volume_unit: str = "auto") -> list[str]:
    """Rebuild `td.unit_factors`/`td.unit_detections`/`td.unit_warnings` from the raw, untouched
    columns in `td.df` -- lazy detection, not eager conversion (see units.py's module docstring).
    Idempotent: calling this again with the same arguments recomputes the same factors from the
    same raw data every time, so there is no compounding risk from calling it on every recompute.

    Plain-string params (not a PickState) so io_load stays below model.py in the import graph --
    model.compute_all calls this at the top of every compute_all with the state's column names
    and unit overrides. `pressure_col`/`rate_col`/`volume_col` may be empty/None (channel not yet
    mapped); that channel is then simply skipped, leaving no key in the resulting dicts and no
    warning for it. A column name that doesn't exist in `td.df` (a foreign picks JSON, or a
    folder-mode source switch that leaves a stale column name in `PickState`) is skipped the same
    way rather than raising a `KeyError` -- it just gets no detection entry, so `column()` falls
    back to factor 1.0 for it. Returns the list of warnings from this detection pass --
    compute_all extends `DerivedResults.warnings` with it.
    """
    factors: dict[str, float] = {}
    detections: dict[str, UnitDetection] = {}
    warnings: list[str] = []

    pressure_detection: Optional[UnitDetection] = None
    if pressure_col and pressure_col in td.df.columns:
        raw = pd.to_numeric(td.df[pressure_col], errors="coerce").to_numpy(dtype=float)
        pressure_detection = detect_channel_unit(pressure_col, raw, "pressure", pressure_unit)
        factors[pressure_col] = pressure_detection.factor
        detections[pressure_col] = pressure_detection
        warnings.extend(_maybe_warn("pressure", pressure_detection))

    for col, override, kind in ((rate_col, rate_unit, "rate"), (volume_col, volume_unit, "volume")):
        if not col or col not in td.df.columns:
            continue
        raw = pd.to_numeric(td.df[col], errors="coerce").to_numpy(dtype=float)
        det = detect_channel_unit(col, raw, kind, override, pressure_detection=pressure_detection)
        factors[col] = det.factor
        detections[col] = det
        warnings.extend(_maybe_warn(kind, det))

    td.unit_factors = factors
    td.unit_detections = detections
    td.unit_warnings = warnings
    return warnings


# --------------------------------------------------------------------------------------------------
# elapsed-time-column fallback (FIX B)
# --------------------------------------------------------------------------------------------------
# Recognized elapsed-time units, keyed by the lowercased parenthesized suffix (e.g. "Delta(Hrs)"
# has unit "hrs"). No default unit is guessed when the suffix is missing or unrecognized.
_ELAPSED_UNIT_SECONDS = {
    "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
}


def _find_elapsed_column(df: pd.DataFrame) -> Optional[tuple[str, np.ndarray]]:
    """Find a usable elapsed-time column to fall back on when the datetime column is unusable.

    Measured case: Goodnight_DFIT_data.csv's ``Date/Time`` column was mangled by Excel into
    ``MM:SS.0`` strings that dateutil silently misreads as a time-of-day on today's date (see
    ``load_csv``'s valid-fraction check), but the file carries a clean, monotonic ``Delta(Hrs)``
    column. Candidates are columns whose name contains "delta" or "elapsed", excluding anything
    that also looks like a physical channel ("rate", "vol", "press", "temp" -- those can carry a
    "delta" in their own name, e.g. "Delta Pressure(psi)"). The unit must come from a
    parenthesized suffix recognized in ``_ELAPSED_UNIT_SECONDS``; a missing or unrecognized unit
    is rejected rather than guessed. The column itself must be numeric, with at least 2 finite
    values, non-decreasing across those finite values, and a strictly positive span. Returns the
    column name and its values converted to seconds, or None.
    """
    for col in df.columns:
        name_lc = col.lower()
        if not ("delta" in name_lc or "elapsed" in name_lc):
            continue
        if any(x in name_lc for x in ("rate", "vol", "press", "temp")):
            continue
        mult = _ELAPSED_UNIT_SECONDS.get(_unit_of(col))
        if mult is None:
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            continue
        if not np.all(np.diff(finite) >= 0):
            continue
        if not (finite[-1] - finite[0] > 0):
            continue

        return col, vals * mult

    return None


# Bare name-token unit hints (no parenthesized suffix), e.g. "Time_sec" -> tokens
# ["time", "sec"], "Minutes" -> ["minutes"]. Used only by the broader numeric-time-column
# fallback below (_find_numeric_time_column), not by _find_elapsed_column's stricter
# parenthesized-suffix-only rule above.
_ELAPSED_NAME_TOKEN_SECONDS = {
    "s": 1.0, "sec": 1.0, "secs": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "mins": 60.0, "minute": 60.0, "minutes": 60.0,
    "hr": 3600.0, "hrs": 3600.0, "hour": 3600.0, "hours": 3600.0,
}


def _row_unit_hint(cell: object) -> Optional[str]:
    """A recognized elapsed-unit token from one row's cell text, or None.

    Measured case: Badger 25N-3HZ's CSV carries a literal ``"(min) ,(date time), , ..."`` row
    right after the header -- a units-DECLARATION row that ``pd.read_csv`` reads as an ordinary
    (garbage, NaN-after-numeric-coercion) first DATA row, not a header, since it carries no
    special marker of its own. Matches either a parenthesized token (``"(min)"`` -> ``"min"``,
    reusing ``_unit_of``) or a bare unit word with no parens at all (``"min"``).
    """
    if not isinstance(cell, str):
        return None
    text = cell.strip().lower()
    paren = _unit_of(text)
    if paren in _ELAPSED_UNIT_SECONDS:
        return paren
    if text in _ELAPSED_UNIT_SECONDS:
        return text
    return None


def _resolve_elapsed_seconds_factor(col: str, df: pd.DataFrame) -> Optional[float]:
    """Resolve `col`'s elapsed-time unit to a seconds multiplier, in priority order: the
    column's own header parenthesized suffix (``"Time (sec)"``) -> a units-declaration row (see
    ``_row_unit_hint``, checked only at ``df`` row 0) -> a unit token anywhere in the column's
    name (``_tokens``, e.g. ``"Time_sec"`` -> ``"sec"``, ``"Minutes"`` -> ``"minutes"``). None
    when none of the three resolves a unit -- the caller (``_find_numeric_time_column``) decides
    whether a bare "Time" header is then worth guessing at.
    """
    header_unit = _unit_of(col)
    if header_unit in _ELAPSED_UNIT_SECONDS:
        return _ELAPSED_UNIT_SECONDS[header_unit]
    if len(df) > 0:
        row_unit = _row_unit_hint(df[col].iloc[0])
        if row_unit is not None:
            return _ELAPSED_UNIT_SECONDS[row_unit]
    for t in _tokens(col):
        if t in _ELAPSED_NAME_TOKEN_SECONDS:
            return _ELAPSED_NAME_TOKEN_SECONDS[t]
    return None


def _find_numeric_time_column(
    df: pd.DataFrame, dt_col: str
) -> Optional[tuple[str, np.ndarray, list[str]]]:
    """A broader FIX B fallback for when NO column parses as a datetime at all: a plain numeric,
    (mostly) non-decreasing elapsed-time column, even one named just ``"Time"`` with no
    "delta"/"elapsed" in its name at all (unlike ``_find_elapsed_column`` above, which this does
    not replace -- ``load_csv`` tries that first and only reaches this one when it finds
    nothing).

    Measured case: a Fracpro ASCII export whose only time-ish column is bare elapsed minutes
    (e.g. Encore 1C/26N/40N/41N, English Farms 15N-8HZ Pump Data) -- before the plausible-
    Excel-serial-range guard on ``parse_datetime``'s Fallback 1 (see there), a bare number like
    this was silently misread as a bogus ~1900 date; guarded, it correctly reads as no date at
    all, so this fallback exists to still load the file, correctly, as elapsed time, instead of
    raising.

    Candidates are `dt_col` itself (the column ``load_csv`` already tried and failed to parse as
    a datetime -- the common case for these single-time-column exports), followed by any other
    datetime-name-matching column (see ``_datetime_column_candidates``). The unit is resolved by
    ``_resolve_elapsed_seconds_factor``; failing that, a column whose bare name is exactly "time"
    (no other hint anywhere) is assumed to be MINUTES -- the plain Fracpro ASCII convention --
    with a returned warning, since this is a guess, not a read. Returns the column name, its
    values converted to seconds, and any such warning (folded into ``TestData.load_warnings``),
    or None.
    """
    candidates = [dt_col] + [
        c for c in _datetime_column_candidates(list(df.columns)) if c != dt_col
    ]
    for col in candidates:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size < 2:
            continue
        if not np.all(np.diff(finite) >= 0):
            continue
        if not (finite[-1] - finite[0] > 0):
            continue

        mult = _resolve_elapsed_seconds_factor(col, df)
        if mult is not None:
            return col, vals * mult, []
        if _bare_name(col) == "time":
            warn = (
                f"Column {col!r} has no recognized time unit (no header suffix, units row, or "
                f"unit token in its name) -- assuming MINUTES, the plain Fracpro ASCII "
                f"elapsed-time convention."
            )
            return col, vals * 60.0, [warn]

    return None


def _clock_seconds_of_day(s: pd.Series) -> pd.Series:
    """Seconds-since-midnight for a bare clock/time-of-day string (H:MM, H:MM:S, or H:MM:SS,
    optional AM/PM), or NaN when the string doesn't match ``_TIME_OF_DAY_RE`` at all, its
    hour/minute/second is out of range, or its shape is the Dressler-ambiguous "MM:SS.f" one --
    exactly two colon-separated fields plus a fraction, e.g. "27:32.7" -- indistinguishable from
    an "H:MM" reading with a fractional minute, so never treated as a usable clock value here
    (see ``_find_clock_column``, which relies on this exclusion to keep Dressler's own file
    correctly failing rather than landing on a wrong reading through this fallback instead).
    Hour range is 0-23 with no AM/PM marker, 1-12 with one (``12 AM`` -> midnight, ``12 PM`` ->
    noon, matching the ordinary 12-hour convention).
    """
    s = s.astype("string").str.strip()
    m = s.str.extract(_TIME_OF_DAY_RE)
    matched = m["h"].notna().to_numpy()
    h = pd.to_numeric(m["h"], errors="coerce").to_numpy(dtype=float)
    mi = pd.to_numeric(m["m"], errors="coerce").to_numpy(dtype=float)
    has_sec = m["s"].notna().to_numpy()
    sec = pd.to_numeric(m["s"], errors="coerce").to_numpy(dtype=float)
    has_frac = m["frac"].notna().to_numpy()
    frac = pd.to_numeric(m["frac"], errors="coerce").to_numpy(dtype=float)
    ampm = m["ampm"].str.upper().fillna("")
    has_ampm = ampm.to_numpy() != ""
    is_am = ampm.to_numpy() == "AM"
    is_pm = ampm.to_numpy() == "PM"

    ambiguous = (~has_sec) & has_frac
    with np.errstate(invalid="ignore"):
        hour_ok = np.where(has_ampm, (h >= 1) & (h <= 12), (h >= 0) & (h <= 23))
        min_ok = (mi >= 0) & (mi <= 59)
        sec_ok = (~has_sec) | ((sec >= 0) & (sec <= 59))
    ok = matched & ~ambiguous & hour_ok & min_ok & sec_ok

    h24 = np.where(is_am & (h == 12), 0.0, h)
    h24 = np.where(is_pm & (h24 != 12), h24 + 12.0, h24)
    sec_val = np.where(has_sec, sec, 0.0)
    frac_val = np.where(has_frac, frac, 0.0)
    total = h24 * 3600.0 + mi * 60.0 + sec_val + frac_val
    total = np.where(ok, total, np.nan)
    return pd.Series(total, index=s.index)


def _unwrap_midnight_rollovers(raw_secs: np.ndarray) -> np.ndarray:
    """Add 24h for every backward step of more than 12h in `raw_secs` (seconds-of-day, NaN
    allowed, as returned by ``_clock_seconds_of_day``).

    A clock-only column has no date to carry the day boundary, so a value that drops by more
    than half a day from the previous FINITE sample is read as the clock having wrapped past
    midnight, not as time running backward. NaN entries are skipped when looking for the
    previous sample (never treated as a step themselves) and stay NaN in the result.
    """
    raw_secs = np.asarray(raw_secs, dtype=float)
    out = np.full(raw_secs.shape, np.nan)
    valid = np.isfinite(raw_secs)
    vals = raw_secs[valid]
    if vals.size:
        diffs = np.diff(vals)
        rollover = diffs < -12 * 3600.0
        offset = np.concatenate([[0.0], np.cumsum(np.where(rollover, 24 * 3600.0, 0.0))])
        out[valid] = vals + offset
    return out


def _find_clock_column(
    df: pd.DataFrame, dt_col: str
) -> Optional[tuple[str, np.ndarray, list[str]]]:
    """H1's clock-only fallback for when no column parses as a datetime at all and no elapsed
    numeric column exists either: a column whose non-empty values are PREDOMINANTLY (fraction
    >= ``_MIN_VALID_DT_FRACTION``, over non-empty cells) bare clock/time-of-day strings (see
    ``_clock_seconds_of_day``) and nothing else -- the file's only time base is a wall clock with
    no date column anywhere, a plain instrument/pump log rather than a Fracpro ASCII export.
    Converted to elapsed seconds from the first valid sample (by the shared rebase-and-synthesize
    code in ``load_csv``, same as ``_find_numeric_time_column``), with midnight rollovers
    unwrapped first (``_unwrap_midnight_rollovers``) so a job that runs past midnight doesn't
    read as time running backward.

    Measured case: Great Western's "Rate Pres.csv" (``"Time Stamp"``, AM/PM), Tap Rock's "Enron
    9 State Com 1 Pump Data.csv" (``"Time"``, 24-hour), Fifth Creek's "Pump_data.csv"
    (``"JobTime"``, 24-hour, under a ``"(datetime)"`` units row), WPX's "Olson 12-1 HX  Zones
    1-7.csv" (``"Time"``, AM/PM, under a ``"(hh:mm:ss)"`` units row) -- all single-day pump jobs
    logged by wall clock alone, previously either an outright load failure (once the bare-clock
    guard in ``parse_datetime`` correctly stopped treating them as same-day-by-default) or, on
    a job crossing midnight, a wrong reading from that same defaulting.

    Candidates are `dt_col` itself, then any other datetime-name-matching column (mirroring
    ``_find_numeric_time_column``). Deliberately never matches Crescent Point's Dressler
    "...1secdata.csv": its clock values are the ambiguous "MM:SS.f" shape that
    ``_clock_seconds_of_day`` excludes on principle, so that file keeps failing to load here
    exactly as it does everywhere else in this module, rather than landing on a wrong reading
    through this fallback instead.
    """
    candidates = [dt_col] + [
        c for c in _datetime_column_candidates(list(df.columns)) if c != dt_col
    ]
    for col in candidates:
        raw = df[col].astype("string").str.strip()
        n_non_empty = int(_non_empty_mask(raw).sum())
        if n_non_empty == 0:
            continue
        secs_of_day = _clock_seconds_of_day(raw)
        frac = float(secs_of_day.notna().sum()) / n_non_empty
        if frac < _MIN_VALID_DT_FRACTION:
            continue
        raw_secs = secs_of_day.to_numpy(dtype=float)
        if np.isfinite(raw_secs).sum() < 2:
            continue

        unwrapped = _unwrap_midnight_rollovers(raw_secs)
        warn = (
            "Time column has clock times only (no date); elapsed time computed from the first "
            "sample, midnight rollovers unwrapped."
        )
        return col, unwrapped, [warn]

    return None


# --------------------------------------------------------------------------------------------------
# preamble skipping (FIX C)
# --------------------------------------------------------------------------------------------------
def _is_numeric_field(field: str) -> bool:
    """True if a stripped, non-empty CSV field parses as a bare number."""
    try:
        float(field.strip())
        return True
    except ValueError:
        return False


def _detect_header_skiprows(path: str) -> int:
    """Find the header row of a CSV that carries a leading preamble, by field count.

    Measured case: 16 corpus files carry a "Job ID: ..., Spotter: ..." line and a "Row(s): N"
    line before the real header, which makes ``pd.read_csv`` raise ``ParserError`` (the
    preamble lines have a different field count than the data rows). A naive "retry with
    skiprows=1, 2, 3... and take the first one that parses" loop is wrong: on one such file,
    skiprows=1 parses *successfully* into a single-column frame named "Row(s): 24297" -- silent
    garbage, not an error. Instead, read at most the first 20 lines with the stdlib csv reader
    (so a quoted field containing a comma counts as one field, not two -- ``itertools.islice``
    stops the reader after 20 lines rather than filtering an unbounded read, so a huge file is
    never read in full just to keep 20 lines of it), find the modal (most-common, non-empty-line)
    field count -- that is the real table width.

    DEFECT 2: a candidate line is rejected outright if any non-empty stripped field parses as a
    bare number (``float()`` succeeds) -- a header essentially never has a purely numeric field,
    a data row (or a numeric metadata row, e.g. an all-"0" INSITE row) almost always does. This
    catches a *narrower* real header than the data rows it precedes (a ragged export with a
    trailing comma on data rows only): the modal width first appears on the first data row, not
    the header, and the old width-only check would silently adopt that data row as the header.
    Returns the first modal-width line with at least one non-empty field and no numeric field, or
    0 (meaning "no preamble found, let the caller's ``ParserError`` propagate unchanged") when no
    such line exists in the sampled window.

    DEFECT 6: any reader error -- a field over the stdlib csv module's size limit, a bad byte for
    this encoding, or an OS-level read failure -- is caught and turned into that same safe 0
    rather than escaping in place of the caller's informative ``ParserError``.
    """
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            lines = list(itertools.islice(reader, 20))
    except (csv.Error, UnicodeDecodeError, OSError):
        return 0

    counts = [len(row) for row in lines if len(row) > 0]
    if not counts:
        return 0
    modal = Counter(counts).most_common(1)[0][0]

    for i, row in enumerate(lines):
        if len(row) != modal:
            continue
        nonempty = [field for field in row if field.strip()]
        if not nonempty:
            continue
        if any(_is_numeric_field(field) for field in nonempty):
            continue
        return i
    return 0


# Datetime-name needles/avoid-words for _datetime_column_candidates below -- the same needles
# suggest_channels' own find("datetime", "date/time", "timestamp", "time", "date") uses to pick
# ITS single datetime_col, and the same rate/vol/press exclusion it applies afterward to that one
# column (see suggest_channels). Kept as shared constants so the two stay in lockstep by
# construction rather than by two independently-edited literal tuples.
_DATETIME_NAME_NEEDLES = ("datetime", "date/time", "timestamp", "time", "date")
_DATETIME_NAME_AVOID = ("rate", "vol", "press")


def _datetime_column_candidates(columns: list[str]) -> list[str]:
    """Every column that looks datetime-ish by name, not just the first (unlike
    ``suggest_channels``' own ``find()``, which stops at the first match and only then checks the
    avoid-words -- so a false-positive first match containing an avoid word blanks its result
    entirely instead of falling through to a later, real datetime column). Used by ``load_csv``
    to decide whether a multi-candidate evaluation (DEFECT: ``Time`` elapsed-minutes vs. ``Job
    Time`` wall-clock, see below) is even needed.

    Excludes any column that is another candidate's FIX A companion time-of-day column (see
    ``_companion_time_col``) -- a bare ``Time`` column reserved to be JOINED with a date-like-
    not-time-like sibling (e.g. plain ``Date``) must not also compete as an independent
    candidate: parsed alone, with no date at all, dateutil defaults it to today's date, which can
    make it look more "valid" than the fragmentary date column it actually belongs with (measured
    case: ``test_load_csv_keeps_joined_when_it_parses_more_than_date_only``'s mostly-blank
    ``Date`` + bare-time-of-day ``Time`` -- without this exclusion, comparing them head to head
    picks the companion and discards the real date entirely).
    """
    raw = [c for c in columns
           if any(n in c.lower() for n in _DATETIME_NAME_NEEDLES)
           and not any(a in c.lower() for a in _DATETIME_NAME_AVOID)]
    reserved = {comp for d in raw if (comp := _companion_time_col(d, columns)) in raw}
    return [c for c in raw if c not in reserved]


# _best_datetime_column scores each candidate on a bounded sample rather than the full column --
# parsing a multi-million-row column once per candidate (Horsetail 07E-0636, Fox Creek 504 gauge)
# measured a 5-10x slowdown on files with >1 datetime-name-matching column. 5,000 evenly-spaced
# non-null values is far more than enough to tell "mostly valid" from "mostly garbage" apart.
_DATETIME_SCORE_SAMPLE_CAP = 5000

# A bare time-of-day/clock value, e.g. "9:22:20 AM", "13:05:01", "15:10:9" (single-digit seconds
# -- the seconds group is ``\d{1,2}``, not ``\d{2}``, so a non-zero-padded second is still
# recognized), or an Excel-mangled elapsed "MM:SS.f" reading like "09:07.0" (no hours, no date at
# all) -- anchored on both ends so a full datetime string ("12/21/2017 9:22:20 AM") never
# matches. The trailing ``(?P<frac>\.\d+)?`` sits OUTSIDE the optional ``:(?P<s>\d{1,2})`` seconds
# group (not nested inside it) so it also matches a bare "MM:SS.f" shape with no seconds-colon at
# all -- measured case: Crescent Point's Dressler "...1secdata.csv" has an unnamed,
# header-detection-missed first column whose real values are ``09:07.0``, ``09:08.0``, ...
# (elapsed minutes:seconds, no date, from an ``INSITE`` treatment log under an undetected
# preamble). Read alone, dateutil parses "09:07.0" as 09:07:00 defaulted to TODAY's date -- a
# wrong-in-a-different-way-every-run "valid" timestamp -- while a value with minutes >= 24 (an
# invalid hour) correctly fails as NaT, so the column ends up a mix of NaT and spuriously-dated
# rows that (before the guard below) computed a multi-YEAR span once one genuinely dated row
# elsewhere in the column (this file's "11-Aug-14" job-header line, misread as a data row for the
# same undetected-preamble reason) anchored the elapsed-time math against a pile of today-dated
# rows years apart. Named groups (h/m/s/frac/ampm) let this same pattern double as the clock-only
# elapsed-time fallback's extractor (see ``_clock_seconds_of_day`` below) -- one regex, two jobs:
# "does this look like a bare clock string at all" (rejection, via a plain ``str.match``) and,
# separately, "what time-of-day does it actually encode" (extraction).
_TIME_OF_DAY_RE = re.compile(
    r"^(?P<h>\d{1,2}):(?P<m>\d{1,2})(?::(?P<s>\d{1,2}))?(?P<frac>\.\d+)?\s*(?P<ampm>AM|PM)?$",
    re.IGNORECASE,
)


def _sample_series(raw: pd.Series, cap: int = _DATETIME_SCORE_SAMPLE_CAP) -> pd.Series:
    """An evenly-spaced sample of up to `cap` non-null values from `raw`, preserving order.

    Drops nulls on the RAW (still-numeric/object) column first, before any string cast or
    parsing -- a single vectorized pass, not the expensive part -- so only the bounded sample
    itself ever gets cast to string and parsed (see ``_score_datetime_candidate``).
    """
    non_null = raw.dropna()
    n = len(non_null)
    if n <= cap:
        return non_null
    idx = np.unique(np.linspace(0, n - 1, cap).round().astype(int))
    return non_null.iloc[idx]


def _looks_like_time_of_day_only(sample: pd.Series) -> bool:
    """True when every value in `sample` (already non-null, string-cast) is a bare time-of-day,
    with no date part at all.

    Measured case: a column of pure ``"9:22:20 AM"``-shaped values -- dateutil silently defaults
    the missing date to today when parsed alone, so it can score a deceptively high valid-parse
    fraction and beat a real datetime column that merely has a few blank rows. Excluded from
    winning entirely (scored 0.0) rather than trusted at face value.
    """
    if len(sample) == 0:
        return False
    return bool(sample.str.match(_TIME_OF_DAY_RE).all())


def _score_datetime_candidate(df: pd.DataFrame, col: str) -> float:
    """The fraction of a bounded sample of `col` that parses as a valid timestamp -- see
    ``_sample_series`` and ``_looks_like_time_of_day_only``. 0.0 for an empty or bare-time-of-day
    column."""
    sample = _sample_series(df[col])
    if len(sample) == 0:
        return 0.0
    sample = sample.astype("string")
    if _looks_like_time_of_day_only(sample):
        return 0.0
    return float(parse_datetime(sample).notna().mean())


def _best_datetime_column(candidates: list[str], df: pd.DataFrame, default: str) -> str:
    """Among >1 datetime-name-matching columns, pick the one that actually parses into the most
    valid timestamps, scored on a bounded sample (see ``_score_datetime_candidate``).

    Measured case: Badger 25N-3HZ's CSV has both ``Time`` (elapsed minutes, e.g. ``85.00000``)
    and ``Job Time`` (wall-clock ``12/21/2017 9:22:20 AM`` strings) -- both match the needles
    above, and column order alone (``suggest_channels``' tiebreak) picks ``Time``, the wrong one,
    misparsing it as ~480 h of Excel-serial dates instead of ~20 min of real elapsed time.
    Scoring valid-parse fraction picks ``Job Time`` correctly instead. `default` is always scored
    too, even when it isn't itself among `candidates` (e.g. no needle matched it at all, so
    ``load_csv`` fell back to the first column) -- otherwise an unscored default reads as 0.0 and
    loses to literally any candidate, however bad. A tie -- including every candidate scoring
    0.0 -- keeps `default`, ``suggest_channels``' own single-column choice, unchanged.
    """
    best = default
    best_score = _score_datetime_candidate(df, default)
    for c in candidates:
        if c == default:
            continue
        score = _score_datetime_candidate(df, c)
        if score > best_score:
            best = c
            best_score = score
    return best


def _companion_time_col(datetime_col: str, columns: list[str]) -> Optional[str]:
    """The FIX A companion time-of-day column for ``datetime_col``, replicating
    ``suggest_channels``' own rule: only when ``datetime_col`` is date-like-but-not-time-like
    (its name contains "date" and not "time"), and there is a separate column whose bare name is
    exactly "time". Re-derived here (rather than trusted from ``suggest_channels``' ``guess``)
    because ``load_csv`` can swap in a different winning ``datetime_col`` -- see
    ``_best_datetime_column`` -- for which the original guess's companion, if any, no longer
    applies.
    """
    name = datetime_col.lower()
    if "date" in name and "time" not in name:
        for c in columns:
            if c != datetime_col and _bare_name(c) == "time":
                return c
    return None


def load_csv(path: str) -> TestData:
    """Load a DFIT CSV and attach an elapsed-seconds time base."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except pd.errors.ParserError:
        skiprows = _detect_header_skiprows(path)
        if skiprows == 0:
            raise
        df = pd.read_csv(path, encoding="utf-8-sig", skiprows=skiprows)

    df.columns = [c.strip() for c in df.columns]

    guess = suggest_channels(list(df.columns))
    dt_col = guess["datetime"] or df.columns[0]
    time_col = guess["time"]

    # DEFECT: more than one column name looks datetime-ish (e.g. an elapsed-minutes "Time"
    # column alongside a real wall-clock "Job Time" column) -- suggest_channels' own find() just
    # takes the first by column order, which can be the wrong one. Only pay for the extra parses
    # when there's actually more than one candidate to weigh.
    dt_candidates = _datetime_column_candidates(list(df.columns))
    if len(dt_candidates) == 1 and dt_candidates[0] != dt_col:
        # L1: exactly one real candidate survives _datetime_column_candidates' own companion-
        # reservation logic (see there), but it isn't the column suggest_channels' order-
        # dependent find() picked -- e.g. a bare "Time" column sitting BEFORE its real "Date"
        # pair in column order, where find() (first substring match, in column order) picks
        # "Time" outright and never even reaches "Date". No scoring is needed here: with only one
        # real candidate, it wins by construction, so switch to it directly.
        dt_col = dt_candidates[0]
        time_col = _companion_time_col(dt_col, list(df.columns))
    elif len(dt_candidates) > 1:
        winner = _best_datetime_column(dt_candidates, df, dt_col)
        if winner != dt_col:
            dt_col = winner
            time_col = _companion_time_col(dt_col, list(df.columns))

    def _parse_dt() -> pd.Series:
        # FIX A: a separate companion Time-of-day column has to be joined to the date column
        # before parsing, or the date alone collapses many-samples-per-day down to one. Adding
        # a missing value on either side (nullable StringDtype) propagates to a missing combined
        # value rather than the literal "nan"/"<NA>" text, so it still becomes NaT below.
        if time_col is not None:
            date_s = df[dt_col].astype("string").str.strip()
            # DEFECT 1b: recover a "HH:MM:SS:mmm" time base before joining (see
            # _normalize_ms_colon); a shape this doesn't recognize (garbage, or anything else)
            # passes through unchanged and is handled by the DEFECT 1a fallback just below.
            time_s = _normalize_ms_colon(df[time_col].astype("string").str.strip())
            # reject_bare_clock=False: a blank Date field beside a real Time value joins to a
            # date-less string by construction (DEFECT 1a, see parse_datetime's docstring) --
            # that is the intended, tested behavior for this specific join, not the Dressler-
            # shaped bug the guard exists to catch.
            joined = parse_datetime(date_s + " " + time_s, reject_bare_clock=False)
            date_only = parse_datetime(date_s)
            # DEFECT 1a: never let joining regress a file that was openable on the date column
            # alone. A companion Time column that parse_datetime can't make sense of (garbage, or
            # a shape the normalizer above doesn't cover) can join to a string that parses to
            # nothing -- measured on the Lucero Tahu files, joined yields 0 valid vs. 1,476 valid
            # on Date alone, which would turn a loadable (if date-only-resolution) file into a
            # hard load failure. Keep whichever parse recovered more timestamps; a tie keeps the
            # joined one since it is the higher-resolution result when it works. Measured on
            # Strathcona's 100-01-28-061-03W6-rt Aug15.csv: joined yields 333,234 valid vs.
            # 231,423 date-only, so joined (sub-second resolution) wins there.
            if date_only.notna().sum() > joined.notna().sum():
                return date_only
            return joined
        return parse_datetime(df[dt_col])

    dt = _parse_dt()

    # FIX D: a wholly reverse-chronological export (newest row first) would otherwise produce a
    # descending, negative-going elapsed-time base. Only reverse when the *entire* column is
    # reverse-ordered -- a handful of out-of-order rows, or an all-equal column, is left alone.
    valid = dt.dropna()
    if len(valid) > 1 and valid.is_monotonic_decreasing and not valid.is_monotonic_increasing:
        df = df.iloc[::-1].reset_index(drop=True)
        dt = _parse_dt()

    # FIX B: the datetime column parsed too little of the file to trust (see
    # _MIN_VALID_DT_FRACTION) -- fall back to a clean elapsed-time or clock-only column if the
    # file has one. The fraction is computed over `dt_col`'s own NON-EMPTY cells (see
    # _non_empty_mask -- a blank cell OR an Excel literal error-formula sentinel like "#REF!"),
    # not every row in the file: some exports carry blank or Excel-error-corrupted trailing
    # padding (no real data at all past the last logged sample), and dividing by the full row
    # count would drag the fraction down even though every REAL row parsed fine.
    n_non_empty = int(_non_empty_mask(df[dt_col]).sum())
    valid_frac = (dt.notna().sum() / n_non_empty) if n_non_empty else 0.0
    if valid_frac < _MIN_VALID_DT_FRACTION:
        elapsed = _find_elapsed_column(df)
        elapsed_warnings: list[str] = []
        if elapsed is None:
            # Stricter delta/elapsed-named path found nothing -- try the broader numeric-time-
            # column fallback (a bare Fracpro ASCII "Time" export, see _find_numeric_time_column)
            # before giving up.
            found = _find_numeric_time_column(df, dt_col)
            if found is not None:
                found_col, found_secs, elapsed_warnings = found
                elapsed = (found_col, found_secs)
        if elapsed is None:
            # H1: neither elapsed-numeric path matched either -- try a clock-only (wall-clock,
            # no date at all) column before giving up (see _find_clock_column).
            found = _find_clock_column(df, dt_col)
            if found is not None:
                found_col, found_secs, elapsed_warnings = found
                elapsed = (found_col, found_secs)
        if elapsed is not None:
            _, secs = elapsed
            # DEFECT 4: TestData.t_s is documented as "elapsed seconds from first sample", which
            # the datetime path guarantees via elapsed_seconds() (it subtracts the first valid
            # timestamp). This fallback must match: the source column need not itself start at
            # 0 (measured: True Oil\Abra Data's spotter files start at 10.00008 / 1.00008), so
            # rebase against the first FINITE value. NaN is left as NaN rather than rebased away
            # or invented a value -- missing elapsed data mirrors a NaT in the datetime path.
            finite = secs[np.isfinite(secs)]
            secs = secs - finite[0]
            synth_col = "DateTime"
            n = 2
            while synth_col in df.columns:
                synth_col = f"DateTime ({n})"
                n += 1
            df[synth_col] = _epoch_plus_seconds(_UNIX_EPOCH_NP, secs)
            return TestData(
                path=path, df=df, datetime_col=synth_col, t_s=secs, columns=list(df.columns),
                load_warnings=elapsed_warnings,
            )
        # H2: no elapsed or clock fallback either -- a datetime column below the trust threshold
        # with nothing to fall back to is untrustworthy, full stop, regardless of how many
        # "valid" timestamps happen to survive (a low, isolated minority is not a working time
        # base, so there is no ">=2 valid" exception here any more). Measured case: Liberty's
        # Anderson "DFIT-FINAL.csv" -- an undetected preamble collapses its real Date + Real
        # Time columns to anonymous "Unnamed: N" ones, landing dt_col on a plain sample-INDEX
        # column instead ("1", "2", "3", ...); roughly 1 in 11 of those integers happens to fall
        # inside the plausible Excel-serial range (_EXCEL_SERIAL_MIN/MAX) and gets misread as a
        # real (but wrong) date, reaching a nonzero valid COUNT -- 40,178 of 431,349 rows -- that
        # a "some rows parsed" check alone would have accepted; the FRACTION (~9%) is what
        # correctly flags it as untrustworthy instead, matching Crescent Point's Dressler
        # "...1secdata.csv" (a single genuine date surviving among thousands of rejected
        # bare-clock rows, see _TIME_OF_DAY_RE) under the exact same rule.
        raise ValueError(
            f"Could not parse a usable datetime column from {dt_col!r} "
            f"({dt.notna().sum()} valid of {n_non_empty} non-empty rows, no elapsed-time or "
            f"clock-only fallback found)"
        )

    df[dt_col] = dt
    t_s = elapsed_seconds(dt)

    return TestData(path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=list(df.columns))


# --------------------------------------------------------------------------------------------------
# Fracpro .DBS binary format
# --------------------------------------------------------------------------------------------------
# Reverse-engineered layout (little-endian throughout):
#   0x000            4-byte magic: 77 EF CD AB
#   0x004  uint32    file save timestamp (Unix epoch seconds; varies per file -- not validated)
#   0x2B4  uint32    n_channels
#   0x2B8  uint32    n_samples
#   0x2C0  float32   sample interval, in MINUTES
#   0x2C4  float32   total duration in minutes (informational only, not used here; has been seen
#                    not to equal n_samples * interval_min in a real merged file)
#   0x2CC  uint32    data_offset -- start of the sample records
#   0x334  ...       channel table: n_channels records of 84 bytes each
#     +0   char[4]   tag (e.g. "THCS", "SLRT")
#     +16  cstr      display name (latin-1, NUL-terminated); rest of the record is unused
#   data_offset ...  n_samples records of (4 + 4*n_channels) bytes each:
#                    uint32 sample index, then one float32 per channel (table order)
# There is no absolute start timestamp for the *data* in the file -- only elapsed time
# (index * interval). The 0x004 timestamp is when the file was saved, not when logging started.
_DBS_MAGIC = bytes.fromhex("77efcdab")
_DBS_HEADER_SIZE = 0x334
_DBS_CHANNEL_RECORD_SIZE = 84


def load_dbs(path: str) -> TestData:
    """Load a Fracpro ``.DBS`` binary file and attach an elapsed-seconds time base.

    See the module-level comment above for the binary layout. Returns the same ``TestData``
    shape as ``load_csv``: a synthetic ``"DateTime"`` column (the file carries no wall-clock
    time, only elapsed samples) plus one float64 column per channel.

    Defensive truncation of trailing padding records: some Fracpro exports pre-allocate a
    fixed-size record buffer sized for a planned recording duration and save the file before
    that buffer is fully written. The header's ``n_samples`` still reports the full
    pre-allocated count, but the never-written trailing records are zero-filled padding
    (``idx == 0`` repeated for every one of them, though not necessarily every channel float in
    those records -- garbage-looking non-zero values have been seen there too, so detection
    relies only on ``idx``, never on the channel floats). This is not a malicious or vanishingly
    rare case, just something this reverse-engineered parser has to tolerate: `expected_size ==
    len(data)` can't catch it, since the file's on-disk size genuinely matches
    ``data_offset + n_samples * record_size`` -- the padding is *within* the declared record
    count. A real recorder's ``idx`` is a monotonic sample counter that can legitimately skip
    values (a dropped/missed sample), but a genuine mid-file counter can also *jitter* -- repeat
    or double-count a step and then self-correct with a compensating skip a sample or two later
    (e.g. ``..., 28, 29, 30, 30, 32, 33, ...``) -- without the file being corrupted at all; the
    counter still reaches a proper terminal value by the last record. So detection can't just
    stop at the first non-forward step (that truncates real data out of otherwise-good files).
    Instead this scans *backward* from the last record: padding is a run of non-forward-progress
    that never recovers before EOF, while jitter always resumes forward progress before EOF. The
    scan walks back while ``idx[i] <= idx[i-1]`` and stops at the first (from-the-end) genuine
    forward step; everything after that position is the corrupted trailing suffix and is dropped
    (from ``rec``, and therefore from ``t_s`` and every per-channel column) before the caller
    ever sees it. Without this, ``t_s`` for those padding rows would collapse to time zero
    (derived from ``idx``, not array position) and pollute every plot and computation that reads
    ``td.t_s``/columns unfiltered. On a well-formed file (including one with ordinary mid-file
    jitter) this is a complete no-op. When truncation happens, a one-line note is appended to the
    returned ``TestData``'s ``load_warnings`` (folded into the analyst-visible warnings panel by
    ``model.compute_all``); a clean file leaves ``load_warnings`` empty.

    Two known, accepted edge cases (neither seen in the real corpus this was checked against,
    every observed padding run is a constant ``idx == 0``): a file whose very last real record
    happens to be an ordinary jitter repeat loses that one sample and reports it as padding --
    ambiguous by construction, since a single trailing repeat can't be told apart from a
    genuine one-record padding tail, but harmless (one sample, not a cascading truncation).
    And padding held at some constant value ABOVE the last genuine ``idx`` (rather than 0 or
    anything below it) would have its first padding record misread as one more genuine forward
    step and kept as a single spurious sample -- the scan only catches a run that fails to make
    forward progress, not one that jumps to an implausible value.
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != _DBS_MAGIC:
        raise ValueError(f"{path!r}: not a Fracpro DBS file (bad magic)")

    n_channels, n_samples = struct.unpack_from("<II", data, 0x2B4)
    interval_min = struct.unpack_from("<f", data, 0x2C0)[0]
    data_offset = struct.unpack_from("<I", data, 0x2CC)[0]

    if not (1 <= n_channels <= 64):
        raise ValueError(f"{path!r}: n_channels {n_channels} out of sane range (1..64)")
    if n_samples <= 0:
        raise ValueError(f"{path!r}: n_samples {n_samples} is not positive")
    if not (np.isfinite(interval_min) and interval_min > 0):
        raise ValueError(f"{path!r}: sample interval {interval_min} is not a positive finite number")

    expected_offset = _DBS_HEADER_SIZE + _DBS_CHANNEL_RECORD_SIZE * n_channels
    if data_offset != expected_offset:
        raise ValueError(
            f"{path!r}: data_offset {data_offset} != expected {expected_offset} "
            f"for {n_channels} channel(s)"
        )

    record_size = 4 + 4 * n_channels
    expected_size = data_offset + n_samples * record_size
    if expected_size != len(data):
        raise ValueError(
            f"{path!r}: file size {len(data)} != expected {expected_size} "
            f"for {n_samples} samples of {record_size} bytes"
        )

    # Channel names: display name at +16 in each 84-byte record, NUL-terminated latin-1, falling
    # back to the 4-char tag if blank. Duplicate names are deduped by suffixing the tag, then by
    # appending " (2)", " (3)", ... until unique -- every channel must land in its own column, or
    # the dict-based assembly below would silently overwrite one channel's data with another's.
    # "DateTime" is seeded into `seen` so a channel literally named that can't clobber the
    # synthetic datetime column.
    names: list[str] = []
    seen: set[str] = {"DateTime"}
    for i in range(n_channels):
        rec_off = _DBS_HEADER_SIZE + _DBS_CHANNEL_RECORD_SIZE * i
        tag = data[rec_off:rec_off + 4].decode("latin-1", errors="replace").strip("\x00").strip()
        raw_name = data[rec_off + 16:rec_off + _DBS_CHANNEL_RECORD_SIZE]
        nul = raw_name.find(b"\x00")
        if nul >= 0:
            raw_name = raw_name[:nul]
        name = raw_name.decode("latin-1").strip() or tag
        if name in seen:
            name = f"{name} [{tag}]"
        n = 2
        base = name
        while name in seen:
            name = f"{base} ({n})"
            n += 1
        seen.add(name)
        names.append(name)

    # Bulk-read the whole data region in one call: a structured dtype of (index, c0, c1, ...).
    dtype = np.dtype([("idx", "<u4")] + [(f"c{i}", "<f4") for i in range(n_channels)])
    rec = np.frombuffer(data, dtype=dtype, count=n_samples, offset=data_offset)

    # Detect and drop a trailing corrupted/padding suffix by scanning backward from the last
    # record (see the docstring above): a repeat/drop that never recovers before EOF is padding,
    # while one that resumes forward progress before EOF is just ordinary counter jitter and
    # must be left alone. This is a no-op whenever the last record is part of a run that made
    # genuine forward progress into it.
    load_warnings: list[str] = []
    idx_i64 = rec["idx"].astype(np.int64)
    if idx_i64.size > 1:
        i = idx_i64.size - 1
        while i > 0 and idx_i64[i] <= idx_i64[i - 1]:
            i -= 1
        valid_n = i + 1
        if valid_n < idx_i64.size:
            dropped = idx_i64.size - valid_n
            load_warnings.append(
                f"File declares {n_samples:,} samples but only the first {valid_n:,} were "
                f"recorded; ignored {dropped:,} empty trailing records."
            )
            rec = rec[:valid_n]

    t_s = rec["idx"].astype(np.float64) * float(interval_min) * 60.0

    dt_col = "DateTime"
    cols = {dt_col: _epoch_plus_seconds(_UNIX_EPOCH_NP, t_s)}
    for i, name in enumerate(names):
        cols[name] = rec[f"c{i}"].astype(np.float64)
    df = pd.DataFrame(cols)

    return TestData(path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=list(df.columns),
                     load_warnings=load_warnings)


def load(path: str) -> TestData:
    """Dispatch to ``load_dbs`` or ``load_csv`` based on the file extension."""
    if path.lower().endswith(".dbs"):
        return load_dbs(path)
    return load_csv(path)
