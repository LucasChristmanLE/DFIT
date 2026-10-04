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
import datetime as _dt
import itertools
import re
import struct
import xml.etree.ElementTree as _ET
import zipfile
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
# A third exact fast path: the primary layout with fractional seconds (e.g.
# "01/02/2020 03:04:05.123456", a millisecond-stamped gauge export). Tried after the AM/PM path
# so an ordinary whole-second column never pays for it; %f accepts 1-6 fractional digits, so a
# millisecond ("...05.123") or microsecond ("...05.123456") stamp both match.
_PRIMARY_DT_FORMAT_US = "%m/%d/%Y %H:%M:%S.%f"
_PRIMARY_DT_FORMAT_US_DAYFIRST = "%d/%m/%Y %H:%M:%S.%f"
# A fourth exact fast path for Caprito's "03-26-2018_16:44:05"-shaped stamps (dashes, an
# underscore separator, no AM/PM) -- shared by both the CSV and XLSX loaders, since both funnel
# through this one function.
_CAPRITO_DT_FORMAT = "%m-%d-%Y_%H:%M:%S"
_CAPRITO_DT_FORMAT_DAYFIRST = "%d-%m-%Y_%H:%M:%S"

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

# When the EXACT, vectorized fast paths in parse_datetime (24-hour, 12-hour AM/PM, fractional-
# second, and Caprito's underscore-separated layout) have already resolved at least this
# fraction of a column's non-empty values, Fallback 2 (the
# generic dateutil parse) is skipped entirely rather than run over the leftovers. Measured case:
# WPX Energy's "Emma Owner DFIT RawData.csv" -- 1.73M AM/PM-formatted rows resolve completely
# via fast path 2 except for one "(date time)" units-declaration row, but pandas' own format-
# INFERENCE (used when to_datetime is given no explicit `format`) never learns the AM/PM layout
# at all, not even from 1.73M clean examples of it -- so Fallback 2 falls straight to its slow,
# per-element dateutil.parser.parse() for every one of those already-resolved rows all over
# again (~55s), even with the masking above in place. At this point the leftover, unresolved
# rows are -- by construction, once the fast paths already own the overwhelming majority of the
# column -- units/header lines or corrupt cells, not a second, rescuably-different valid format;
# leaving them NaT (they already are, from the fast paths' own failure) rather than paying for a
# whole-column dateutil pass to confirm that is the right trade here.
_FAST_PATH_SKIP_FRACTION = 0.90

# Excel's own literal error-formula sentinels. A cell holding one of these is treated the same
# as a blank cell when deciding how much of a column is "real" data (see _non_empty_mask below)
# -- but ONLY when it sits inside a CONTIGUOUS run at the very start or end of the column (see
# _edge_run_mask). A scattered error cell mixed into otherwise-real data still counts as a
# non-empty, invalid cell -- exempting it unconditionally would let a column that's mostly (even
# ~99.9%) "#N/A" pass the trust-fraction check outright, which is exactly backwards. Measured
# case for the edge-run rule itself: Crestone Peak's "21011234 raw data.csv" has 233,470
# genuinely valid, contiguous "Date Time" rows followed by ~713,000 trailing "#REF!" rows (a
# broken Excel formula reference, not a reading of anything, but genuinely padding-shaped: one
# unbroken run at the end of the file) -- a real, good DFIT record whose real Pressure/Temp data
# actually CONTINUES through that trailing block (see the edge-block extrapolation below), not
# the ~9%-valid, no-real-data-at-all shape _MIN_VALID_DT_FRACTION's unconditional raise (see
# load_csv) exists to catch.
_EMPTY_CELL_TOKENS = frozenset({
    "#ref!", "#n/a", "#value!", "#div/0!", "#name?", "#null!", "#num!",
})


def _edge_run_mask(is_pad: np.ndarray) -> np.ndarray:
    """True for every position in a CONTIGUOUS run of `is_pad` at the very start of the array,
    or a contiguous run at the very end -- never for one in the middle. Used to tell genuine
    leading/trailing padding (or corruption) apart from a scattered cell of the same shape (see
    ``_non_empty_mask``/``_EMPTY_CELL_TOKENS``).
    """
    n = len(is_pad)
    mask = np.zeros(n, dtype=bool)
    i = 0
    while i < n and is_pad[i]:
        mask[i] = True
        i += 1
    j = n - 1
    while j >= i and is_pad[j]:
        mask[j] = True
        j -= 1
    return mask


def _blank_or_error_mask(s: pd.Series) -> np.ndarray:
    """True where `s` (a raw column) holds a blank cell or an Excel literal error-formula
    sentinel (see _EMPTY_CELL_TOKENS) -- i.e. a cell with no real value at all, regardless of
    where it sits. This is the shared "empty" test behind both ``_non_empty_mask``'s
    edge-run-restricted exemption (below) and the edge-block extrapolation's own readable-cell
    rule (``_extrapolate_or_warn_edge_block``): a cell that fails THIS test is never treated as
    "nothing was ever recorded here" -- if it also fails to parse, that is a real, if wrong- or
    corrupt-looking, reading, not a gap to paper over.
    """
    stripped = s.astype("string").str.strip()
    is_blank = (stripped.isna() | (stripped.str.len() == 0)).to_numpy()
    is_error = stripped.str.lower().isin(_EMPTY_CELL_TOKENS).fillna(False).to_numpy()
    return is_blank | is_error


def _numeric_data_bearing_mask(df: pd.DataFrame, cols: list, start: int, end: int) -> np.ndarray:
    """True for each row of ``df.iloc[start:end]`` that holds at least one finite NUMERIC value
    in one of ``cols`` -- used by ``_extrapolate_or_warn_edge_block`` to decide which rows of an
    edge NaT block are genuinely "real data, just no usable timestamp" versus padding. A cell
    merely being non-null is not enough: a units-declaration row (e.g. ``"(min) "``, ``" psi "``)
    or any other non-numeric text is not real data even though ``pd.notna`` would call it
    present -- ``pd.to_numeric(..., errors="coerce")`` maps it to NaN like a genuinely blank
    cell would be. Measured cases: Ballard's "Dilts 31-24 TH DFIT CSV Data.csv" has a units row
    right after its header (``"(min) ,(date time), (psi), (bpm), (psi), (bbls)"``) that a plain
    ``notna()`` check would count as one data-bearing row with no usable timestamp; Black Hills'
    "Cope 107 -108 16HS BHP Fracpro.CSV" has a blank-cells-plus-bare-"psi"-string units row
    right before its real data that a plain ``notna()`` check would extrapolate a timestamp for,
    one full median-step before the first real sample.
    """
    if not cols:
        return np.zeros(end - start, dtype=bool)
    sub = df.iloc[start:end][cols]
    data_bearing = np.zeros(len(sub), dtype=bool)
    for c in cols:
        vals = pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float)
        data_bearing |= np.isfinite(vals)
    return data_bearing


def _non_empty_mask(s: pd.Series) -> pd.Series:
    """True where `s` (already read as a raw column) holds something other than a blank cell, or
    an Excel literal error-formula sentinel (see _EMPTY_CELL_TOKENS) that sits inside a
    contiguous leading/trailing run of blank-or-error cells (see _edge_run_mask) -- i.e.
    something that OUGHT to be a real value, whether or not it actually parses as a date. A
    blank cell is excluded wherever it sits (leading, trailing, or scattered in the middle) --
    only the error-token exemption is position-restricted.
    """
    stripped = s.astype("string").str.strip()
    is_blank = (stripped.isna() | (stripped.str.len() == 0)).to_numpy()
    is_error = stripped.str.lower().isin(_EMPTY_CELL_TOKENS).fillna(False).to_numpy()
    in_edge_run = _edge_run_mask(is_blank | is_error)
    empty = is_blank | (is_error & in_edge_run)
    return pd.Series(~empty, index=s.index)

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


def _fill_na_exact(dt: pd.Series, s: pd.Series, fmt: str) -> pd.Series:
    """Fill the NaT rows of ``dt`` from an exact-format parse of ``s`` with ``fmt``, parsing only
    those rows. An explicit-format ``pd.to_datetime`` is elementwise, so this equals
    ``dt.combine_first(parse(s, fmt))`` on the whole column (``_fill_na_exact_reference``)."""
    na_mask = dt.isna().to_numpy()
    if not na_mask.any():
        return dt
    sub = pd.to_datetime(s[na_mask], format=fmt, errors="coerce").astype("datetime64[us]")
    vals = dt.to_numpy(copy=True)
    vals[na_mask] = sub.to_numpy()
    return pd.Series(vals, index=dt.index, name=dt.name)


def _fill_na_exact_reference(dt: pd.Series, s: pd.Series, fmt: str) -> pd.Series:
    """Original whole-column form of ``_fill_na_exact`` (fuzz-test reference)."""
    if not dt.isna().any():
        return dt
    full = pd.to_datetime(s, format=fmt, errors="coerce").astype("datetime64[us]")
    return dt.combine_first(full)


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

    # Fast paths 2-4 are exact-format strptime parses, i.e. purely elementwise, so each only
    # needs to run over the rows every earlier path left NaT (see _fill_na_exact); a column
    # path 1 mostly resolved never pays for three more whole-column parses.
    # Fast path 2: the 12-hour AM/PM layout (see _PRIMARY_DT_FORMAT_AMPM above), so a whole-file
    # AM/PM column never has to fall through to the much slower per-element dateutil inference
    # in Fallback 2 below.
    dt = _fill_na_exact(
        dt, s, _PRIMARY_DT_FORMAT_AMPM_DAYFIRST if dayfirst else _PRIMARY_DT_FORMAT_AMPM)

    # Fast path 3: the primary layout with fractional seconds (see _PRIMARY_DT_FORMAT_US above,
    # e.g. a millisecond-stamped gauge export).
    dt = _fill_na_exact(
        dt, s, _PRIMARY_DT_FORMAT_US_DAYFIRST if dayfirst else _PRIMARY_DT_FORMAT_US)

    # Fast path 4: Caprito's "03-26-2018_16:44:05"-shaped stamps (see _CAPRITO_DT_FORMAT above).
    dt = _fill_na_exact(dt, s, _CAPRITO_DT_FORMAT_DAYFIRST if dayfirst else _CAPRITO_DT_FORMAT)

    # Reject anything either exact fast path matched outside the plausible 1990-2100 window too
    # -- see _PLAUSIBLE_DT_MIN/MAX above, which used to guard only the two dateutil-based
    # fallbacks. An exact-format strptime match has no range check of its own (a 4-digit year
    # field accepts ANY 4 digits), so a corrupted-but-well-shaped date -- e.g. a typo'd year --
    # sails through a fast path untouched by any guard otherwise. Measured case: Civitas
    # Allred's "6 - 21011210.DTF.csv" has several implausible years (1941, 4221, 7127) that
    # match the exact format perfectly and previously poisoned the reported span by billions of
    # seconds. Applied before the fast-path-resolved snapshot below, so an implausible date
    # doesn't count as "resolved" for that gate either -- it gets exactly the same second look
    # from Fallback 2 (and, per Fallback 2's own guard, gets rejected there identically).
    in_range = (dt >= _PLAUSIBLE_DT_MIN) & (dt <= _PLAUSIBLE_DT_MAX)
    dt = dt.where(in_range | dt.isna(), np.datetime64("NaT", "us"))

    # Cheap snapshot (a boolean array copy, no string work) of what the two EXACT fast paths
    # above resolved, before either fallback below -- see the fast-path-skip gate on Fallback 2
    # further down for why this has to be captured here and not later, once Fallback 1 may have
    # resolved MORE rows itself. The actual fraction (which needs _non_empty_mask's string work)
    # is computed lazily at that gate, only for the (common) case of a column with at least one
    # value neither fast path resolved -- a column both fast paths fully resolve never pays for
    # it at all.
    fast_path_resolved = dt.notna().to_numpy()

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

    # Fallback 2: flexible parse for anything still missing (other string layouts) -- but ONLY
    # when the two exact fast paths above haven't already resolved the overwhelming majority of
    # the column (see _FAST_PATH_SKIP_FRACTION). Skipped entirely otherwise: pandas' own format-
    # INFERENCE (used here, with no explicit `format`) simply never learns some layouts at all,
    # no matter how many clean examples of it the column has -- measured case: WPX Energy's
    # "Emma Owner DFIT RawData.csv", 1.73M rows, ALL but one exact 12-hour-AM/PM-formatted (the
    # exception a "(date time)" units-declaration row). Fast path 2 resolves every real row
    # already; Fallback 2 (masking notwithstanding -- see below) still burns ~55s running the
    # slow, per-element dateutil.parser.parse() over the other 1,729,999 already-resolved rows,
    # because pandas' auto-format-guess never recognizes the AM/PM layout to begin with and so
    # always falls to that slow path for this shape, REGARDLESS of how clean the data is. Once
    # the fast paths already own >=90% of the column, the small remainder is -- by construction
    # -- units/header lines or corrupt cells, not a second, rescuably-different valid format, so
    # this is a fair trade: leave them NaT (they already are, from the fast paths' own failure)
    # rather than pay for a whole-column dateutil pass just to confirm that.
    still_na = dt.isna()
    if not still_na.any():
        fast_path_frac = 1.0
    elif not fast_path_resolved.any():
        # Nothing at all resolved by the fast paths -- the fraction is 0 regardless of the
        # denominator, so skip _non_empty_mask(s) (string work over the whole column) entirely
        # rather than compute it just to divide 0 by it. Measured case: a bare "date"-only
        # column (no time component) with several other datetime-name-matching sibling columns
        # -- every candidate this column loses to still gets scored via this same function, and
        # none of its rows ever match either exact fast-path format.
        fast_path_frac = 0.0
    else:
        non_empty = _non_empty_mask(s)
        n_non_empty = int(non_empty.sum())
        fast_path_frac = (
            float((fast_path_resolved & non_empty.to_numpy()).sum()) / n_non_empty
            if n_non_empty else 0.0
        )
    if still_na.any() and fast_path_frac < _FAST_PATH_SKIP_FRACTION:
        # What must happen BEFORE the parse call below, not after, is blanking (to NA, not
        # removing -- length must stay full so every remaining value keeps its real neighbors)
        # every value that could never legitimately become a NEW date through this fallback
        # anyway:
        #
        # - A bare number: dateutil's generic inference happily reads "2024" as 2024-01-01,
        #   "1500.5" as 1500-05-01, or "20171221" as 2017-12-21 -- exactly the class of bug
        #   Fallback 1's plausible-serial-range guard exists to prevent, just reached by a
        #   different path. A purely numeric string gets exactly one sanctioned route to a date
        #   -- Fallback 1's Excel-serial parse above -- and anything that fallback already
        #   rejected (out of the plausible 1990-2100 range) must stay NaT here too, not get a
        #   second, unguarded attempt.
        # - A bare time-of-day/clock string (see _TIME_OF_DAY_RE above, e.g. "9:22:20 AM" or the
        #   Dressler-shaped "09:07.0") -- it has no date component at all, so whatever date
        #   dateutil defaulted it to (today, always) is not a reading of the file, just an
        #   artifact of when this code happened to run. Skippable (see the docstring's
        #   ``reject_bare_clock`` note) for FIX A's join, which relies on this exact defaulting
        #   for a blank Date value beside a real Time one.
        # - A value with no digit in it at all -- it cannot be a date or time under ANY
        #   interpretation, and dateutil's format-GUESS itself (not just the eventual per-row
        #   parse) can fail outright on one, which is worse than merely wasting time on it:
        #   pandas then gives up guessing ANY format and falls back to the slow, per-element
        #   dateutil.parser.parse() for EVERY remaining non-null value in the array, not just
        #   this one. This masking is what keeps a file BELOW the 90% fast-path-skip threshold
        #   fast when it still has to reach this branch -- e.g. a column where the fast paths
        #   resolve only 70-80% of rows and a "(date time)" units line is one of the genuinely
        #   unresolved remainder: without masking it out first, that one non-digit value can
        #   still force the slow path over the WHOLE column, the same failure mode the skip gate
        #   above exists to avoid for a much-more-resolved column, just below its threshold.
        #
        # Reindexing to a smaller, already-more-homogeneous-garbage subset instead of blanking
        # in place would reopen a correctness gap on top of all this: pd.to_datetime's format
        # guess reads from the WHOLE array's context, so a smaller subset can change what gets
        # inferred for a row that would parse fine with its real neighbors visible. Measured
        # case: Strathcona's 100-09-14-062-04W6-rt.csv has a corrupted row (a Date cell's
        # leading day digits are simply gone, joined with a real Time into
        # "/10/2022 22:11:25") sitting among thousands of well-formed
        # "31/10/2022 22:11:25"-shaped neighbors -- parsed WITH those neighbors visible, pandas
        # guesses their shared format and the corrupted row simply fails to match it (NaT,
        # correct); parsed alone, dateutil's own single-value guess reads it as a
        # plausible-but-wrong 2022-10-01. (In practice this file's fast-path-resolved fraction is
        # itself so high that the skip gate above never even lets this call run for it any
        # more -- the corrupted row is already NaT from the fast paths' own failure, which is the
        # same correct outcome by construction. This reasoning is kept for the rarer file that
        # falls just under the skip threshold.)
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
    not "time" -- so "Date/Time", "DateTime", and "Timestamp (MST)" never trigger it); see
    ``_companion_time_col`` (called directly, so the two stay in lockstep) for which column wins
    when more than one is "time"-shaped -- an exact bare "Time" always wins first, in column
    order, else a multi-token "...Time"-named column (e.g. an XLSX header folded to "Real Time
    (HH:MM:SS)"), also in column order.

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

    time_col = _companion_time_col(datetime_col, columns) if datetime_col else None

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
        return [f"{label} unit ambiguous (psi or kPa); assumed psi"]
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
            warn = f"Column {col!r} has no time unit; assumed minutes"
            return col, vals * 60.0, [warn]

    return None


def _clock_extract(s: pd.Series) -> dict:
    """Regex-extract h/m/s/frac/AM-PM plus derived per-row booleans from a clock-shaped string
    series (see ``_TIME_OF_DAY_RE``'s named groups) -- shared by ``_clock_seconds_of_day`` (the
    ordinary wall-clock reading), the two elapsed-duration readings below, and
    ``_classify_clock_mode``'s decision between them.
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
    # Dressler's own ambiguous shape (exactly two colon fields plus a fraction, no seconds
    # group) -- never a usable value through ANY of the three readings below.
    ambiguous_shape = (~has_sec) & has_frac
    with np.errstate(invalid="ignore"):
        min_ok = (mi >= 0) & (mi <= 59)
        sec_ok = (~has_sec) | ((sec >= 0) & (sec <= 59))
    return dict(
        index=s.index, matched=matched, h=h, mi=mi, has_sec=has_sec, sec=sec,
        has_frac=has_frac, frac=frac, has_ampm=has_ampm, is_am=is_am, is_pm=is_pm,
        ambiguous_shape=ambiguous_shape, min_ok=min_ok, sec_ok=sec_ok,
    )


def _clock_seconds_of_day(s: pd.Series) -> pd.Series:
    """Seconds-since-midnight for a bare clock/time-of-day string (H:MM, H:MM:S, or H:MM:SS,
    optional AM/PM), or NaN when the string doesn't match ``_TIME_OF_DAY_RE`` at all, its
    hour/minute/second is out of range, or its shape is the Dressler-ambiguous "MM:SS.f" one --
    exactly two colon-separated fields plus a fraction, e.g. "27:32.7" -- indistinguishable from
    an "H:MM" reading with a fractional minute, so never treated as a usable clock value here
    (see ``_find_clock_column``, which relies on this exclusion to keep Dressler's own file
    correctly failing rather than landing on a wrong reading through this fallback instead).
    Hour range is 0-23 with no AM/PM marker, 1-12 with one (``12 AM`` -> midnight, ``12 PM`` ->
    noon, matching the ordinary 12-hour convention). This is the "clock" mode reading -- see
    ``_classify_clock_mode`` for when a value this shape is actually an elapsed DURATION
    instead, which is read by ``_elapsed_hms_seconds``/``_elapsed_mmss_seconds`` below.
    """
    g = _clock_extract(s)
    with np.errstate(invalid="ignore"):
        hour_ok = np.where(g["has_ampm"], (g["h"] >= 1) & (g["h"] <= 12),
                            (g["h"] >= 0) & (g["h"] <= 23))
    ok = g["matched"] & ~g["ambiguous_shape"] & hour_ok & g["min_ok"] & g["sec_ok"]
    h24 = np.where(g["is_am"] & (g["h"] == 12), 0.0, g["h"])
    h24 = np.where(g["is_pm"] & (h24 != 12), h24 + 12.0, h24)
    sec_val = np.where(g["has_sec"], g["sec"], 0.0)
    frac_val = np.where(g["has_frac"], g["frac"], 0.0)
    total = h24 * 3600.0 + g["mi"] * 60.0 + sec_val + frac_val
    total = np.where(ok, total, np.nan)
    return pd.Series(total, index=g["index"])


def _elapsed_hms_seconds(s: pd.Series) -> pd.Series:
    """h*3600 + m*60 + s for a 3-field "H:MM:SS" value read as an elapsed DURATION -- the hour
    field is left UNBOUNDED (no <=23 check) rather than a wall-clock hour, since a duration can
    genuinely run past 24h (see ``_classify_clock_mode``'s "elapsed_hms" mode, e.g. "39:59:00"
    forty minutes before hour 40). No AM/PM marker is accepted here -- a duration with one would
    be self-contradictory.
    """
    g = _clock_extract(s)
    with np.errstate(invalid="ignore"):
        ok = (
            g["matched"] & g["has_sec"] & ~g["has_ampm"] & ~g["ambiguous_shape"]
            & g["min_ok"] & g["sec_ok"] & (g["h"] >= 0)
        )
    total = g["h"] * 3600.0 + g["mi"] * 60.0 + g["sec"]
    total = np.where(ok, total, np.nan)
    return pd.Series(total, index=g["index"])


def _elapsed_mmss_seconds(s: pd.Series) -> pd.Series:
    """m*60 + s for a 2-field "MM:SS" value read as an elapsed DURATION -- the first field is
    left UNBOUNDED (an elapsed-minutes count, not a wall-clock hour), since it can genuinely run
    past 59 (see ``_classify_clock_mode``'s "elapsed_mmss" mode). No AM/PM marker and no
    fraction (Dressler's own ambiguous shape) are accepted here.
    """
    g = _clock_extract(s)
    with np.errstate(invalid="ignore"):
        ok = (
            g["matched"] & (~g["has_sec"]) & (~g["has_frac"]) & (~g["has_ampm"])
            & g["min_ok"] & (g["h"] >= 0)
        )
    total = g["h"] * 60.0 + g["mi"]
    total = np.where(ok, total, np.nan)
    return pd.Series(total, index=g["index"])


def _median_run_length(vals: np.ndarray) -> float:
    """The median length, in samples, of each maximal run of consecutive EQUAL values in
    `vals` (order preserved, no sorting) -- used by ``_classify_clock_mode`` to tell whether a
    2-field value's own resolution is coarser than the sample rate (it repeats for multiple
    consecutive rows before changing, consistent with an hour:minute CLOCK sampled faster than
    once a minute) or matches it (a new value nearly every row, consistent with a minutes:
    seconds elapsed count at roughly 1 Hz). Returns 0.0 for fewer than 2 values.
    """
    if vals.size < 2:
        return 0.0
    changed = np.diff(vals) != 0
    run_starts = np.concatenate(([0], np.flatnonzero(changed) + 1, [vals.size]))
    run_lengths = np.diff(run_starts)
    return float(np.median(run_lengths))


def _classify_clock_mode(raw: pd.Series) -> str:
    """Decide how a clock-shaped column's matched values should actually be read -- the SAME
    "H:MM[:SS]" shape is genuinely ambiguous between a wall clock and an elapsed duration, and
    guessing wrong silently produces a plausible-looking but wrong span (see the measured cases
    below). Returns one of:

    - ``"clock"``: an ordinary wall-clock reading. Read by ``_clock_seconds_of_day``, with
      midnight-rollover unwrapping.
    - ``"elapsed_hms"``: a 3-field "H:MM:SS" reading where at least one value's hour exceeds 23
      -- proof it's an elapsed DURATION (a duration can run arbitrarily long), not a wall-clock
      hour that wrapped. Measured (synthetic): "39:59:00", forty minutes before hour 40, on a
      column that would otherwise misread as a wall clock stuck at "23:59:00"-ish, truncating a
      genuine 40h record to the ~24h a naive wall-clock reading (with no unwrap trigger, since
      nothing ever drops) would show.
    - ``"elapsed_mmss"``: a 2-field, no-AM/PM reading where the first field exceeds 23 at some
      point -- too high to be even a 24-hour hour, so it must be elapsed MINUTES. Measured
      (synthetic): a 45-minute "00:00".."45:00" MM:SS record that a naive "H:MM" wall-clock
      reading turned into a bogus ~23-HOUR span. A record under 24 minutes never proves this.
    - ``"ambiguous"``: a 2-field, no-AM/PM reading that fits none of the proofs below. Never
      guessed -- see ``_find_clock_column``, which raises instead.

    A value carrying AM/PM is unambiguous on its own (only a real clock hour takes one), so its
    presence anywhere in the column settles "clock" outright before anything else runs. For a
    2-field, no-AM/PM reading (H:MM vs. MM:SS is otherwise genuinely ambiguous), the remaining
    checks run in this fixed order (1 and 2 are proofs; 3 is a heuristic, returned as
    ``"clock_inferred"`` so the caller adds a low-confidence warning):

    1. The first field ever exceeding 23 (checked BEFORE anything else below) is conclusive on
       its own -- too high to be even a 24-hour hour, so it must be elapsed MINUTES
       (``"elapsed_mmss"``) regardless of what the other checks would have said.
    2. A midnight wrap -- a backward step from a first field >=23 down to a first field of 0 --
       is conclusive the other way: elapsed minutes never reset to 0 after climbing past 23,
       they just keep counting (24, 25, ...), so a genuine drop back to 0 right after reaching
       the top of the hour range is proof of an actual hour rolling over -- ``"clock"``.
    3. Otherwise, the column's own resolution vs. its sample rate: if the same (first, second)
       pair typically repeats across multiple consecutive rows (``_median_run_length`` >= 2),
       the field's own resolution is coarser than the sample rate -- consistent with a real
       hour:minute clock sampled faster than once a minute -- ``"clock"``. If instead nearly
       every row has a fresh (first, second) pair (median run length < 2), that matches a
       minutes:seconds elapsed count advancing roughly every sample -- but is NOT on its own
       proof of MM:SS (unlike the first two checks), so this falls through to ``"ambiguous"``
       rather than assuming ``"elapsed_mmss"`` without the first field ever having proven it.
    """
    g = _clock_extract(raw)
    valid_shape = g["matched"] & ~g["ambiguous_shape"] & g["min_ok"] & g["sec_ok"]
    if bool((valid_shape & g["has_ampm"]).any()):
        return "clock"

    has_sec_vals = valid_shape & g["has_sec"]
    no_sec_vals = valid_shape & ~g["has_sec"]
    if bool(has_sec_vals.any()) and not bool(no_sec_vals.any()):
        # Dominant/only shape is 3-field (H:MM:SS).
        with np.errstate(invalid="ignore"):
            if bool((g["h"][has_sec_vals] > 23).any()):
                return "elapsed_hms"
        return "clock"
    if bool(no_sec_vals.any()) and not bool(has_sec_vals.any()):
        # Dominant/only shape is 2-field, no AM/PM -- genuinely ambiguous on its own. `first`/
        # `minute` are kept in original row order (no reindexing to a 0-based positional array
        # would be needed either way, but boolean masking already preserves order) so the wrap
        # and run-length checks below read real consecutive-row relationships.
        first = g["h"][no_sec_vals]
        minute = g["mi"][no_sec_vals]
        with np.errstate(invalid="ignore"):
            if bool((first > 23).any()):
                return "elapsed_mmss"
            if first.size > 1 and bool(((first[:-1] >= 23) & (first[1:] == 0)).any()):
                return "clock"
        pair_id = first * 100.0 + minute  # collapses (first, minute) into one comparable value
        if _median_run_length(pair_id) >= 2:
            return "clock_inferred"
        return "ambiguous"
    # Mixed 2-field and 3-field shapes in the same column (not seen in the corpus) -- fall back
    # to the ordinary wall-clock reading and let its own per-value range checks sort out what
    # they can, rather than guessing at a mode for a shape this function has no real case for.
    return "clock"


def _unwrap_midnight_rollovers(raw_secs: np.ndarray) -> np.ndarray:
    """Add 24h for every backward step of more than 1h in `raw_secs` (seconds-of-day, NaN
    allowed, as returned by ``_clock_seconds_of_day``).

    A clock-only column has no date to carry the day boundary, so a value that drops by more
    than an hour from the previous FINITE sample is read as the clock having wrapped past
    midnight, not as time running backward -- 1h (rather than requiring something closer to a
    full 12h) so a genuine but sparsely-sampled overnight gap (e.g. a ~13h break between an
    evening and the next morning's readings) still unwraps correctly; a small backward step
    (a few seconds, from ordinary out-of-order noise) stays well under this and is left alone.
    NaN entries are skipped when looking for the previous sample (never treated as a step
    themselves) and stay NaN in the result.
    """
    raw_secs = np.asarray(raw_secs, dtype=float)
    out = np.full(raw_secs.shape, np.nan)
    valid = np.isfinite(raw_secs)
    vals = raw_secs[valid]
    if vals.size:
        diffs = np.diff(vals)
        rollover = diffs < -3600.0
        offset = np.concatenate([[0.0], np.cumsum(np.where(rollover, 24 * 3600.0, 0.0))])
        out[valid] = vals + offset
    return out


def _find_clock_column(
    df: pd.DataFrame, dt_col: str
) -> Optional[tuple[str, np.ndarray, list[str], bool]]:
    """H1's clock-only fallback for when no column parses as a datetime at all and no elapsed
    numeric column exists either: a column whose non-empty values are PREDOMINANTLY (fraction
    >= ``_MIN_VALID_DT_FRACTION``, over non-empty cells) bare clock/time-of-day strings and
    nothing else -- the file's only time base is a wall clock (or, per ``_classify_clock_mode``,
    an elapsed duration written in the same H:MM[:SS] shape) with no date column anywhere, a
    plain instrument/pump log rather than a Fracpro ASCII export. Converted to elapsed seconds
    from the first valid sample (by the shared rebase-and-synthesize code in ``load_csv``, same
    as ``_find_numeric_time_column``).

    A genuine wall-clock reading gets two more corrections before midnight-rollover unwrapping
    (``_unwrap_midnight_rollovers``): if MOST of its consecutive steps run backward (a
    newest-first log, mirroring FIX D above but for this column specifically -- FIX D itself
    only ever looks at the PRIMARY datetime column, which this fallback runs only after that one
    has already failed), the row order is reversed first, matching FIX D's own treatment of a
    genuinely reverse-chronological export.

    Returns ``(column, elapsed_seconds, warnings, was_reversed)`` -- `was_reversed` tells
    ``load_csv`` to reverse `df` itself (every other column along with the time base) the same
    way FIX D does, or ``None`` if no candidate qualifies. Raises ``ValueError`` directly,
    rather than returning ``None``, when NO candidate resolves cleanly and at least one hit a
    genuinely unresolvable clock-vs-elapsed ambiguity (see ``_classify_clock_mode``'s
    ``"ambiguous"``) -- a clear, specific error (naming the FIRST such candidate) beats silently
    falling through to load_csv's generic "no fallback found" message when the tool DID find a
    column that's clearly meant to be the time base and just can't tell what it means. An
    ambiguous candidate is not fatal on its own, though -- a LATER candidate that resolves
    cleanly still wins normally.

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
    ``_clock_extract`` excludes on principle (the same exclusion feeds all three readings above),
    so that file keeps failing to load here exactly as it does everywhere else in this module,
    rather than landing on a wrong reading through this fallback instead.
    """
    candidates = [dt_col] + [
        c for c in _datetime_column_candidates(list(df.columns)) if c != dt_col
    ]
    ambiguous_col: Optional[str] = None
    for col in candidates:
        raw = df[col].astype("string").str.strip()
        n_non_empty = int(_non_empty_mask(raw).sum())
        if n_non_empty == 0:
            continue
        mode = _classify_clock_mode(raw)
        inferred = mode == "clock_inferred"
        if inferred:
            mode = "clock"
        if mode == "ambiguous":
            # Try the remaining candidates before giving up on this one -- a later column might
            # still resolve cleanly even though this one can't. Remembered (first one only) so
            # a clear, specific error can still be raised below if NOTHING else works out either.
            if ambiguous_col is None:
                ambiguous_col = col
            continue
        if mode == "elapsed_hms":
            secs_of_day = _elapsed_hms_seconds(raw)
        elif mode == "elapsed_mmss":
            secs_of_day = _elapsed_mmss_seconds(raw)
        else:
            secs_of_day = _clock_seconds_of_day(raw)
        frac = float(secs_of_day.notna().sum()) / n_non_empty
        if frac < _MIN_VALID_DT_FRACTION:
            continue
        raw_secs = secs_of_day.to_numpy(dtype=float)
        if np.isfinite(raw_secs).sum() < 2:
            continue

        reversed_ = False
        if mode == "clock":
            valid_vals = raw_secs[np.isfinite(raw_secs)]
            diffs = np.diff(valid_vals)
            if int((diffs < 0).sum()) > int((diffs > 0).sum()):
                raw = raw.iloc[::-1].reset_index(drop=True)
                secs_of_day = _clock_seconds_of_day(raw)
                raw_secs = secs_of_day.to_numpy(dtype=float)
                reversed_ = True
            unwrapped = _unwrap_midnight_rollovers(raw_secs)
            warn = (
                "Time column has clock times only (no date)"
                + ("; rows reversed (newest first)" if reversed_ else "")
                + ("; H:MM vs MM:SS guessed (low confidence)" if inferred else "")
            )
        else:
            # An elapsed duration is already monotonic by construction (its "hour" or "minutes"
            # field is exactly the thing that keeps counting past the ordinary wrap point), so
            # no unwrap or reversal check applies here.
            unwrapped = raw_secs
            shape = "H:MM:SS" if mode == "elapsed_hms" else "MM:SS"
            warn = f"Time column read as {shape} elapsed duration"
        return col, unwrapped, [warn], reversed_

    if ambiguous_col is not None:
        raise ValueError(
            f"Column {ambiguous_col!r}: H:MM vs MM:SS ambiguous (no AM/PM marker, and neither "
            f"field's range, a midnight wrap, nor the column's own resolution proves it either "
            f"way)."
        )
    return None


def _find_edge_nat_block(dt: pd.Series) -> Optional[tuple[str, int, int]]:
    """The bounds of a contiguous leading or trailing run of NaT in `dt`, if one exists at
    either edge -- or None if there is no such run (including a column that's entirely NaT,
    which is the trust-fraction check's job in load_csv, not this one's). Returns
    ``(side, start, end)`` (end-exclusive), `side` one of ``"leading"``/``"trailing"``. When
    both edges carry a run, the longer one wins (a tie favors trailing, the more common shape --
    a mid-export corruption or cutoff).
    """
    n = len(dt)
    valid = dt.notna().to_numpy()
    if valid.all() or not valid.any():
        return None
    i = 0
    while i < n and not valid[i]:
        i += 1
    leading_len = i
    j = n - 1
    while j >= 0 and not valid[j]:
        j -= 1
    trailing_start = j + 1
    trailing_len = n - trailing_start
    if trailing_len >= leading_len and trailing_len > 0:
        return ("trailing", trailing_start, n)
    if leading_len > 0:
        return ("leading", 0, leading_len)
    return None


_EDGE_CONTINUITY_WINDOW = 100
_EDGE_COLUMN_FINITE_FRAC = 0.90
_EDGE_LOCAL_SLOPE_STEPS = 200
_EDGE_LOCAL_SLOPE_TOL = 0.01


def _regular_step(diffs: np.ndarray, med: float) -> Optional[float]:
    """The per-row step to extrapolate a timestamp-less edge block at, or None when the good run
    is not regular enough: >=99% of consecutive steps within 1% of the median -> the median step.
    A jittery run (sub-second stamps wandering around the cadence) is NOT extrapolated: it gets
    the "N data-bearing rows had no usable timestamp" warning instead, the conservative outcome.
    """
    if np.mean(np.abs(diffs - med) <= 0.01 * med) >= 0.99:
        return float(med)
    return None


def _local_step_agrees(diffs: np.ndarray, med: float, side: str) -> bool:
    """True when the median step over the last (trailing block) or first (leading block)
    ``_EDGE_LOCAL_SLOPE_STEPS`` steps of the good run matches the whole-run median within 1% --
    a rate change confined to the end nearest the block would otherwise be extrapolated at the
    wrong, whole-run rate."""
    near = diffs[-_EDGE_LOCAL_SLOPE_STEPS:] if side == "trailing" else diffs[:_EDGE_LOCAL_SLOPE_STEPS]
    if near.size == 0:
        return True
    return bool(abs(float(np.median(near)) - med) <= _EDGE_LOCAL_SLOPE_TOL * med)


def _finite_numeric(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _columns_continue(df: pd.DataFrame, cols: list, side: str, sub_start: int, sub_end: int,
                      good_start: int, good_end: int) -> Optional[str]:
    """Column-continuity gate for extrapolating an edge block. Returns None when the block's
    columns continue the adjacent good run, else a short reason string.

    (a) The set of numeric-data-bearing columns (>=90% finite numeric) in the block must equal
    the set in the adjacent window of the good run (``good_start:good_end``, up to
    ``_EDGE_CONTINUITY_WINDOW`` rows). A column dying (or appearing) at the timestamp's death is
    a different dataset, not a continuation.
    (b) No column may hold one constant value across the whole block while it varied in the
    window: constant padding (a scheduled-volume column zero-filled past the end of a log) is
    not data."""
    bearing_good, bearing_blk = set(), set()
    const_in_block, varied_in_good = set(), set()
    for c in cols:
        g = _finite_numeric(df[c].iloc[good_start:good_end])
        k = _finite_numeric(df[c].iloc[sub_start:sub_end])
        if g.size and np.isfinite(g).mean() >= _EDGE_COLUMN_FINITE_FRAC:
            bearing_good.add(c)
        if k.size and np.isfinite(k).mean() >= _EDGE_COLUMN_FINITE_FRAC:
            bearing_blk.add(c)
        kf, gf = k[np.isfinite(k)], g[np.isfinite(g)]
        if kf.size and kf.min() == kf.max():
            const_in_block.add(c)
        if gf.size and gf.min() != gf.max():
            varied_in_good.add(c)
    if bearing_good != bearing_blk:
        diff = sorted(map(str, bearing_good ^ bearing_blk))
        return f"data columns differ across the boundary ({', '.join(diff[:4])})"
    pad = sorted(map(str, const_in_block & varied_in_good))
    if pad:
        return f"column(s) constant across the block ({', '.join(pad[:4])})"
    return None


def _extrapolate_or_warn_edge_block(
    dt: pd.Series, df: pd.DataFrame, dt_col: str, time_col: Optional[str]
) -> tuple[pd.Series, list[str]]:
    """After ordinary parsing, a contiguous leading or trailing run of NaT (see
    ``_find_edge_nat_block``) that still carries real values in some OTHER column is real,
    data-bearing rows that just happen to have no usable timestamp -- typically the same
    corruption ``_non_empty_mask`` exempts from the trust-fraction check in the first place
    (a genuinely EMPTY edge run, with nothing else in it either, is left alone here -- there is
    nothing to extrapolate FOR). If the surviving good run's own sample interval is regular
    (>=99% of consecutive steps within 1% of the median, plus the local-slope and column-continuity gates), the missing timestamps are
    extrapolated at that median step; otherwise they are left as NaT (never guessed at an
    irregular spacing), and a warning says so either way, so this is never silent.

    Measured case: Crestone Peak's "21011234 raw data.csv" -- 233,470 valid, contiguous 1 Hz
    "Date Time" rows, then ~713,000 trailing "#REF!" rows whose Pressure/Temp columns keep
    right on reading real, continuing values. Extrapolating the trailing block at the prefix's
    exact 1.0 s step turns the file's reported span from ~64.85h (the trustworthy prefix alone)
    into the full ~262.88h record.

    Two further conditions restrict WHICH rows of that block actually get a manufactured
    timestamp, both of which can leave a strict subset of the block unresolved (a "N data-
    bearing rows had no usable timestamp" warning alongside the extrapolation one when that
    happens, not instead of it -- unless NOTHING in the block qualifies, when it's the only one):

    "Data-bearing" itself means at least one OTHER column holds a finite NUMERIC value
    (``_numeric_data_bearing_mask``, ``pd.to_numeric(..., errors="coerce")``) -- a units-
    declaration row (e.g. Ballard's "(min) ,(date time), (psi), (bpm), (psi), (bbls)") or a bare
    unit string (e.g. Black Hills' "                 ,               psi ") is not real data
    even though the cell is non-null; a plain ``notna()`` check would wrongly count either as a
    data-bearing row and either warn about a units row with "no usable timestamp" or extrapolate
    a fabricated timestamp one median-step before the real data.

    - Contiguity: only the rows that are THEMSELVES data-bearing in an unbroken run starting
      right at the anchor (the last valid timestamp, for a trailing block; the first, for a
      leading one) are reachable at all -- a run of fully-empty (or units-only, non-numeric)
      rows breaks it, and nothing past that break is reachable here even if it has real data
      further out. Measured case: Great Western's "Seltzer Pump - 036HN.csv" has 5,419 trailing
      NaT rows, but the row immediately after its last valid timestamp is ALREADY fully empty
      (nothing in any column) -- so the run's own last 60 rows, which DO carry real Pressure/
      Rate data, sit past that break and stay unresolved (reported as "60 data-bearing rows had
      no usable timestamp") rather than being extrapolated as if they were the very next sample.
    - Readable cells: within whatever's reachable, only a row whose own datetime cell (and
      companion Time cell, if this is a FIX-A join) is genuinely blank or an Excel error token
      (``_blank_or_error_mask``) gets a manufactured value -- a cell that has real, readable
      text and simply failed to PARSE (e.g. an implausible-year date the plausible-range guard
      correctly rejected) is a real reading, not a gap, and is never overwritten. Measured case:
      Extraction Oil & Gas's "Wake 33-20-13-21011204-PRESSURE.csv" opens with 63 rows dated
      "7/25/1987" (a stuck default clock before the gauge was synced) ahead of its real
      1990-2100-range data -- readable, not blank/error, so none of them get a fabricated
      "elapsed backward from the first good sample" timestamp; the file still reports its real
      ~1,199.7h span from the good data alone.
    """
    block = _find_edge_nat_block(dt)
    if block is None:
        return dt, []
    side, start, end = block
    exclude_cols = {dt_col} if time_col is None else {dt_col, time_col}
    other_cols = [c for c in df.columns if c not in exclude_cols]
    if not other_cols:
        return dt, []
    data_bearing = _numeric_data_bearing_mask(df, other_cols, start, end)
    n_data_bearing_total = int(data_bearing.sum())
    if n_data_bearing_total == 0:
        return dt, []

    # Shrink to the contiguous data-bearing run starting right at the anchor edge.
    if side == "trailing":
        k = 0
        while k < len(data_bearing) and data_bearing[k]:
            k += 1
        sub_start, sub_end = start, start + k
    else:
        k = 0
        while k < len(data_bearing) and data_bearing[len(data_bearing) - 1 - k]:
            k += 1
        sub_start, sub_end = end - k, end
    if sub_start == sub_end:
        return dt, [f"{n_data_bearing_total} data-bearing rows had no usable timestamp."]

    valid = dt.dropna()
    valid_pos = np.flatnonzero(dt.notna().to_numpy())
    if len(valid) < 2:
        return dt, [f"{n_data_bearing_total} data-bearing rows had no usable timestamp."]
    secs = (valid - valid.iloc[0]).dt.total_seconds().to_numpy()
    diffs = np.diff(secs)
    med = np.median(diffs) if len(diffs) else np.nan
    if not (np.isfinite(med) and med > 0):
        return dt, [f"{n_data_bearing_total} data-bearing rows had no usable timestamp."]
    step = _regular_step(diffs, med)
    if step is None or not _local_step_agrees(diffs, med, side):
        return dt, [f"{n_data_bearing_total} data-bearing rows had no usable timestamp."]

    # Column continuity: the block must look like the same dataset as the adjacent good run.
    if side == "trailing":
        good_end = start
        good_start = max(0, good_end - _EDGE_CONTINUITY_WINDOW)
    else:
        good_start = end
        good_end = min(len(df), good_start + _EDGE_CONTINUITY_WINDOW)
    why = _columns_continue(df, other_cols, side, sub_start, sub_end, good_start, good_end)
    if why is not None:
        return dt, [
            f"{n_data_bearing_total} rows without timestamps not extrapolated: they do not "
            f"continue the timestamped record ({why})"
        ]

    # Within the reachable run, only rows whose own cell(s) are genuinely blank/error-token are
    # extrapolatable -- a readable-but-unparsed cell is left alone (see the docstring's Wake
    # example).
    n_reachable = sub_end - sub_start
    extrapolatable = _blank_or_error_mask(df[dt_col].iloc[sub_start:sub_end])
    if time_col is not None:
        extrapolatable = extrapolatable & _blank_or_error_mask(
            df[time_col].iloc[sub_start:sub_end])
    n_extrapolatable = int(extrapolatable.sum())
    if n_extrapolatable == 0:
        return dt, [f"{n_data_bearing_total} data-bearing rows had no usable timestamp."]

    dt = dt.copy()
    positions = np.arange(sub_start, sub_end)[extrapolatable]
    if side == "trailing":
        anchor = dt.iloc[:start].dropna().iloc[-1]
        offsets = (np.arange(1, n_reachable + 1) * step)[extrapolatable]
        new_vals = (anchor + pd.to_timedelta(offsets, unit="s")).to_numpy().astype(
            "datetime64[us]")
        dt.iloc[positions] = new_vals
        warn = f"{n_extrapolatable} rows after {anchor}: timestamps extrapolated at {step:g} s"
    else:
        anchor = dt.iloc[end:].dropna().iloc[0]
        offsets = (np.arange(n_reachable, 0, -1) * step)[extrapolatable]
        new_vals = (anchor - pd.to_timedelta(offsets, unit="s")).to_numpy().astype(
            "datetime64[us]")
        dt.iloc[positions] = new_vals
        warn = f"{n_extrapolatable} rows before {anchor}: timestamps extrapolated at {step:g} s"
    warnings_out = [warn]
    n_unresolved = n_data_bearing_total - n_extrapolatable
    if n_unresolved > 0:
        warnings_out.append(f"{n_unresolved} data-bearing rows had no usable timestamp.")
    return dt, warnings_out


def _mask_isolated_timestamp_outliers(dt: pd.Series) -> tuple[pd.Series, int]:
    """After ordinary parsing, a single valid sample whose elapsed time differs from BOTH its
    immediate valid neighbors by more than ``max(1h, 100x the median step)`` -- while those two
    neighbors are themselves consistent with EACH OTHER (within that same threshold) -- is a
    corrupted timestamp, not a genuine sudden jump, and is set to NaT.

    Measured case: Strathcona's 100-09-14-062-04W6-rt.csv, row 5563 -- a Date cell reads
    "8/10/2022" where every neighbor (both before and after) reads "28/10/2022" (a dropped
    leading digit). Read alone this sample lands 20 days before its neighbors, and the very next
    sample jumps back 20 days to rejoin them -- a lone sample surrounded on both sides by a huge,
    but mutually-consistent, discontinuity is exactly the signature of one bad cell, not two
    genuine jumps. This is what tells it apart from a REAL discontinuity (e.g. a tail-trim-
    worthy gauge dropout or a genuine gap in logging): a real gap changes the LEVEL and stays
    there, so the sample after it is close to the sample that caused the jump, not back near
    where it started.

    Returns ``(dt, n_masked)`` -- `dt` unchanged (same object) when nothing is masked.
    """
    valid_mask = dt.notna().to_numpy()
    valid_idx = np.flatnonzero(valid_mask)
    if valid_idx.size < 3:
        return dt, 0
    valid_vals = dt.to_numpy()[valid_idx]
    secs = (valid_vals - valid_vals[0]) / np.timedelta64(1, "s")
    secs = secs.astype(float)
    step = np.abs(np.diff(secs))
    if step.size < 2:
        return dt, 0
    med_step = float(np.median(step))
    threshold = max(3600.0, 100.0 * med_step)
    jump_left = step[:-1]
    jump_right = step[1:]
    neighbor_gap = np.abs(secs[2:] - secs[:-2])
    is_outlier = (jump_left > threshold) & (jump_right > threshold) & (neighbor_gap <= threshold)
    n_outliers = int(is_outlier.sum())
    if n_outliers == 0:
        return dt, 0
    outlier_positions = valid_idx[1:-1][is_outlier]
    dt = dt.copy()
    dt.iloc[outlier_positions] = pd.NaT
    return dt, n_outliers


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


def _detect_header_skiprows(path: str, encoding: str = "utf-8-sig") -> int:
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
        with open(path, "r", encoding=encoding, newline="") as f:
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


def _datetime_column_candidates(
    columns: list[str], df: Optional[pd.DataFrame] = None
) -> list[str]:
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

    `df`, when given, adds one more check before a name-matched companion is allowed to reserve
    anything: its own sampled VALUES must actually be time-of-day-shaped (``_companion_is_time_of_day``),
    not a second, independent full datetime -- e.g. a "Job Time" column holding
    ``"04/10/2016 12:00:00"``-shaped values beside a bare "Date" column is a real standalone
    datetime candidate in its own right, and must not be reserved away (and so silently excluded
    from `dt_candidates`) just because its NAME also happens to end in "Time". Left `None` (the
    default), this check is skipped and only the name rule applies -- unchanged for the two other
    callers of this function that use it for an unrelated elapsed-time fallback search, not the
    FIX A join decision, and never had a `df` to check against in the first place.
    """
    raw = [c for c in columns
           if any(n in c.lower() for n in _DATETIME_NAME_NEEDLES)
           and not any(a in c.lower() for a in _DATETIME_NAME_AVOID)]

    def _reserved_by(d: str) -> Optional[str]:
        comp = _companion_time_col(d, columns)
        if comp is None or df is None:
            return comp
        return comp if _companion_is_time_of_day(comp, df) else None

    reserved = {comp for d in raw if (comp := _reserved_by(d)) in raw}
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


# _companion_is_time_of_day tolerates up to this fraction of a companion's sampled values NOT
# matching the bare time-of-day shape before vetoing it -- a real corpus companion column
# routinely has one leading units-declaration row (e.g. "(hh:mm:ss)", "HH:mm:ss") sampled right
# alongside thousands of genuine "14:03:18"-shaped rows (`_sample_series` always includes index 0
# via its linspace), which `_looks_like_time_of_day_only`'s strict ALL-match would otherwise
# reject outright over one non-data row. A column that is actually a second full datetime (the
# case this veto exists to catch) fails far more broadly than one stray row, so this margin does
# not let one through.
_COMPANION_TIME_OF_DAY_MIN_FRACTION = 0.9


def _companion_is_time_of_day(comp: str, df: pd.DataFrame) -> bool:
    """True if at least ``_COMPANION_TIME_OF_DAY_MIN_FRACTION`` of `comp`'s own sampled VALUES
    (not just its name) are shaped like a bare time-of-day, never a full datetime -- over the
    same bounded sample ``_score_datetime_candidate`` uses (see ``_TIME_OF_DAY_RE``). Used to
    veto a same-named "...Time" companion column (``_companion_time_col`` matches purely by
    name) whose actual values are a second, independent full-datetime column -- e.g. a "Job
    Time" wall-clock column sitting next to a bare "Date" column -- rather than a genuine
    time-of-day pair to join with it. `comp` not being a real column at all (shouldn't happen
    given how this is called, but never trusted) reads as "not time-of-day" -- there's nothing to
    join.

    Normalizes a ``HH:MM:SS:mmm`` colon-milliseconds shape (``_normalize_ms_colon``, DEFECT 1b)
    before matching -- the same rewrite the real FIX A join applies -- so a genuine time-of-day
    companion in that shape isn't vetoed just because its RAW form doesn't match
    ``_TIME_OF_DAY_RE`` yet.
    """
    if comp not in df.columns:
        return False
    sample = _normalize_ms_colon(_sample_series(df[comp]).astype("string"))
    if len(sample) == 0:
        return False
    sample = sample.str.strip()
    frac = float(sample.str.match(_TIME_OF_DAY_RE).fillna(False).mean())
    return frac >= _COMPANION_TIME_OF_DAY_MIN_FRACTION


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


# A multi-token "...Time"-named column is never a companion when one of its OTHER tokens names
# it as elapsed/relative time rather than a time-of-day reading -- "Elapsed Time", "Test Time",
# "Delta Time", "Duration Time" are all measured-from-somewhere durations, not a wall-clock
# companion to join with a date. Checked before the "prefixed"/"preferred" split below, so one of
# these never wins even when it sits earlier in column order than a genuine time-of-day column.
_COMPANION_TIME_EXCLUDE_TOKENS = frozenset({"elapsed", "test", "delta", "duration"})
# Among the surviving multi-token "...Time" candidates, one naming itself as a genuine wall-clock
# reading ("Real Time", "Clock Time") is preferred over any other -- column order is only the
# tiebreak among (or, absent one of these, between) the rest.
_COMPANION_TIME_PREFERRED_TOKENS = frozenset({"real", "clock"})


def _companion_time_col(datetime_col: str, columns: list[str]) -> Optional[str]:
    """The FIX A companion time-of-day column for ``datetime_col``: only when ``datetime_col``
    is date-like-but-not-time-like (its name contains "date" and not "time"), and there is a
    separate column whose bare name ends in the "time" token. Re-derived here (rather than
    trusted from ``suggest_channels``' ``guess``) because ``load_csv`` can swap in a different
    winning ``datetime_col`` -- see ``_best_datetime_column`` -- for which the original guess's
    companion, if any, no longer applies. ``suggest_channels`` calls this directly rather than
    duplicating the rule, so the two stay in lockstep.

    A bare-"time" column (e.g. plain ``"Time"``) always wins outright when one exists, in column
    order -- the original, narrower rule, kept exactly as before for every existing caller. Only
    when no such exact match exists does a multi-token "...Time"-named column get picked instead:
    one naming itself a genuine wall-clock/time-of-day reading (``_COMPANION_TIME_PREFERRED_TOKENS``
    -- e.g. an XLSX header folded to ``"Real Time (HH:MM:SS)"``, bare-named "Real Time") wins over
    any other such column, in column order among ties; failing that, the first remaining
    multi-token "...Time" column in column order, EXCLUDING one named as an elapsed/relative
    duration (``_COMPANION_TIME_EXCLUDE_TOKENS`` -- "Elapsed Time", "Test Time", "Delta Time",
    "Duration Time" are never eligible at all, regardless of position). This is what pairs
    Southern Ute's downhole-gauge "Real Date"/"Real Time" split correctly, ahead of that same
    sheet's unrelated "Test Time" (elapsed minutes) column sitting right after it -- both by the
    "real" preference and by "Test Time"'s own exclusion -- and what pairs a "Date, Elapsed Time,
    Real Time" file with "Real Time" rather than the "Elapsed Time" column column order alone
    would otherwise have picked first. Restricting the broader match to a fallback (rather than
    treating both matches as equally eligible from the start) means no existing bare-"Time"
    corpus file's companion choice can ever change.
    """
    name = datetime_col.lower()
    if "date" not in name or "time" in name:
        return None
    exact: Optional[str] = None
    preferred: Optional[str] = None
    prefixed: Optional[str] = None
    for c in columns:
        if c == datetime_col:
            continue
        tokens = _tokens(_bare_name(c))
        if not tokens or tokens[-1] != "time":
            continue
        if len(tokens) == 1:
            if exact is None:
                exact = c
            continue
        if any(t in _COMPANION_TIME_EXCLUDE_TOKENS for t in tokens):
            continue
        if preferred is None and any(t in _COMPANION_TIME_PREFERRED_TOKENS for t in tokens):
            preferred = c
        elif prefixed is None:
            prefixed = c
    if exact is not None:
        return exact
    if preferred is not None:
        return preferred
    return prefixed


def _read_csv_frame(path: str) -> pd.DataFrame:
    """Read a DFIT CSV into a raw DataFrame: ``pd.read_csv`` (with the preamble-skiprows retry
    on ``ParserError``, see ``_detect_header_skiprows``) plus the column-name strip. Everything
    after this -- datetime choice, elapsed/clock fallbacks, extrapolation, the outlier guard --
    lives in ``_finish_frame`` so ``load_xlsx`` can share it against a frame built its own way.

    Tries UTF-8 (BOM-tolerant) first and re-reads as Latin-1 on a ``UnicodeDecodeError`` (the
    ``Sn#...`` downhole-gauge exports carry a bare 0xB0 degree byte); that fallback sets
    ``df.attrs["encoding_warning"]`` for ``load_csv`` to surface.
    """
    def _read(encoding: str) -> pd.DataFrame:
        try:
            return pd.read_csv(path, encoding=encoding)
        except pd.errors.ParserError:
            skiprows = _detect_header_skiprows(path, encoding)
            if skiprows == 0:
                raise
            return pd.read_csv(path, encoding=encoding, skiprows=skiprows)

    try:
        df = _read("utf-8-sig")
    except UnicodeDecodeError:
        df = _read("latin-1")
        df.attrs["encoding_warning"] = "File is not UTF-8; read as Latin-1"

    df.columns = [c.strip() for c in df.columns]
    return df


def _finish_frame(df: pd.DataFrame, path: str) -> TestData:
    """Turn a raw, column-stripped frame into a ``TestData``: datetime-column choice (including
    the multi-candidate scoring and FIX-A companion join), the reverse-chronological flip, the
    isolated-outlier guard, the elapsed/clock-only fallbacks (FIX B), and the edge-NaT-block
    extrapolation. Shared, unchanged, by ``load_csv`` and ``load_xlsx`` -- everything upstream of
    this (turning a CSV or an XLSX sheet into a raw frame) is format-specific and lives in
    ``_read_csv_frame``/``load_xlsx`` instead.
    """
    guess = suggest_channels(list(df.columns))
    dt_col = guess["datetime"] or df.columns[0]

    # DEFECT: more than one column name looks datetime-ish (e.g. an elapsed-minutes "Time"
    # column alongside a real wall-clock "Job Time" column) -- suggest_channels' own find() just
    # takes the first by column order, which can be the wrong one. Only pay for the extra parses
    # when there's actually more than one candidate to weigh. Passing `df` lets the companion-
    # reservation check inside _datetime_column_candidates veto a same-named "...Time" column
    # whose own values are a second full datetime, not a genuine time-of-day pair (see
    # _companion_is_time_of_day) -- without it, that column would be wrongly reserved as some
    # OTHER candidate's companion and drop out of contention here entirely.
    dt_candidates = _datetime_column_candidates(list(df.columns), df=df)
    if len(dt_candidates) == 1 and dt_candidates[0] != dt_col:
        # L1: exactly one real candidate survives _datetime_column_candidates' own companion-
        # reservation logic (see there), but it isn't the column suggest_channels' order-
        # dependent find() picked -- e.g. a bare "Time" column sitting BEFORE its real "Date"
        # pair in column order, where find() (first substring match, in column order) picks
        # "Time" outright and never even reaches "Date". No scoring is needed here: with only one
        # real candidate, it wins by construction, so switch to it directly.
        dt_col = dt_candidates[0]
    elif len(dt_candidates) > 1:
        winner = _best_datetime_column(dt_candidates, df, dt_col)
        if winner != dt_col:
            dt_col = winner

    # FIX A's companion is always re-derived against the FINAL dt_col here (never trusted as-is
    # from the initial `guess` above, which is name-only and can't see this column's real
    # values) and vetoed unless its own sampled values are genuinely time-of-day-shaped (see
    # _companion_is_time_of_day) -- so a same-named "...Time" column that actually holds a
    # second, independent full datetime (e.g. a "Job Time" wall-clock column beside a bare
    # "Date") is never joined as if it were a time-of-day pair.
    time_col = _companion_time_col(dt_col, list(df.columns))
    if time_col is not None and not _companion_is_time_of_day(time_col, df):
        time_col = None

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

    # Isolated-timestamp-outlier guard: a single corrupted cell (e.g. a dropped date digit)
    # surrounded by otherwise-consistent neighbors is set to NaT rather than left to poison the
    # reported span with a huge, spurious swing (see _mask_isolated_timestamp_outliers).
    dt, n_outliers = _mask_isolated_timestamp_outliers(dt)
    outlier_warnings = (
        [f"{n_outliers} isolated timestamp outlier(s) masked"]
        if n_outliers else []
    )

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
            # no date at all) column before giving up (see _find_clock_column). May raise
            # directly (an unresolvable clock-vs-elapsed-duration ambiguity -- see
            # _classify_clock_mode) rather than returning None.
            found = _find_clock_column(df, dt_col)
            if found is not None:
                found_col, found_secs, elapsed_warnings, was_reversed = found
                if was_reversed:
                    # Mirrors FIX D above, but for this fallback column specifically -- FIX D
                    # itself only ever looks at the primary datetime column, which this
                    # fallback runs only after that one has already failed.
                    df = df.iloc[::-1].reset_index(drop=True)
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
            # outlier_warnings deliberately excluded here: it describes outlier masking done on
            # dt_col's OWN parse, which this branch has just abandoned in favor of a synthetic
            # elapsed/clock column -- attaching it would describe a column that isn't the one
            # actually used.
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

    # A contiguous leading/trailing NaT block that still carries real data in some OTHER column
    # (see _extrapolate_or_warn_edge_block) gets its timestamps extrapolated, or an explicit
    # warning if the surviving run's own spacing is too irregular to extrapolate from -- either
    # way, never silent. A no-op (empty warnings, dt unchanged) for the ordinary case of no such
    # block, or one that really is just padding with nothing else in it either.
    dt, edge_warnings = _extrapolate_or_warn_edge_block(dt, df, dt_col, time_col)
    df[dt_col] = dt
    t_s = elapsed_seconds(dt)

    return TestData(
        path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=list(df.columns),
        load_warnings=outlier_warnings + edge_warnings,
    )


def load_csv(path: str) -> TestData:
    """Load a DFIT CSV and attach an elapsed-seconds time base."""
    df = _read_csv_frame(path)
    enc_warning = df.attrs.get("encoding_warning")
    td = _finish_frame(df, path)
    if enc_warning:
        td.load_warnings.append(enc_warning)
    return td


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
    jitter) this is a complete no-op. The dropped records hold no data, so truncation is silent:
    ``load_warnings`` stays empty either way.

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
    idx_i64 = rec["idx"].astype(np.int64)
    if idx_i64.size > 1:
        i = idx_i64.size - 1
        while i > 0 and idx_i64[i] <= idx_i64[i - 1]:
            i -= 1
        valid_n = i + 1
        if valid_n < idx_i64.size:
            rec = rec[:valid_n]  # empty trailing records: dropped without a warning

    t_s = rec["idx"].astype(np.float64) * float(interval_min) * 60.0

    dt_col = "DateTime"
    cols = {dt_col: _epoch_plus_seconds(_UNIX_EPOCH_NP, t_s)}
    for i, name in enumerate(names):
        cols[name] = rec[f"c{i}"].astype(np.float64)
    df = pd.DataFrame(cols)

    return TestData(path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=list(df.columns))


# --------------------------------------------------------------------------------------------------
# XLSX time-series workbooks
# --------------------------------------------------------------------------------------------------
# A header row must sit within the first N rows of a sheet (a preamble -- "Company Name: ...",
# "Well Name: ...", blank rows -- otherwise pushes it further down); a units row, if present,
# always sits directly below the header, and real data directly below THAT (or directly below the
# header, if there's no units row) -- so the numeric-data lookahead only needs 2 rows: "the next
# row is data" or "the next row is a units row, and the one after that is data". Deliberately
# NOT a wider window: a real corpus preamble row can itself contain both a "Date"-named cell and
# a couple of real datetime values in adjacent cells of the SAME multi-field preamble line (e.g.
# Southern Ute's downhole-gauge file, row 1: "Well Name: SOUTHERN UTE 32-8 NO. <date> Date of
# Test: <date>") -- with several blank rows before the REAL header. A wide lookahead would reach
# straight past those blank rows into the real header/units/data block a few rows further down
# and false-positive on the preamble line instead.
_XLSX_HEADER_SCAN_ROWS = 60
_XLSX_DATA_LOOKAHEAD_ROWS = 2

# Units-row token vocabulary (see _xlsx_looks_like_unit_token): recognized bare unit words, and a
# date/time-FORMAT token (e.g. "MM/DD/YY", "HH:MM:SS") made only of the letters M/D/Y/H/S plus
# '/', ':', '.', and spaces. Anything parenthesized (e.g. "(psi)") is accepted outright, matching
# the CLAUDE.md rule ("for example parenthesized tokens") without needing its contents to be a
# recognized word at all.
_XLSX_UNIT_WORDS = frozenset({
    "psi", "psig", "psia", "kpa", "kpag", "mpa", "bar",
    "bpm", "gpm", "gal", "gals", "gallon", "gallons",
    "bbl", "bbls", "barrel", "barrels", "m3", "m3/min",
    "ft", "feet", "min", "mins", "minute", "minutes",
    "sec", "secs", "second", "seconds", "hr", "hrs", "hour", "hours",
    "deg f", "degf", "deg c", "degc", "f", "c", "deltap", "delta p",
})
_XLSX_UNIT_FORMAT_RE = re.compile(r"^[mdyhs/:. ]+$", re.IGNORECASE)


def _xlsx_is_number(cell: object) -> bool:
    """True for a real, finite numeric cell -- ``bool`` is excluded (openpyxl can hand back a
    literal ``True``/``False`` for a checkbox-shaped cell, which is never DFIT sample data)."""
    return isinstance(cell, (int, float)) and not isinstance(cell, bool) and np.isfinite(cell)


def _xlsx_is_datetimeish(cell: object) -> bool:
    return isinstance(cell, (_dt.datetime, _dt.date, _dt.time))


def _xlsx_looks_like_unit_token(text: str) -> bool:
    """True if the already-stripped, non-empty string `text` looks like one units-row cell --
    see the module comment above ``_XLSX_UNIT_WORDS`` for the three accepted shapes."""
    if text.startswith("(") and text.endswith(")"):
        return True
    low = text.lower().rstrip(".")
    if low in _XLSX_UNIT_WORDS:
        return True
    if _XLSX_UNIT_FORMAT_RE.match(text) and any(ch in "mdyhs" for ch in low):
        return True
    return False


def _xlsx_row_is_units_row(cells: tuple) -> bool:
    """True if `cells` (one raw openpyxl row) is a units-declaration row: at least one non-blank
    cell, every non-blank cell is a string that looks like a unit token (see
    ``_xlsx_looks_like_unit_token``), and none is a real number or a datetime/date/time value --
    a genuine data row always has at least one of those, a units row never does.
    """
    non_empty = [c for c in cells
                 if c is not None and not (isinstance(c, str) and c.strip() == "")]
    if not non_empty:
        return False
    for c in non_empty:
        if _xlsx_is_number(c) or _xlsx_is_datetimeish(c):
            return False
        if not isinstance(c, str) or not _xlsx_looks_like_unit_token(c.strip()):
            return False
    return True


_XLSX_DATA_WINDOW_MAX_SKIP = 3


def _xlsx_data_window(rows: list, start: int, count: int) -> list:
    """Up to `count` rows from `rows[start:]`, skipping past up to
    ``_XLSX_DATA_WINDOW_MAX_SKIP`` leading rows that are PURELY textual -- every non-blank cell a
    string, none a real number or datetime/date/time value -- so ``_xlsx_looks_like_data_rows``
    below judges real data, never diluted by a run of label/units rows that would otherwise drag
    a genuine header's actual data rows below the 50% threshold. A units-declaration row (see
    ``_xlsx_row_is_units_row``) is always one of these, but so is any other all-text row a real
    corpus file inserts between the header and its data -- e.g. GMT's "... dfit pump info.xlsx"
    files, which carry a units row with a bare "%" column label ``_xlsx_row_is_units_row``
    doesn't recognize as a unit token, and one file in that same family which inserts an EXTRA
    "Instantaneous value" descriptive row before its units row (two skippable rows, not one).
    Real DFIT data never legitimately starts with a run of cells that are ALL text (even a
    key/value preamble row's one real value stops the skip immediately, since it's a number or a
    date -- see ``_xlsx_looks_like_data_rows``'s own docstring for the preamble cases that
    correctly reject), so this is safe to apply generally rather than only to a recognized units
    vocabulary. Stops at the first row with at least one non-blank, non-string cell, or once the
    row-count bound is reached.
    """
    i = start
    skipped = 0
    while i < len(rows) and skipped < _XLSX_DATA_WINDOW_MAX_SKIP:
        row = rows[i]
        non_empty = [c for c in row
                     if c is not None and not (isinstance(c, str) and c.strip() == "")]
        if non_empty and all(isinstance(c, str) for c in non_empty):
            i += 1
            skipped += 1
            continue
        break
    return rows[i: i + count]


def _xlsx_looks_like_data_rows(rows: list) -> bool:
    """True if `rows` (the header candidate's data-lookahead window, see ``_xlsx_data_window``)
    looks like real tabular data, not a key/value preamble line that merely carries a number or a
    date next to it: pooled across every row, at least half of the non-empty cells are numeric or
    a datetime/date/time value, AND that numeric/datetime data spans at least 2 distinct columns.

    The column-count clause is what tells a genuine multi-channel data row apart from a
    preamble's "Label:  value" pairing even when the pooled fraction alone would pass -- e.g. two
    lookahead rows each shaped like ``("Test No:", 5)`` / ``("TVD:", 8500)`` pool to exactly 50%
    numeric (one real number in one string-and-number pair per row), but both numbers sit in the
    SAME column (the "value" column), never the >=2 distinct data columns a real header's rows
    carry. Measured corpus false positive this fixes: Vermilion's "... DFITTEXT.xlsx" has a
    "Date, Customer, Well Name/No., Job Type" header candidate whose very next row is
    ``(datetime, "Vermillion Energy", "Grand 6-24TH", "DFIT")`` -- one real datetime among three
    other strings, 25% numeric, correctly rejected in favor of the real header 3 rows further
    down; GMT's "... dfit pump info.xlsx" has a similar "Date" / "08/08/2018" (string, not a real
    date value) key/value preamble pair whose only lookahead evidence is a lone ``datetime.time``
    next to more label text, also 25% pooled and rejected.
    """
    non_empty = 0
    numeric = 0
    numeric_cols: set = set()
    for row in rows:
        for j, c in enumerate(row):
            if c is None or (isinstance(c, str) and c.strip() == ""):
                continue
            non_empty += 1
            if _xlsx_is_number(c) or _xlsx_is_datetimeish(c):
                numeric += 1
                numeric_cols.add(j)
    if non_empty == 0:
        return False
    return (numeric / non_empty) >= 0.5 and len(numeric_cols) >= 2


def _xlsx_find_header(rows: list) -> Optional[tuple]:
    """The header row of an XLSX time-series sheet: the first row (within the first
    ``_XLSX_HEADER_SCAN_ROWS`` of `rows`) with at least 2 non-empty string cells, at least one of
    which matches a datetime/time name needle (``_DATETIME_NAME_NEEDLES`` -- the same ones
    ``suggest_channels`` uses), no real number or datetime/date/time value anywhere in the row
    ITSELF (a genuine header is text-only; a numeric tally/schedule row that happens to carry an
    incidental "time"-ish label in one cell is not a header -- measured corpus false positives:
    a casing-tally row naming a component "RESOURCE TIME DELAY SLEEVE" among a run of real
    joint-length numbers, and a completion-notes row pairing a "Start pump time:" label with a
    real logged `datetime` two cells over), and with real, DATA-shaped numeric or datetime
    content in the next ``_XLSX_DATA_LOOKAHEAD_ROWS`` rows (``_xlsx_data_window``,
    ``_xlsx_looks_like_data_rows`` -- pooled >=50% numeric/datetime across >=2 columns, not
    merely "any cell", which a key/value preamble row can satisfy with a single number or date
    next to a label; see ``_xlsx_looks_like_data_rows``'s own docstring for two measured corpus
    cases this rejects). Returns ``(row_index, row_cells)`` (0-based within `rows`), or ``None``.
    """
    scan_limit = min(len(rows), _XLSX_HEADER_SCAN_ROWS)
    for i in range(scan_limit):
        row = rows[i]
        str_cells = [c.strip() for c in row if isinstance(c, str) and c.strip()]
        if len(str_cells) < 2:
            continue
        if not any(any(n in c.lower() for n in _DATETIME_NAME_NEEDLES) for c in str_cells):
            continue
        if any(_xlsx_is_number(c) or _xlsx_is_datetimeish(c) for c in row):
            continue
        window = _xlsx_data_window(rows, i + 1, _XLSX_DATA_LOOKAHEAD_ROWS)
        if not _xlsx_looks_like_data_rows(window):
            continue
        return i, row
    return None


def _xlsx_open(path: str):
    import openpyxl  # lazy: costs ~0.4s to import and this path isn't hit on every app start.
    return openpyxl.load_workbook(path, read_only=True, data_only=True)


def _xlsx_merge_two_row_header(rows: list, header_idx: int, header_cells: tuple) -> tuple:
    """Fold a name from the row directly ABOVE the chosen header row into any of `header_cells`'
    own BLANK cells -- the two-row-header shape a real corpus file can have (Sanjel's "DFIT
    Data.xlsx": the row above the header carries "Pump\\nBPM"/"Annulus\\nPSI"/"Tbg\\nPSI"/
    "dVol\\nbbls" for exactly the columns the header row itself ("Date", "Time", ..., "FluidIdx",
    "StageIdx") leaves blank). Only a genuinely blank header cell is ever filled -- a header cell
    that already carries its own text is never overwritten by whatever sits above it, so an
    ordinary one-row header (no useful text directly above it, or nothing blank to fill) is
    untouched. No merge at all when `header_idx == 0` (no row above to look at).
    """
    if header_idx == 0:
        return header_cells
    above = rows[header_idx - 1]
    merged = list(header_cells)
    for j, cell in enumerate(merged):
        text = "" if cell is None else str(cell).strip()
        if text:
            continue
        above_cell = above[j] if j < len(above) else None
        above_text = "" if above_cell is None else str(above_cell).strip()
        if above_text:
            merged[j] = above_text
    return tuple(merged)


def _xlsx_scan_sheet(ws) -> Optional[dict]:
    """Peek at `ws`'s first rows (see ``_xlsx_find_header``) and decide whether it qualifies as
    an XLSX data sheet. Returns a dict with ``header_cells``, ``units_cells`` (or ``None``), and
    ``data_start_row`` (0-based, the first row past the header and any units row) on a qualifying
    sheet, else ``None``. ``header_cells`` has already been through ``_xlsx_merge_two_row_header``,
    so a blank header cell with a real name sitting directly above it (a two-row header) is filled
    before anything downstream (folding, deduping) ever sees it.
    """
    peek_n = _XLSX_HEADER_SCAN_ROWS + _XLSX_DATA_LOOKAHEAD_ROWS + 1
    rows = list(ws.iter_rows(min_row=1, max_row=peek_n, values_only=True))
    found = _xlsx_find_header(rows)
    if found is None:
        return None
    header_idx, header_cells = found
    header_cells = _xlsx_merge_two_row_header(rows, header_idx, header_cells)
    units_cells = None
    data_start = header_idx + 1
    if data_start < len(rows) and _xlsx_row_is_units_row(rows[data_start]):
        units_cells = rows[data_start]
        data_start += 1
    return {"header_cells": header_cells, "units_cells": units_cells,
            "data_start_row": data_start}


_XLSX_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_XLSX_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_XLSX_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_XLSX_SNIFF_PEEK_ROWS = _XLSX_HEADER_SCAN_ROWS + _XLSX_DATA_LOOKAHEAD_ROWS + 1
_XLSX_COL_REF_RE = re.compile(r"([A-Z]+)\d*")


def _xlsx_col_ref_index(ref: Optional[str]) -> Optional[int]:
    """0-based column index from a cell's ``r`` attribute (e.g. ``"C5"`` -> 2), or ``None`` if
    `ref` is missing or unparseable -- a row's cells in the raw worksheet XML are sparse (a blank
    cell is usually simply omitted, not written out as an empty ``<c>``), so this is what lets
    ``_xlsx_zip_peek_rows`` place each cell at its real column position instead of the next
    position in document order, which would silently misalign every cell after the first gap."""
    if not ref:
        return None
    m = _XLSX_COL_REF_RE.match(ref)
    if not m:
        return None
    idx = 0
    for ch in m.group(1):
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1 if idx else None


def _xlsx_worksheet_parts(z: zipfile.ZipFile) -> list[str]:
    """The zip member names of every real WORKSHEET part in `z`, resolved via ``xl/workbook.xml``
    (its ``<sheets>`` list, in workbook order) and ``xl/_rels/workbook.xml.rels`` (the ``r:id`` ->
    target mapping) -- the only reliable way to tell a worksheet part from a chartsheet/
    dialogsheet/macrosheet, which can also live under a member name that superficially looks like
    a worksheet's. Falls back to every zip member matching the ordinary
    ``xl/worksheets/sheetN.xml`` naming convention, sorted, if the workbook/rels parts are
    missing, malformed, or reference something not actually in the archive -- this also already
    excludes chart sheets on its own (those live under ``xl/chartsheets/``), so a corrupt or
    unconventional ``workbook.xml`` never turns into "no sheets found at all". Never raises.
    """
    try:
        with z.open("xl/workbook.xml") as f:
            wb_root = _ET.parse(f).getroot()
        sheet_rids = [
            el.get(_XLSX_DOC_REL_NS + "id") for el in wb_root.iter(_XLSX_MAIN_NS + "sheet")
        ]
        with z.open("xl/_rels/workbook.xml.rels") as f:
            rel_root = _ET.parse(f).getroot()
        target_by_id = {
            el.get("Id"): el.get("Target")
            for el in rel_root.iter(_XLSX_PKG_REL_NS + "Relationship")
        }
        names = set(z.namelist())
        parts = []
        for rid in sheet_rids:
            target = target_by_id.get(rid)
            if not target:
                continue
            norm = target.replace("\\", "/").lstrip("/")
            path = norm if norm.startswith("xl/") else f"xl/{norm}"
            if path in names and "worksheets/" in path:
                parts.append(path)
        if parts:
            return parts
    except Exception:
        pass
    return sorted(
        n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")
    )


def _xlsx_zip_peek_rows(z: zipfile.ZipFile, name: str, max_rows: int) -> list:
    """Up to `max_rows` rows from worksheet part `name`, stream-parsed with ``iterparse`` rather
    than a full-DOM read -- cheap even against a sheet with a million real (or, per FIX 5,
    falsely-declared) rows, since parsing stops as soon as enough rows are collected. Each row is
    a list of ``(kind, value)`` cell entries (or ``None`` for a blank cell), gap-filled by column
    position from each cell's own ``r`` attribute (see ``_xlsx_col_ref_index``) so a sparse row
    (blank cells simply omitted from the XML) still lines up column-for-column with every other
    row: ``"s"`` (a shared-string index, resolved later by ``_xlsx_zip_shared_strings``), ``"str"``
    (an inline string), ``"b"`` (boolean), or ``"n"`` (a plain number -- a date/time cell is ALSO
    stored as a plain number in the raw XML; its number-FORMAT lives in styles.xml, which this
    never reads, but ``_xlsx_find_header``'s rule treats a number and a date identically anyway,
    so no style lookup is needed here).

    ROWS are gap-filled too, the same way columns are: a `<row>` element also carries its own
    ``r`` attribute (its real, 1-based row number), and a fully-blank row is routinely omitted
    from the XML entirely, exactly like a blank cell within a row is. Without padding those gaps
    back in as empty rows, this function's OWN output list would silently compress -- a real
    row's TEXT is unchanged, but its INDEX shifts earlier than its true worksheet row number by
    however many blank rows preceded it, which can make ``_xlsx_find_header``'s scan land on
    (or reject) entirely the wrong row purely by coincidence of how many blank rows happened to
    sit above it. Measured corpus case: a "Notes-Well Data" sheet's real row 8 ("Notes:", a long
    free-text comment that happens to contain the substring "time") shifted to index 6 with no
    padding, landing its (false-positive) header-candidacy two rows earlier than intended and
    changing which lookahead rows got checked against it.

    Returns `[]`, never raises, on any parse failure.
    """
    rows: list = []
    try:
        with z.open(name) as f:
            cur: Optional[list] = None
            cur_row_num: Optional[int] = None
            next_expected = 1  # 1-based -- the next row NUMBER rows[] should hold, gaps padded
            for event, el in _ET.iterparse(f, events=("start", "end")):
                tag = el.tag
                if event == "start" and tag == _XLSX_MAIN_NS + "row":
                    cur = []
                    r_attr = el.get("r")
                    try:
                        cur_row_num = int(r_attr) if r_attr else next_expected
                    except ValueError:
                        cur_row_num = next_expected
                elif event == "end" and tag == _XLSX_MAIN_NS + "c":
                    if cur is not None:
                        col = _xlsx_col_ref_index(el.get("r"))
                        if col is None:
                            col = len(cur)
                        while len(cur) <= col:
                            cur.append(None)
                        t = el.get("t")
                        v = el.find(_XLSX_MAIN_NS + "v")
                        if t == "s" and v is not None and v.text is not None:
                            try:
                                cur[col] = ("s", int(v.text))
                            except ValueError:
                                cur[col] = None
                        elif t == "inlineStr":
                            isv = el.find(_XLSX_MAIN_NS + "is")
                            cur[col] = ("str", "".join(isv.itertext()) if isv is not None else "")
                        elif t == "str" and v is not None:
                            cur[col] = ("str", v.text)
                        elif t == "b" and v is not None:
                            cur[col] = ("b", v.text)
                        elif v is not None and v.text is not None:
                            cur[col] = ("n", v.text)
                    el.clear()
                elif event == "end" and tag == _XLSX_MAIN_NS + "row":
                    if cur is not None:
                        # Pad in any fully-blank rows the XML omitted between the last row
                        # emitted and this one, so ROW position (not just column position, see
                        # _xlsx_col_ref_index above) stays aligned with the real worksheet -- a
                        # row with zero non-blank cells is routinely omitted from the XML
                        # entirely, exactly like a blank cell within a row is. Without this, a
                        # header/data row's TEXT content is unchanged but its INDEX within
                        # `rows` silently shifts earlier than its real worksheet row number,
                        # which can make `_xlsx_find_header`'s scan land on the wrong row -- or
                        # find a false header candidate several rows too early -- entirely by
                        # coincidence of which rows happened to be blank above it.
                        if cur_row_num is None:
                            cur_row_num = next_expected
                        while next_expected < cur_row_num and len(rows) < max_rows:
                            rows.append([])
                            next_expected += 1
                        if len(rows) < max_rows:
                            rows.append(cur)
                            next_expected = cur_row_num + 1
                    cur = None
                    cur_row_num = None
                    el.clear()
                    if len(rows) >= max_rows:
                        break
    except Exception:
        return rows
    return rows


def _xlsx_zip_shared_strings(z: zipfile.ZipFile, need: set) -> dict:
    """The shared-string table entries at indices in `need`, or `{}` if `path` has none (a
    workbook with no shared strings at all -- every string is inline) or the part fails to parse.
    Stream-parsed and stopped as soon as every needed index has been passed, rather than reading
    the whole table -- `need` is only ever the handful of indices actually referenced by the
    peeked rows, never the workbook's full string pool.
    """
    if not need or "xl/sharedStrings.xml" not in z.namelist():
        return {}
    out: dict = {}
    try:
        mx = max(need)
        i = 0
        with z.open("xl/sharedStrings.xml") as f:
            for event, el in _ET.iterparse(f, events=("end",)):
                if el.tag == _XLSX_MAIN_NS + "si":
                    if i in need:
                        out[i] = "".join(el.itertext())
                    i += 1
                    el.clear()
                    if i > mx:
                        break
    except Exception:
        return out
    return out


def _xlsx_zip_convert_row(row: list, shared: dict) -> tuple:
    """One `_xlsx_zip_peek_rows` row of raw ``(kind, value)`` entries into the same shape
    ``_xlsx_find_header`` already expects from openpyxl's ``values_only`` rows: a plain string, a
    ``float``, a ``bool``, or ``None``."""
    out = []
    for c in row:
        if c is None:
            out.append(None)
            continue
        kind, val = c
        if kind == "s":
            out.append(shared.get(val))
        elif kind == "str":
            out.append(val)
        elif kind == "b":
            out.append(val not in (None, "0", "", "false", "False"))
        elif kind == "n":
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                out.append(val)
        else:
            out.append(val)
    return tuple(out)


def sniff_xlsx_data(path: str) -> bool:
    """Cheap check: True iff some worksheet in `path` has a header row within its first
    ``_XLSX_HEADER_SCAN_ROWS`` rows naming a datetime/date/time/timestamp column, with real,
    data-shaped numeric or datetime content in the rows just below it (``_xlsx_find_header``,
    the same rule ``load_xlsx``/``_xlsx_scan_sheet`` use) -- i.e. it looks like time-series data,
    not a summary/tally/schedule workbook.

    Implemented as a direct zip/XML peek (``_xlsx_worksheet_parts``, ``_xlsx_zip_peek_rows``,
    ``_xlsx_zip_shared_strings``) rather than opening the workbook with openpyxl: only the first
    ``_XLSX_SNIFF_PEEK_ROWS`` rows of each worksheet part's raw XML are ever stream-parsed, and
    ``sharedStrings.xml`` is read only up to the largest string index any peeked cell actually
    references -- openpyxl's own reader has no equivalent short-circuit and pays for the whole
    worksheet part's XML structure regardless of how many rows are asked for, measured at
    ~0.18s -> ~32s scanning a real corpus folder with 113 candidate `.xlsx` files once this sniff
    is gated into `store._group_data_files`, most of it openpyxl's own per-file open cost.
    Chart/dialog/macro sheets are excluded by ``_xlsx_worksheet_parts`` (via workbook.xml's own
    sheet list, or, failing that, by the worksheet-part zip-member naming convention). Never
    raises and always closes the zip: a corrupt, password-protected (not a real zip at all,
    or a real zip without the expected parts), or otherwise unreadable workbook just returns
    False, like any other unreadable sheet. Stops at the first qualifying sheet -- ``load_xlsx``
    is what picks the winning sheet by real row count when there's more than one (FIX 5).
    """
    try:
        with zipfile.ZipFile(path) as z:
            try:
                parts = _xlsx_worksheet_parts(z)
            except Exception:
                parts = []
            if not parts:
                return False
            raw_by_part: dict = {}
            need: set = set()
            for part in parts:
                rows = _xlsx_zip_peek_rows(z, part, _XLSX_SNIFF_PEEK_ROWS)
                raw_by_part[part] = rows
                for row in rows:
                    for c in row:
                        if c is not None and c[0] == "s":
                            need.add(c[1])
            try:
                shared = _xlsx_zip_shared_strings(z, need)
            except Exception:
                shared = {}
            for rows in raw_by_part.values():
                try:
                    conv_rows = [_xlsx_zip_convert_row(r, shared) for r in rows]
                    if _xlsx_find_header(conv_rows) is not None:
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        return False


def _xlsx_fold_unit(header: str, unit_cell: object) -> str:
    """Fold one units-row cell into its column's header name as ``"Name (unit)"``, stripping any
    existing parenthesized suffix from `header` first (so a header that already carries one
    isn't doubled up). `unit_cell` wrapped in its own parens (e.g. ``"(psi)"``) has them stripped
    before folding, so the result is always a single, clean ``"(unit)"`` suffix -- e.g.
    ``"Time"`` + ``"(hh:mm:ss)"`` -> ``"Time (hh:mm:ss)"``; ``"Tubing Pressure"`` + ``"psi"`` ->
    ``"Tubing Pressure (psi)"``. Returns `header` unchanged when `unit_cell` is blank or missing
    (no units row, or a blank cell in that column of it).
    """
    if unit_cell is None:
        return header
    text = str(unit_cell).strip()
    if not text:
        return header
    inner = _unit_of(text)
    unit_str = inner if inner is not None else text
    base = _UNIT_RE.sub("", header).strip()
    # A blank header cell with only a units-row label (e.g. Southern Ute's unnamed "DeltaP"
    # column -- a real data column, just missing its own header text) gets that label AS its
    # name outright, unparenthesized -- "(unit)" folded onto an empty base would otherwise read
    # as a bare, meaningless "(DeltaP)" column name.
    if not base:
        return unit_str
    return f"{base} ({unit_str})"


def _xlsx_dedupe_headers(names: list) -> list:
    """Make header cells unique the same way ``pd.read_csv`` does: a blank cell becomes
    ``"Unnamed: N"`` (0-based by position), and a name that repeats later gets ``".1"``, ``".2"``,
    ... appended.
    """
    filled = []
    for i, name in enumerate(names):
        text = "" if name is None else str(name).strip()
        filled.append(text if text else f"Unnamed: {i}")
    counts: dict = {}
    out = []
    for name in filled:
        if name not in counts:
            counts[name] = 0
            out.append(name)
            continue
        counts[name] += 1
        candidate = f"{name}.{counts[name]}"
        while candidate in counts:
            counts[name] += 1
            candidate = f"{name}.{counts[name]}"
        counts[candidate] = 0
        out.append(candidate)
    return out


def _xlsx_trim_trailing_empty(rows: list, header: list) -> tuple:
    """Drop trailing all-blank columns (an auto-generated ``"Unnamed: N"`` header AND blank in
    every data row) and trailing all-blank data rows -- a real Excel "used range" routinely
    over-reports its own extent by a stray formatted-but-empty cell or row. Only ever trims from
    the tail; a blank column or row sitting in the middle of real data is left alone.
    """
    def _blank(v: object) -> bool:
        return v is None or (isinstance(v, str) and v.strip() == "")

    def _col_all_blank(j: int) -> bool:
        if not header[j].startswith("Unnamed: "):
            return False
        return all(_blank(row[j]) if j < len(row) else True for row in rows)

    end = len(header)
    while end > 0 and _col_all_blank(end - 1):
        end -= 1
    header = header[:end]
    rows = [row[:end] for row in rows]

    def _row_all_blank(row: tuple) -> bool:
        return all(_blank(v) for v in row)

    trimmed = list(rows)
    while trimmed and _row_all_blank(trimmed[-1]):
        trimmed.pop()
    return trimmed, header


def _xlsx_cell_to_value(cell: object, date_only: bool) -> object:
    """Convert one openpyxl cell into what ``_finish_frame`` (shared with the CSV loader)
    expects: a ``datetime`` cell becomes a string in the primary fast-path format
    ``parse_datetime`` recognizes (``_PRIMARY_DT_FORMAT``), or, when it carries a nonzero
    ``microsecond``, the fractional-seconds fast-path format instead (``_PRIMARY_DT_FORMAT_US``,
    ``parse_datetime``'s fast path 3) so a millisecond-stamped workbook (e.g. a downhole gauge
    logged sub-second) doesn't get truncated to whole seconds on the way through; a
    ``datetime.time`` cell becomes an ``"HH:MM:SS"`` string, or ``"HH:MM:SS.%f"`` under the same
    nonzero-microsecond rule; a bare ``datetime.date`` becomes a date-only string; and, when
    `date_only` is set (this column's cells are exactly midnight and paired with a separate time
    column -- see ``load_xlsx``), a midnight ``datetime`` cell is formatted date-only too, exactly
    like a CSV's own date-only companion column, so the FIX-A join in ``_finish_frame`` sees the
    same shape either way (a cell that is exactly midnight by definition has no fractional
    seconds either, so this never collides with the microsecond case above). A number or blank
    cell passes through unchanged.
    """
    if isinstance(cell, _dt.datetime):
        if date_only and (cell.hour, cell.minute, cell.second, cell.microsecond) == (0, 0, 0, 0):
            return cell.strftime("%m/%d/%Y")
        if cell.microsecond:
            return cell.strftime("%m/%d/%Y %H:%M:%S.%f")
        return cell.strftime(_PRIMARY_DT_FORMAT)
    if isinstance(cell, _dt.time):
        if cell.microsecond:
            return cell.strftime("%H:%M:%S.%f")
        return cell.strftime("%H:%M:%S")
    if isinstance(cell, _dt.date):
        return cell.strftime("%m/%d/%Y")
    return cell


_XLSX_ROW_COUNT_BLANK_STOP = 1000
# Excel's own hard sheet-row limit -- passed explicitly as `iter_rows`' `max_row` everywhere this
# module reads a sheet's real data, so iteration is never silently capped by the worksheet's own
# declared `<dimension>` instead. openpyxl's READ-ONLY mode takes `ws.max_row`/`ws.dimensions`
# straight from that declaration (it never scans cells to recompute it, which is what makes
# read-only mode cheap) -- correct when the dimension is accurate or over-states the real extent
# (an `iter_rows()` call still stops at the last row actually present in the XML either way), but
# a dimension that UNDER-states the real extent (stale after an edit that removed the tag's own
# update, or otherwise just wrong) silently truncates iteration at that too-small bound if `iter_rows`
# is ever called with no explicit `max_row`, or with one derived from `ws.max_row` itself --
# exactly the same failure this module's sheet-choice fix exists to avoid, just reachable at the
# READ step instead of the ranking step. Passing this constant costs nothing extra for a
# well-formed file: `iter_rows` still only reads the `<row>` elements actually present in the
# XML, regardless of how high `max_row` is set.
_XLSX_MAX_ROW_CEILING = 1_048_576


def _xlsx_count_data_rows(ws, data_start_row: int) -> int:
    """The real, non-blank data row count for `ws` below `data_start_row` (0-based, past the
    header and any units row) -- used to rank candidate sheets by ACTUAL size, never by
    ``ws.max_row``/``ws.dimensions``. Those are only as good as the worksheet's declared
    ``<dimension>``, which a real workbook can wildly over-report (measured: a sheet with 669
    real data rows whose ``max_row`` reports 1,047,735 -- an inflated "used range" from stray
    formatting far past the last real row) or under-report -- down to whatever openpyxl falls
    back to (often 1) when the ``<dimension>`` element is missing entirely, or to a stale,
    too-small real value when the tag is simply wrong -- either of which can rank a real,
    populated data sheet below a much smaller notes/comments sheet. `max_row=
    _XLSX_MAX_ROW_CEILING` on the walk below is what makes the under-report case safe too: see
    that constant's own comment.

    Counts every row with at least one non-blank cell, stopping early -- without walking the rest
    of a huge, mostly-empty "used range" -- once ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive
    fully-blank rows are seen. Never walks a million empty rows just to confirm there is nothing
    there. This is the openpyxl fallback; ``load_xlsx`` normally counts with the much cheaper
    zip-level ``_xlsx_zip_count_data_rows`` (same rule).
    """
    count = 0
    consec_blank = 0
    for row in ws.iter_rows(
        min_row=data_start_row + 1, max_row=_XLSX_MAX_ROW_CEILING, values_only=True
    ):
        if all(c is None for c in row):
            consec_blank += 1
            if consec_blank >= _XLSX_ROW_COUNT_BLANK_STOP:
                break
        else:
            consec_blank = 0
            count += 1
    return count


def _xlsx_zip_count_data_rows_et(z: zipfile.ZipFile, part: str, data_start_row: int) -> int:
    """ElementTree ``iterparse`` version of ``_xlsx_zip_count_data_rows`` -- the fallback for a
    sheet part whose rows the byte-regex scan cannot find (e.g. a prefixed ``<x:row>`` namespace).
    A ``<row>`` counts when any of its cells
    holds a ``<v>`` or ``<is>`` with text; a row number skipped in the XML is a blank row (the
    way openpyxl's padded iteration treats it), and ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive
    blank rows -- omitted or present-but-empty -- end the count, so a styled cell near row
    1,048,000 is never walked to."""
    row_tag = _XLSX_MAIN_NS + "row"
    v_tag = _XLSX_MAIN_NS + "v"
    is_tag = _XLSX_MAIN_NS + "is"
    first = data_start_row + 1  # 1-based
    count = 0
    blank = 0
    last_r = first - 1
    with z.open(part) as f:
        for _event, el in _ET.iterparse(f, events=("end",)):
            if el.tag != row_tag:
                continue
            r_attr = el.get("r")
            try:
                r = int(r_attr) if r_attr else last_r + 1
            except ValueError:
                r = last_r + 1
            if r < first:
                el.clear()
                continue
            blank += max(0, r - last_r - 1)
            last_r = r
            if blank >= _XLSX_ROW_COUNT_BLANK_STOP:
                break
            has = False
            for c in el:
                v = c.find(v_tag)
                if v is not None and v.text is not None:
                    has = True
                    break
                isv = c.find(is_tag)
                if isv is not None and "".join(isv.itertext()):
                    has = True
                    break
            el.clear()
            if has:
                blank = 0
                count += 1
            else:
                blank += 1
                if blank >= _XLSX_ROW_COUNT_BLANK_STOP:
                    break
    return count


_XLSX_ROW_RE = re.compile(rb"<row\b([^>]*?)(?:/>|>(.*?)</row>)", re.S)
_XLSX_ROW_R_RE = re.compile(rb'\br="(\d+)"')
_XLSX_ROW_VALUE_RE = re.compile(rb"<v>[^<]|<t(?:\s[^>]*)?>[^<]")


# Bytes of worksheet XML ``_xlsx_zip_count_data_rows`` reads per block.
_XLSX_COUNT_BLOCK_BYTES = 1 << 22


def _xlsx_count_region_fast(region: bytes, st: list) -> bool:
    """C-speed path of ``_xlsx_zip_count_data_rows`` for a block of complete ``<row>`` elements:
    applies only when every row has an explicit number continuing exactly from the last one
    (no gaps, no self-closing rows, no inline strings, no empty ``<v>``) and every row holds a
    ``<v>`` -- i.e. nothing for the blank-run rule to see. Updates `st` ([count, blank, last_r,
    seen]) and returns True; returns False, touching nothing, when any condition fails."""
    n = region.count(b"</row>")
    if (n == 0 or region.count(b"<row ") != n or region.count(b"<is") or
            region.count(b"<v></v>") or region.count(b"<v/>")):
        return False
    i = region.find(b"<row ")
    j = region.rfind(b"<row ")
    mi = _XLSX_ROW_R_RE.search(region, i, i + 200)
    mj = _XLSX_ROW_R_RE.search(region, j, j + 200)
    if mi is None or mj is None:
        return False
    if int(mi.group(1)) != st[2] + 1 or int(mj.group(1)) != st[2] + n:
        return False
    if region.count(b"<v>") < n:
        return False
    with_v = sum(1 for p in region.split(b"</row>") if b"<v>" in p)
    if with_v != n:
        return False
    st[0] += n
    st[1] = 0
    st[2] += n
    st[3] = True
    return True


def _xlsx_count_region_slow(region: bytes, st: list, first: int) -> bool:
    """Exact per-row version of ``_xlsx_count_region_fast`` (gaps, blanks, header rows above
    `first`). Returns True once ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive blank rows are hit."""
    for m in _XLSX_ROW_RE.finditer(region):
        st[3] = True
        rm = _XLSX_ROW_R_RE.search(m.group(1))
        r = int(rm.group(1)) if rm else st[2] + 1
        if r < first:
            continue
        st[1] += max(0, r - st[2] - 1)
        st[2] = r
        if st[1] >= _XLSX_ROW_COUNT_BLANK_STOP:
            return True
        body = m.group(2)
        if body is not None and _XLSX_ROW_VALUE_RE.search(body):
            st[1] = 0
            st[0] += 1
        else:
            st[1] += 1
            if st[1] >= _XLSX_ROW_COUNT_BLANK_STOP:
                return True
    return False


def _xlsx_zip_count_data_rows(z: zipfile.ZipFile, part: str, data_start_row: int) -> int:
    """Same count as ``_xlsx_count_data_rows`` but straight off worksheet part `part`'s raw XML
    bytes -- no openpyxl cell objects, shared-string lookups, or number-format conversion, and no
    ElementTree event machinery either (an ``iterparse`` count still cost ~6 s per 700k-row
    sheet). A ``<row>`` counts when its body holds a ``<v>`` with text or an inline-string ``<t>``
    with text; a row number skipped in the XML is a blank row (the way openpyxl's padded
    iteration treats it), and ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive blank rows -- omitted or
    present-but-empty -- end the count, so a styled cell near row 1,048,000 is never walked to.

    Works in 4 MB blocks of complete rows. A block of plain, gap-free, fully populated rows (the
    ordinary data region of a gauge export) is counted with a few C-level byte operations
    (``_xlsx_count_region_fast``); any block with a gap, blank, self-closing or inline-string
    row, and the header rows above the data, take the exact per-row regex path
    (``_xlsx_count_region_slow``). Falls back to the ElementTree version when no ``<row>`` is
    found at all (an unusual namespace prefix)."""
    first = data_start_row + 1  # 1-based
    st = [0, 0, first - 1, False]  # count, consecutive blank, last row number, saw any row
    buf = b""
    with z.open(part) as f:
        while True:
            chunk = f.read(_XLSX_COUNT_BLOCK_BYTES)
            buf += chunk
            idx = buf.rfind(b"</row>")
            if idx < 0:
                if not chunk:
                    break
                continue
            idx += len(b"</row>")
            region, buf = buf[:idx], buf[idx:]
            if not _xlsx_count_region_fast(region, st):
                if _xlsx_count_region_slow(region, st, first):
                    break
            if not chunk:
                break
    if not st[3]:
        return _xlsx_zip_count_data_rows_et(z, part, data_start_row)
    return st[0]


def _xlsx_read_sheet_frame(ws, scan: dict, limit: Optional[int] = None) -> tuple:
    """`ws`'s data rows (below `scan`'s header/units rows) as a raw ``(DataFrame, columns)``
    pair, converted the way ``load_xlsx`` documents. Reads to Excel's row ceiling
    (``_XLSX_MAX_ROW_CEILING``) but stops after ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive
    fully-blank rows and drops that trailing blank run, so a styled cell near row 1,048,000
    never pads the frame with ~1M empty rows. `limit` caps the number of data rows read (a cheap
    sample). Raises ValueError when no data row survives.
    """
    header_cells = list(scan["header_cells"])
    units_cells = scan["units_cells"]
    folded = [
        _xlsx_fold_unit(
            "" if h is None else str(h).strip(),
            units_cells[j] if units_cells is not None and j < len(units_cells) else None,
        )
        for j, h in enumerate(header_cells)
    ]
    columns = _xlsx_dedupe_headers(folded)

    data_start_1based = scan["data_start_row"] + 1
    # max_row=_XLSX_MAX_ROW_CEILING, not ws.max_row: the latter is read straight off this
    # sheet's own declared <dimension> in read-only mode (never recomputed by scanning
    # cells), so a stale/wrong dimension that UNDER-reports the real extent would silently
    # truncate this read at that too-small bound -- see _XLSX_MAX_ROW_CEILING's own comment.
    data_rows: list = []
    blank = 0
    gap_stop_row = None
    for row in ws.iter_rows(
        min_row=data_start_1based, max_row=_XLSX_MAX_ROW_CEILING, max_col=len(header_cells),
        values_only=True,
    ):
        data_rows.append(row)
        if all(c is None for c in row):
            blank += 1
            if blank >= _XLSX_ROW_COUNT_BLANK_STOP:
                gap_stop_row = data_start_1based + len(data_rows) - 1  # 1-based, last blank row
                break
        else:
            blank = 0
            if limit is not None and len(data_rows) >= limit:
                break
    if blank:
        del data_rows[len(data_rows) - blank:]

    data_rows, columns = _xlsx_trim_trailing_empty(data_rows, columns)
    if not data_rows:
        raise ValueError(f"sheet {ws.title!r} has a header row but no data rows")

    # A date-with-midnight-only column paired with a separate time column (see load_xlsx's
    # docstring) is formatted date-only -- the same date-with-companion pairing
    # suggest_channels/_companion_time_col would find, checked against the FINAL column names so
    # it matches whatever _finish_frame will actually see.
    guess = suggest_channels(columns)
    date_only_col = guess["datetime"] if guess["datetime"] and guess["time"] else None

    cols_out: dict = {c: [] for c in columns}
    ncols = len(columns)
    for row in data_rows:
        for j in range(ncols):
            c = columns[j]
            cell = row[j] if j < len(row) else None
            cols_out[c].append(_xlsx_cell_to_value(cell, date_only=(c == date_only_col)))
    out = pd.DataFrame(cols_out)
    if gap_stop_row is not None:
        out.attrs["xlsx_gap_stop_row"] = gap_stop_row
    return out, columns


def _xlsx_rows_after(z: zipfile.ZipFile, part: str, after_row: int) -> int:
    """Count of rows numbered past `after_row` (1-based) in worksheet part `part` that hold a
    value -- a zip-level scan with no blank-run stop (ElementTree ``iterparse``, rows cleared as
    they go)."""
    row_tag = _XLSX_MAIN_NS + "row"
    v_tag = _XLSX_MAIN_NS + "v"
    is_tag = _XLSX_MAIN_NS + "is"
    count = 0
    last_r = 0
    with z.open(part) as f:
        for _event, el in _ET.iterparse(f, events=("end",)):
            if el.tag != row_tag:
                continue
            r_attr = el.get("r")
            try:
                r = int(r_attr) if r_attr else last_r + 1
            except ValueError:
                r = last_r + 1
            last_r = r
            if r > after_row:
                for c in el:
                    v = c.find(v_tag)
                    isv = c.find(is_tag)
                    if (v is not None and v.text) or (
                            isv is not None and "".join(isv.itertext())):
                        count += 1
                        break
            el.clear()
    return count


def _xlsx_gap_stop_warning(wb, path: str, ws, stop_row: int) -> Optional[str]:
    """A load warning when the blank-row stop at `stop_row` cut off later rows that hold values;
    None when nothing lies past it (or the zip scan cannot run). Never raises."""
    try:
        with zipfile.ZipFile(path) as z:
            parts = _xlsx_worksheet_parts(z)
            if len(parts) != len(wb.worksheets):
                return None
            part = parts[[w.title for w in wb.worksheets].index(ws.title)]
            n = _xlsx_rows_after(z, part, stop_row)
    except Exception:
        return None
    if n <= 0:
        return None
    return (f"Sheet {ws.title!r}: {n} rows after a blank gap at row {stop_row} not read")


# Data rows read from a sheet to judge whether its time base is usable when more than one sheet
# qualifies (see ``load_xlsx``).
_XLSX_USABILITY_SAMPLE_ROWS = 2000


def _xlsx_span_s(td: TestData) -> float:
    """Finite max - min of `td.t_s`; 0.0 when undefined."""
    f = td.t_s[np.isfinite(td.t_s)]
    return float(f.max() - f.min()) if f.size else 0.0


def _xlsx_sheet_time_base_usable(ws, scan: dict, path: str) -> bool:
    """True when the first ``_XLSX_USABILITY_SAMPLE_ROWS`` data rows of `ws` run through
    ``_finish_frame`` give >=2 distinct timestamps with a non-zero span. Measured case: a
    Strathcona workbook's raw "Original Data" sheet (more rows than its "Cleaned Data" sheet)
    has a midnight-only Date/Time column and no separate time column, so it has ONE distinct
    timestamp. Never raises."""
    try:
        df, _cols = _xlsx_read_sheet_frame(ws, scan, limit=_XLSX_USABILITY_SAMPLE_ROWS)
        return _xlsx_span_s(_finish_frame(df, path)) > 0.0
    except Exception:
        return False


def load_xlsx(path: str) -> TestData:
    """Load a DFIT time-series XLSX workbook and attach an elapsed-seconds time base.

    Opened read-only (``openpyxl.load_workbook(read_only=True, data_only=True)``); chart sheets
    are skipped automatically (``Workbook.worksheets`` never includes them). Each remaining
    worksheet is scanned for a header row within its first 60 rows (``_xlsx_find_header``): the
    first row with >=2 non-empty string cells, one matching a datetime/date/time/timestamp name
    needle, with real numeric or datetime data in the next few rows below it. A units row
    directly below the header (``_xlsx_row_is_units_row`` -- mostly non-numeric short strings:
    parenthesized tokens, or unit/format tokens like "psi"/"bpm"/"gal"/"deg F"/"minutes"/
    "MM/DD/YY"/"HH:MM:SS") is detected and folded into the header names as ``"Name (unit)"``
    (``_xlsx_fold_unit``) before being dropped, so the existing header-suffix unit detection
    (``_unit_of``) sees it exactly as it would from a CSV -- e.g. a bare "Time" header paired
    with a "(hh:mm:ss)" units cell folds to ``"Time (hh:mm:ss)"``.

    When more than one worksheet qualifies they are ranked by (time base usable, real data
    rows): "usable" means the first ``_XLSX_USABILITY_SAMPLE_ROWS`` data rows, run through
    ``_finish_frame``, give a non-zero elapsed span; the row count is the REAL non-blank count
    (``_xlsx_zip_count_data_rows``, a zip-level streaming byte count with the same early
    stop as ``_xlsx_count_data_rows``), taken only when more than one usable sheet needs ordering
    (a lone sheet, or a lone usable sheet, is never counted) -- never ``ws.max_row``/``ws.dimensions``, which a worksheet's
    declared ``<dimension>`` can badly over- or under-report. The ranked sheets are then tried in
    order: a sheet whose full load fails or yields a zero/undefined span falls through to the
    next. If every sheet falls through, a zero span raises ``ValueError("... no sheet with a
    usable time base")``; otherwise the top-ranked sheet's own error is re-raised. The data read
    stops after ``_XLSX_ROW_COUNT_BLANK_STOP`` consecutive fully-blank rows and drops that
    trailing run (a styled cell near row 1,048,000 never pads the frame).

    Data cells are converted so ``_finish_frame`` (shared, unchanged, with ``load_csv``) sees
    exactly what a CSV would give it (``_xlsx_cell_to_value``): a ``datetime`` cell becomes a
    ``_PRIMARY_DT_FORMAT`` string; a ``datetime.time`` cell becomes an ``"HH:MM:SS"`` string; and
    -- the one XLSX-specific wrinkle -- a datetime column that is exactly midnight in every row
    AND has a separate companion time column (the same pairing ``suggest_channels``/
    ``_companion_time_col`` would find, checked here against the final, folded/deduped column
    names before any row is converted) is formatted date-only, matching a CSV's own split
    Date + Time columns and so exercising the exact same FIX-A join in ``_finish_frame``.
    Trailing fully-empty rows and columns are dropped (``_xlsx_trim_trailing_empty``); duplicate
    or blank header cells are made unique the same way ``pd.read_csv`` does
    (``_xlsx_dedupe_headers``).
    """
    wb = _xlsx_open(path)
    try:
        scans = []
        for ws in wb.worksheets:
            try:
                scan = _xlsx_scan_sheet(ws)
            except Exception:
                scan = None
            if scan is not None:
                scans.append((ws, scan))
        if not scans:
            raise ValueError(
                f"{path!r}: no worksheet looks like DFIT time-series data (no header row with "
                f"a date/time column and numeric data below it, in the first "
                f"{_XLSX_HEADER_SCAN_ROWS} rows of any sheet)"
            )

        # Ranking only matters with >1 qualifying sheet; a lone sheet skips the usability sample
        # and the row count entirely. Otherwise: usability first (a cheap 2000-row sample per
        # sheet), then the real row count -- a full extra pass over the sheet XML -- only for the
        # usable sheets, and only when more than one of them needs ordering.
        ranked = [[False, 0, ws, scan] for ws, scan in scans]
        if len(ranked) > 1:
            for r in ranked:
                r[0] = _xlsx_sheet_time_base_usable(r[2], r[3], path)
            to_count = [r for r in ranked if r[0]] if sum(r[0] for r in ranked) > 1 else (
                ranked if not any(r[0] for r in ranked) else [])
            if len(to_count) > 1:
                # Real row counts via the zip-level parser when worksheet parts line up
                # one-to-one with openpyxl's worksheets (workbook order); openpyxl's own walk
                # otherwise.
                zf = None
                parts: list = []
                try:
                    zf = zipfile.ZipFile(path)
                    parts = _xlsx_worksheet_parts(zf)
                except Exception:
                    parts = []
                part_by_title = (
                    {w.title: parts[k] for k, w in enumerate(wb.worksheets)}
                    if len(parts) == len(wb.worksheets) else {}
                )
                try:
                    for r in to_count:
                        ws, scan = r[2], r[3]
                        n_rows = None
                        if zf is not None and ws.title in part_by_title:
                            try:
                                n_rows = _xlsx_zip_count_data_rows(
                                    zf, part_by_title[ws.title], scan["data_start_row"])
                            except Exception:
                                n_rows = None
                        if n_rows is None:
                            n_rows = _xlsx_count_data_rows(ws, scan["data_start_row"])
                        r[1] = n_rows
                finally:
                    if zf is not None:
                        zf.close()
            ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)

        first_err: Optional[Exception] = None
        zero_span = False
        for _usable, _n, ws, scan in ranked:
            try:
                df, _cols = _xlsx_read_sheet_frame(ws, scan)
                td = _finish_frame(df, path)
            except Exception as e:
                if first_err is None:
                    first_err = e
                continue
            if _xlsx_span_s(td) > 0.0:
                stop_row = df.attrs.get("xlsx_gap_stop_row")
                if stop_row is not None:
                    w = _xlsx_gap_stop_warning(wb, path, ws, stop_row)
                    if w:
                        td.load_warnings.append(w)
                return td
            zero_span = True
        if zero_span:
            raise ValueError(f"{path!r}: no sheet with a usable time base")
        assert first_err is not None
        raise first_err
    finally:
        wb.close()


def load(path: str) -> TestData:
    """Dispatch to ``load_dbs``, ``load_csv``, or ``load_xlsx`` based on the file extension."""
    low = path.lower()
    if low.endswith(".dbs"):
        return load_dbs(path)
    if low.endswith(".xlsx"):
        return load_xlsx(path)
    return load_csv(path)
