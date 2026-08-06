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

# Excel's day-zero (the epoch that already accounts for the 1900 leap-year bug).
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_PRIMARY_DT_FORMAT = "%m/%d/%Y %H:%M:%S"
_PRIMARY_DT_FORMAT_DAYFIRST = "%d/%m/%Y %H:%M:%S"

# psi of hydrostatic head per (ppg * ft): the standard field-units constant.
PSI_PER_PPG_FT = 0.052

# Below this parsed-valid fraction, the datetime column is treated as unusable and load_csv looks
# for an elapsed-time column to fall back on instead (FIX B). Measured: Goodnight_DFIT_data.csv's
# Excel-mangled "Date/Time" column parses 0.40 valid (419,040 / 1,048,575 -- dateutil silently
# misreads "MM:SS.0" as a time-of-day on today's date), while a genuinely good column measures
# 1.00. 0.5 sits between the two with margin on both sides.
_MIN_VALID_DT_FRACTION = 0.5


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
    """
    m = s.str.extract(_LEADING_DM_RE)
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


def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a datetime column that may mix formatted strings and Excel serial numbers.

    Returns a tz-naive datetime64 Series. Any value that cannot be parsed becomes NaT.
    """
    s = series.astype("string").str.strip()
    dayfirst = _dayfirst_hint(s)
    fmt = _PRIMARY_DT_FORMAT_DAYFIRST if dayfirst else _PRIMARY_DT_FORMAT

    # Fast path: the primary formatted-string layout. Normalize to microsecond resolution so the
    # fallbacks below can be merged without lossy-cast errors (pandas 3.0 is unit-strict).
    dt = pd.to_datetime(s, format=fmt, errors="coerce").astype("datetime64[us]")

    # Fallback 1: bare Excel serial numbers (rounded to whole seconds; data is ~1 Hz).
    if dt.isna().any():
        as_num = pd.to_numeric(s, errors="coerce")
        secs = np.round(as_num.to_numpy(dtype=float) * 86400.0)
        excel = pd.Series(
            _EXCEL_EPOCH + pd.to_timedelta(secs, unit="s"), index=s.index
        ).astype("datetime64[us]")
        dt = dt.combine_first(excel)

    # Fallback 2: flexible parse for anything still missing (other string layouts).
    if dt.isna().any():
        generic = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst).astype("datetime64[us]")
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


def suggest_channels(columns: list[str]) -> dict[str, Optional[str]]:
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
    """
    lc = {c: c.lower() for c in columns}

    def find(*needles, avoid=()):
        for c in columns:
            name = lc[c]
            if any(n in name for n in needles) and not any(a in name for a in avoid):
                return c
        return None

    datetime_col = find("datetime", "date/time", "timestamp", "time", "date")
    # A rate/volume column can also contain "time"; don't misassign those as the datetime col.
    if datetime_col and any(x in lc[datetime_col] for x in ("rate", "vol", "press")):
        datetime_col = None

    time_col = None
    if datetime_col and "date" in lc[datetime_col] and "time" not in lc[datetime_col]:
        for c in columns:
            if c != datetime_col and _bare_name(c) == "time":
                time_col = c
                break

    pressure = find("press", "bhp", "whp", "psi") or None
    bhp_guess = find("bhp", "bottom")
    if bhp_guess:
        pressure = bhp_guess
    rate = find("rate", "bpm", "flow", "slurry")
    volume = find("vol", "bbl", avoid=("rate",))

    return {
        "datetime": datetime_col,
        "time": time_col,
        "pressure": pressure,
        "rate": rate,
        "volume": volume,
        "pressure_is_bhp": bool(bhp_guess),
    }


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

    @property
    def n(self) -> int:
        return len(self.df)

    def column(self, name: str) -> np.ndarray:
        return pd.to_numeric(self.df[name], errors="coerce").to_numpy(dtype=float)

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
            joined = parse_datetime(date_s + " " + time_s)
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
    # _MIN_VALID_DT_FRACTION) -- fall back to a clean elapsed-time column if the file has one.
    valid_frac = dt.notna().mean() if len(dt) else 0.0
    if valid_frac < _MIN_VALID_DT_FRACTION:
        elapsed = _find_elapsed_column(df)
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
            df[synth_col] = (
                pd.Timestamp("1970-01-01") + pd.to_timedelta(secs, unit="s")
            ).astype("datetime64[us]")
            return TestData(
                path=path, df=df, datetime_col=synth_col, t_s=secs, columns=list(df.columns)
            )

    if dt.notna().sum() == 0:
        raise ValueError(f"Could not parse any datetimes from column {dt_col!r}")
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

    t_s = rec["idx"].astype(np.float64) * float(interval_min) * 60.0

    dt_col = "DateTime"
    cols = {dt_col: (pd.Timestamp("1970-01-01") + pd.to_timedelta(t_s, unit="s")).astype("datetime64[us]")}
    for i, name in enumerate(names):
        cols[name] = rec[f"c{i}"].astype(np.float64)
    df = pd.DataFrame(cols)

    return TestData(path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=list(df.columns))


def load(path: str) -> TestData:
    """Dispatch to ``load_dbs`` or ``load_csv`` based on the file extension."""
    if path.lower().endswith(".dbs"):
        return load_dbs(path)
    return load_csv(path)
