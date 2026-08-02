"""CSV loading, datetime parsing, channel mapping, and surface->BHP conversion.

The loader is deliberately format-tolerant: DFIT exports carry a datetime column that is usually
``MM/DD/YYYY HH:MM:SS`` but occasionally leaks raw Excel serial numbers (e.g. ``43508.34097``) in
the long falloff tail. Both are parsed onto one elapsed-seconds time base.

Two formats are handled: CSV (``load_csv``) and Fracpro's binary ``.DBS`` format (``load_dbs``).
``load`` dispatches on the file extension.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Excel's day-zero (the epoch that already accounts for the 1900 leap-year bug).
_EXCEL_EPOCH = pd.Timestamp("1899-12-30")
_PRIMARY_DT_FORMAT = "%m/%d/%Y %H:%M:%S"

# psi of hydrostatic head per (ppg * ft): the standard field-units constant.
PSI_PER_PPG_FT = 0.052


# --------------------------------------------------------------------------------------------------
# datetime parsing
# --------------------------------------------------------------------------------------------------
def parse_datetime(series: pd.Series) -> pd.Series:
    """Parse a datetime column that may mix formatted strings and Excel serial numbers.

    Returns a tz-naive datetime64 Series. Any value that cannot be parsed becomes NaT.
    """
    s = series.astype("string").str.strip()

    # Fast path: the primary formatted-string layout. Normalize to microsecond resolution so the
    # fallbacks below can be merged without lossy-cast errors (pandas 3.0 is unit-strict).
    dt = pd.to_datetime(s, format=_PRIMARY_DT_FORMAT, errors="coerce").astype("datetime64[us]")

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
        generic = pd.to_datetime(s, errors="coerce").astype("datetime64[us]")
        dt = dt.combine_first(generic)

    return dt


def elapsed_seconds(dt: pd.Series) -> np.ndarray:
    """Seconds elapsed from the first valid timestamp."""
    t0 = dt.dropna().iloc[0]
    return (dt - t0).dt.total_seconds().to_numpy(dtype=float)


# --------------------------------------------------------------------------------------------------
# channel detection
# --------------------------------------------------------------------------------------------------
_UNIT_RE = re.compile(r"\(([^)]*)\)")


def _unit_of(colname: str) -> Optional[str]:
    m = _UNIT_RE.search(colname)
    return m.group(1).strip().lower() if m else None


def suggest_channels(columns: list[str]) -> dict[str, Optional[str]]:
    """Best-guess mapping of column names to roles.

    Returns keys: ``pressure``, ``rate``, ``volume``, ``pressure_is_bhp`` (bool guess),
    and ``datetime``. Values are column names (or None). The UI presents these as defaults.
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

    pressure = find("press", "bhp", "whp", "psi") or None
    bhp_guess = find("bhp", "bottom")
    if bhp_guess:
        pressure = bhp_guess
    rate = find("rate", "bpm", "flow", "slurry")
    volume = find("vol", "bbl", avoid=("rate",))

    return {
        "datetime": datetime_col,
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


def load_csv(path: str) -> TestData:
    """Load a DFIT CSV and attach an elapsed-seconds time base."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    columns = list(df.columns)

    guess = suggest_channels(columns)
    dt_col = guess["datetime"] or columns[0]

    dt = parse_datetime(df[dt_col])
    if dt.notna().sum() == 0:
        raise ValueError(f"Could not parse any datetimes from column {dt_col!r}")
    df[dt_col] = dt
    t_s = elapsed_seconds(dt)

    return TestData(path=path, df=df, datetime_col=dt_col, t_s=t_s, columns=columns)


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
