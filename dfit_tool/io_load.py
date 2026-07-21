"""CSV loading, datetime parsing, channel mapping, and surface->BHP conversion.

The loader is deliberately format-tolerant: DFIT exports carry a datetime column that is usually
``MM/DD/YYYY HH:MM:SS`` but occasionally leaks raw Excel serial numbers (e.g. ``43508.34097``) in
the long falloff tail. Both are parsed onto one elapsed-seconds time base.

Only CSV is handled in this build; a Fracpro ``.DBS`` reader is a later addition (see ../plan.md).
"""

from __future__ import annotations

import re
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
