"""Unit-conversion constants and header/alias lookup tables for pressure, rate, and volume.

Leaf module: no other ``dfit_tool`` imports, sits below ``io_load`` and ``questionnaire`` in the
import graph (see ../CLAUDE.md's Architecture section), so both can use it without inverting any
dependency.

Everything downstream of the IO boundary (fixed psi offsets, plot labels, panel rows,
``dfit_log.csv`` columns) stays field-unit -- psi / bpm / bbl. Every factor here means "multiply
the raw metric value by this to get the field-unit equivalent".
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------------------
# conversion constants
# --------------------------------------------------------------------------------------------------
KPA_TO_PSI = 0.1450377377
MPA_TO_PSI = 145.0377377
BAR_TO_PSI = 14.50377
M3_TO_BBL = 6.2898107704
M_TO_FT = 3.280839895
KGM3_TO_PPG = 1.0 / 119.8264

# --------------------------------------------------------------------------------------------------
# factor tables, keyed by canonical unit
# --------------------------------------------------------------------------------------------------
PRESSURE_FACTORS = {"psi": 1.0, "kpa": KPA_TO_PSI, "mpa": MPA_TO_PSI, "bar": BAR_TO_PSI}
RATE_FACTORS = {"bpm": 1.0, "m3/min": M3_TO_BBL}
VOLUME_FACTORS = {"bbl": 1.0, "m3": M3_TO_BBL}


# --------------------------------------------------------------------------------------------------
# token normalization
# --------------------------------------------------------------------------------------------------
def _normalize_token(raw: str) -> str:
    """Lowercase, strip, and fold spelling variants down to one comparable key: ``"³"``/``"^3"``
    both fold to ``"3"``, and internal whitespace plus the middle-dot (``"·"``) are dropped -- so
    ``"m³/min"``, ``"m^3/min"``, ``"M3 / MIN"``, and ``"kPa g"`` (a header suffix split by a
    stray space) all reduce to the same string before the alias lookup below."""
    s = raw.strip().lower()
    s = s.replace("³", "3").replace("^3", "3")
    s = s.replace("·", "").replace(" ", "")
    return s


# --------------------------------------------------------------------------------------------------
# alias tables: normalized token -> canonical key in the factor tables above
# --------------------------------------------------------------------------------------------------
_PRESSURE_ALIASES = {
    "psi": "psi", "psia": "psi", "psig": "psi",
    "kpa": "kpa", "kpag": "kpa", "kpaa": "kpa",  # "kpag" covers header suffixes like "KPAg"
    "mpa": "mpa", "mpag": "mpa", "mpaa": "mpa",
    "bar": "bar", "barg": "bar", "bara": "bar",
}
_RATE_ALIASES = {
    "bpm": "bpm", "bbl/min": "bpm", "bbls/min": "bpm", "bbl/m": "bpm", "bbls/m": "bpm",
    "m3/min": "m3/min", "m3min": "m3/min", "m3/m": "m3/min",
}
_VOLUME_ALIASES = {
    "bbl": "bbl", "bbls": "bbl",
    "m3": "m3",
}


def lookup_pressure(token: str) -> tuple[str, float] | None:
    """``(canonical unit, factor)`` for a pressure unit token (a header suffix or a UI
    override), or ``None`` if the token isn't recognized."""
    canon = _PRESSURE_ALIASES.get(_normalize_token(token))
    return (canon, PRESSURE_FACTORS[canon]) if canon is not None else None


def lookup_rate(token: str) -> tuple[str, float] | None:
    """Same as ``lookup_pressure`` for a rate unit token."""
    canon = _RATE_ALIASES.get(_normalize_token(token))
    return (canon, RATE_FACTORS[canon]) if canon is not None else None


def lookup_volume(token: str) -> tuple[str, float] | None:
    """Same as ``lookup_pressure`` for a volume unit token."""
    canon = _VOLUME_ALIASES.get(_normalize_token(token))
    return (canon, VOLUME_FACTORS[canon]) if canon is not None else None
