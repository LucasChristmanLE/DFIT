"""Parse a "DFIT Questionnaire" xlsx for fluid density and true vertical depth.

Pure Python -- no tkinter here. `ui.py` calls `find_questionnaire`/`parse_questionnaire` from
`_load` and wraps the whole thing in try/except so a missing or malformed questionnaire never
blocks CSV loading.

The template is a fixed set of label rows down column A (see `_KNOWN_LABELS`); the answer to each
question is whatever non-empty cells follow it, up to the next known label. Both real-world variants
on hand share this layout but disagree on where the numbers land -- density is sometimes on the
"fluid in the wellbore" line, sometimes only on the "fluid pumped" line; TVD is sometimes a labeled
"TVD: ..." cell, sometimes the second of two bare footage numbers -- so each field is looked up in a
priority order of answer blocks, falling through to the next block when a block yields nothing.

Density is reported as `density_ppg` (ppg): ppg and specific-gravity cells are read directly (SG
converted via `_SG_TO_PPG`), and pressure-gradient cells (psi/ft) are also accepted and converted
to ppg via `_PSI_PER_PPG_FT` (`ppg = gradient / 0.052`), since downstream BHP conversion always
expects ppg.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import openpyxl

# --------------------------------------------------------------------------------------------------
# template labels
# --------------------------------------------------------------------------------------------------
# Case-insensitive *prefixes*: real files add trailing colons, extra whitespace, or trailing text
# ("Formation Hydrocarbon GOR:", "Planned Perforations (MD and TVD): Toesleeve Conversions", "Well
# Name: " with a trailing space), so a cell counts as a label if its stripped text starts with one
# of these. When more than one prefix matches (e.g. "Formation" and "Formation Hydrocarbon" both
# match "Formation Hydrocarbon GOR:"), the longest -- i.e. most specific -- one wins.
_KNOWN_LABELS = (
    "Well Name",
    "Formation",
    "Date Pumped",
    "Monitor Time for Gauges",
    "Surface or downhole gauges",
    "Reservoir Net Height",
    "Reservoir Gross Height",
    "Water Saturation",
    "Porosity",
    "Young's Modulus",
    "Poison's Ratio",
    "Bottomhole Temperature",
    "Formation Hydrocarbon",
    "API Gravity",
    "Type and density of fluid in the wellbore",
    "Planned Perforations",
    "Section to be completed",
    "Was the well loaded",
    "What volume was used to load",
    "Volume of fluid pumped after formation break",
    "What type of fluid was pumped",
    "Does the Acid Pump have a densometer",
    "If there is a densometer",
    "What is the plug depth",
    "Actual perforation depth",
)

# Canonical (lowercased) keys used to look up answer blocks below.
_WELLBORE_FLUID = "type and density of fluid in the wellbore"
_PUMPED_FLUID = "what type of fluid was pumped"
_ACTUAL_PERFS = "actual perforation depth"
_PLANNED_PERFS = "planned perforations"
_WELL_NAME = "well name"
_FORMATION = "formation"

_DENSITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(ppg|lbs?\s*/\s*gal|lbs?\s*per\s*gal|#\s*/\s*gal|specific\s*gravity|sg"
    r"|psi\s*/\s*ft|psi\s*/\s*foot|psi\s*per\s*ft|psi\s*per\s*foot)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[\d,]+(?:\.\d+)?")
# A cell that is *just* a footage number, optionally with "ft"/"'" -- e.g. "15887'" -- as opposed to
# a labeled one like "MD: 21833'" (which _NUMBER_RE handles separately).
_BARE_FOOTAGE_RE = re.compile(r"\s*([\d,]+(?:\.\d+)?)\s*(?:ft\.?)?'?\s*", re.IGNORECASE)

_PPG_MIN, _PPG_MAX = 6.0, 22.0
_SG_MIN, _SG_MAX = 0.8, 2.6
_SG_TO_PPG = 8.345
_PSI_PER_PPG_FT = 0.052  # mirrors io_load.PSI_PER_PPG_FT; gradient(psi/ft) = 0.052 * ppg
_TVD_MIN, _TVD_MAX = 1000.0, 25000.0


@dataclass
class QuestionnaireResult:
    path: str
    density_ppg: float | None = None
    density_source: str | None = None  # raw cell text the number came from
    tvd_ft: float | None = None
    tvd_source: str | None = None
    well_name: str | None = None
    formation: str | None = None
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------------------
# file discovery
# --------------------------------------------------------------------------------------------------
def is_questionnaire_filename(name: str) -> bool:
    """True if `name` is a candidate questionnaire filename: ``.xlsx`` (case-insensitive),
    contains "questionnaire" (case-insensitive), and isn't an Excel lock file (``~$...``).
    Factored out of `find_questionnaire` so other callers (`scripts/triage/features.py`'s
    well-root grouping) apply the exact same predicate rather than a hand-duplicated copy."""
    low = name.lower()
    return low.endswith(".xlsx") and "questionnaire" in low and not name.startswith("~$")


def find_questionnaire(data_path: str) -> tuple[str | None, list[str]]:
    """Look for a ``*questionnaire*.xlsx`` next to `data_path`, then in its parent directory.

    Excel lock files (``~$...``) are skipped. If more than one candidate turns up in the winning
    directory, the first one alphabetically is used and a warning describing the ambiguity is
    returned alongside the path (there's no GUI-visible place for a `warnings.warn` to land).
    """
    start = os.path.dirname(os.path.abspath(data_path))
    for directory in (start, os.path.dirname(start)):
        if not directory or not os.path.isdir(directory):
            continue
        candidates = sorted(
            name for name in os.listdir(directory) if is_questionnaire_filename(name)
        )
        if candidates:
            warns = []
            if len(candidates) > 1:
                warns.append(
                    f"multiple questionnaire files found in {directory!r}; using {candidates[0]!r}"
                )
            return os.path.join(directory, candidates[0]), warns
    return None, []


# --------------------------------------------------------------------------------------------------
# workbook flattening
# --------------------------------------------------------------------------------------------------
def _cell_text(value) -> str | None:
    """Str-ify a cell value (which may be a str, number, datetime, ...); blanks become None."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    return str(value)


def _flatten(wb) -> list[str]:
    """All non-empty cells across every sheet, in reading order (row-major, left to right)."""
    texts: list[str] = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                text = _cell_text(cell.value)
                if text is not None:
                    texts.append(text)
    return texts


def _label_key(text: str) -> str | None:
    """The longest known label whose prefix matches `text` (case-insensitive), or None."""
    stripped = text.strip().lower()
    best: str | None = None
    for label in _KNOWN_LABELS:
        low = label.lower()
        if stripped.startswith(low) and (best is None or len(low) > len(best)):
            best = low
    return best


def _build_blocks(texts: list[str]) -> dict[str, list[str]]:
    """Group cells into answer blocks keyed by the known label that precedes them."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for text in texts:
        label = _label_key(text)
        if label is not None:
            current = label
            blocks.setdefault(current, [])
            continue
        if current is not None:
            blocks[current].append(text)
    return blocks


# --------------------------------------------------------------------------------------------------
# density
# --------------------------------------------------------------------------------------------------
def _is_sg_unit(unit: str) -> bool:
    u = unit.lower()
    return u == "sg" or "specific" in u


def _is_gradient_unit(unit: str) -> bool:
    return "psi" in unit.lower()


def _interpret_density(value: float, unit: str) -> tuple[float | None, list[str]]:
    """Apply the ppg/SG/gradient interpretation rule to a raw (value, unit) match. See module
    docstring."""
    warns: list[str] = []
    if _is_gradient_unit(unit):
        ppg = value / _PSI_PER_PPG_FT
        if _PPG_MIN <= ppg <= _PPG_MAX:
            return ppg, warns
        warns.append(
            f"density gradient {value} psi/ft converts to {ppg:.2f} ppg, outside the expected "
            "ppg range; ignored"
        )
        return None, warns
    sg_unit = _is_sg_unit(unit)
    if _PPG_MIN <= value <= _PPG_MAX:
        if sg_unit:
            warns.append(
                f"density {value} labeled '{unit}' but within the typical ppg range; used as ppg"
            )
        return value, warns
    if sg_unit and _SG_MIN <= value <= _SG_MAX:
        return value * _SG_TO_PPG, warns
    warns.append(f"density value {value} ({unit!r}) is outside the expected ppg/SG ranges; ignored")
    return None, warns


def _extract_density(blocks: dict[str, list[str]]) -> tuple[float | None, str | None, list[str]]:
    warns: list[str] = []
    for key in (_WELLBORE_FLUID, _PUMPED_FLUID):
        for text in blocks.get(key, []):
            m = _DENSITY_RE.search(text)
            if not m:
                continue
            value_ppg, sub_warns = _interpret_density(float(m.group(1)), m.group(2))
            warns.extend(sub_warns)
            if value_ppg is not None:
                return value_ppg, text, warns
    warns.append("no parseable fluid density found in questionnaire")
    return None, None, warns


# --------------------------------------------------------------------------------------------------
# TVD
# --------------------------------------------------------------------------------------------------
def _extract_number(text: str) -> float | None:
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    return float(m.group(0).replace(",", ""))


def _extract_tvd_number(text: str) -> float | None:
    """Number for a cell containing "tvd" -- the first number *after* the (last) "tvd", so a
    combined "MD 21833' / TVD 10929.8'" cell yields 10929.8, not the MD. Falls back to the first
    number in the whole cell if nothing follows "tvd" (e.g. a bare "TVD:" label with no value)."""
    idx = text.lower().rindex("tvd")
    after = _extract_number(text[idx + len("tvd"):])
    return after if after is not None else _extract_number(text)


def _extract_bare_footage(text: str) -> float | None:
    """Parse a cell whose *entire* content is a footage number (e.g. "15887'"), else None."""
    m = _BARE_FOOTAGE_RE.fullmatch(text)
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def _tvd_from_block(texts: list[str]) -> tuple[float | None, str | None, list[str]]:
    warns: list[str] = []
    for text in texts:
        if "tvd" in text.lower():
            value = _extract_tvd_number(text)
            if value is None:
                warns.append(f"TVD label found but no number in {text!r}")
                continue
            if not (_TVD_MIN <= value <= _TVD_MAX):
                warns.append(
                    f"TVD value {value} outside expected {_TVD_MIN:.0f}-{_TVD_MAX:.0f} ft range"
                )
                continue
            return value, text, warns

    # No labeled TVD cell in this block -- fall back to bare footage numbers, template order MD
    # then TVD (this is the Abraxas-style layout: two unlabeled rows under one question).
    bare = [(text, _extract_bare_footage(text)) for text in texts]
    bare = [(text, v) for text, v in bare if v is not None and _TVD_MIN <= v <= _TVD_MAX]
    if len(bare) == 2:
        (md_text, md_val), (tvd_text, tvd_val) = bare
        if tvd_val > md_val:
            warns.append(f"TVD {tvd_val} exceeds MD {md_val} ({md_text!r}); using it anyway")
        return tvd_val, tvd_text, warns
    if len(bare) == 1:
        warns.append(
            f"only one depth value found ({bare[0][0]!r}); can't distinguish MD from TVD"
        )
    return None, None, warns


def _extract_tvd(blocks: dict[str, list[str]]) -> tuple[float | None, str | None, list[str]]:
    warns: list[str] = []
    for key in (_ACTUAL_PERFS, _PLANNED_PERFS):
        value, source, sub_warns = _tvd_from_block(blocks.get(key, []))
        warns.extend(sub_warns)
        if value is not None:
            return value, source, warns
    warns.append("no parseable TVD found in questionnaire")
    return None, None, warns


# --------------------------------------------------------------------------------------------------
# free-text fields (well name, formation)
# --------------------------------------------------------------------------------------------------
def _extract_text(blocks: dict[str, list[str]], key: str) -> str | None:
    """The first non-empty cell of `blocks[key]`, stripped, or None if the block is absent/empty.

    First cell only, not a join of the whole block -- the template is one answer cell per label,
    and joining risks pulling in spillover from whatever follows.
    """
    for text in blocks.get(key, []):
        stripped = text.strip()
        if stripped:
            return stripped
    return None


# --------------------------------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------------------------------
def parse_questionnaire(xlsx_path: str) -> QuestionnaireResult:
    """Parse `xlsx_path` for fluid density and TVD.

    Per-field parsing never raises -- a malformed cell just adds a warning and leaves that field
    None. Only an unreadable workbook (e.g. a truncated/corrupt zip) raises; the caller is expected
    to swallow that too, since a bad questionnaire must never block CSV loading.
    """
    result = QuestionnaireResult(path=str(xlsx_path))
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        blocks = _build_blocks(_flatten(wb))
    finally:
        wb.close()

    try:
        result.density_ppg, result.density_source, warns = _extract_density(blocks)
        result.warnings.extend(warns)
    except Exception as e:  # malformed cell content must never break the whole parse
        result.warnings.append(f"density parsing failed: {e}")

    try:
        result.tvd_ft, result.tvd_source, warns = _extract_tvd(blocks)
        result.warnings.extend(warns)
    except Exception as e:
        result.warnings.append(f"TVD parsing failed: {e}")

    try:
        result.well_name = _extract_text(blocks, _WELL_NAME)
    except Exception as e:
        result.warnings.append(f"well name parsing failed: {e}")

    try:
        result.formation = _extract_text(blocks, _FORMATION)
    except Exception as e:
        result.warnings.append(f"formation parsing failed: {e}")

    return result
