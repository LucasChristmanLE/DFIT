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
converted via `_SG_TO_PPG`), pressure-gradient cells (psi/ft) are accepted and converted to ppg
via `_PSI_PER_PPG_FT` (`ppg = gradient / 0.052`), and kg/m3 cells are converted via
`units.KGM3_TO_PPG`, since downstream BHP conversion always expects ppg. A cell that names a fluid
but gives no number at all (e.g. a bare "Fresh Water") falls back to the fresh-water constant
`_FRESH_WATER_PPG`, flagged with a warning.

TVD is reported as `tvd_ft` (feet) even when the source cell is in meters, detected any of three
ways: an explicit meter suffix -- "mTVD"/"mKB"/"mMD"/"m"/"metres"/"meters" -- checked right after
the matched number (`_is_meters`); the label itself spelled "mTVD" (`_last_label_tvd_index`'s
`is_m_label`, e.g. "mTVD: 3368"); or a delimiter-bounded meter token sitting in the gap between
the label and the number (`_GAP_METER_TOKEN_RE`, e.g. "TVD (mKB): 3368"). The conversion
(`units.M_TO_FT`) runs BEFORE the `_TVD_MIN`/`_TVD_MAX` sanity filter, since a meters value can
otherwise look like a plausible-but-wrong feet value in the same numeric range.

A workbook with more than one sheet (one well per sheet, sheet title = well name) is disambiguated
by `well_hint` (see `_select_sheet`); a single-sheet workbook ignores it. A multi-sheet workbook
with no hint, or a hint that matches no sheet, reads the whole workbook rather than guessing
`sheets[0]` alone -- narrowing to one sheet only happens on a positive hint match.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from . import units

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
    r"|psi\s*/\s*ft|psi\s*/\s*foot|psi\s*per\s*ft|psi\s*per\s*foot"
    r"|kg\s*(?:/|per)\s*m\^?[3³]|kg\s*·\s*m-[3³])\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
# A cell that is *just* a footage or meterage number, optionally with a unit suffix -- e.g.
# "15887'", "3368 mTVD", "6399.88mKB" -- as opposed to a labeled one like "MD: 21833'" (which
# _NUMBER_RE/_extract_tvd_number handle separately). The unit group is captured (not just
# consumed) so callers can tell a meters cell from a feet one -- see _extract_bare_footage.
_BARE_FOOTAGE_RE = re.compile(
    r"\s*(\d[\d,]*(?:\.\d+)?)\s*(?P<unit>ft\.?|m(?:tvd|kb|md)?|met(?:re|er)s?)?'?\s*",
    re.IGNORECASE,
)
# Meter-unit token immediately following a matched number -- "mTVD"/"mKB"/"mMD", a bare "m", or
# "metre(s)"/"meter(s)" -- attached (no space) or spaced. Checked against the text right after
# the number's own match end (_is_meters), not the whole cell, so a stray "m" elsewhere in the
# text (e.g. inside a well name) can't false-positive.
_METER_SUFFIX_RE = re.compile(r"^\s*(m(?:tvd|kb|md)?|met(?:re|er)s?)\b", re.IGNORECASE)
# Same token family as _METER_SUFFIX_RE, but searched anywhere in a *gap* of text (e.g. between a
# "TVD" label and the number that follows it), delimiter-bounded on both sides so it only matches
# a standalone token like "(mKB)"/"in mKB ="/"(m)" and not a letter sequence that happens to
# contain "m" (e.g. "Formation") -- see _extract_tvd_number's label branch.
_GAP_METER_TOKEN_RE = re.compile(
    r"(?<![a-z0-9])(m(?:tvd|kb|md)?|met(?:re|er)s?)(?![a-z0-9])", re.IGNORECASE
)

_PPG_MIN, _PPG_MAX = 6.0, 22.0
_SG_MIN, _SG_MAX = 0.8, 2.6
_SG_TO_PPG = 8.345
_PSI_PER_PPG_FT = 0.052  # mirrors io_load.PSI_PER_PPG_FT; gradient(psi/ft) = 0.052 * ppg
_KGM3_MIN, _KGM3_MAX = 800.0, 2600.0  # mirrors the SG sanity range, in kg/m3 terms
_TVD_MIN, _TVD_MAX = 1000.0, 25000.0

_FRESH_WATER_PPG = 8.34
# Exact, trimmed, case-insensitive whole-cell fluid names that fall back to the fresh-water
# density constant above when no numeric density is found anywhere -- deliberately minimal, the
# only corpus-evidenced case (a Strathcona-style cell that just says "Fresh Water", no number).
_FRESH_WATER_NAMES = {"fresh water", "freshwater", "water"}


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


def _flatten(ws) -> list[str]:
    """All non-empty cells in one worksheet, in reading order (row-major, left to right)."""
    texts: list[str] = []
    for row in ws.iter_rows():
        for cell in row:
            text = _cell_text(cell.value)
            if text is not None:
                texts.append(text)
    return texts


def _flatten_workbook(wb) -> list[str]:
    """`_flatten` across every worksheet, in workbook order -- used whenever `_select_sheet`
    can't narrow to one sheet (no hint, or no match): reading only `sheets[0]` in that case would
    drop any answer that happens to live on a later sheet, a regression for the no-hint
    `scripts/` callers that predate multi-sheet workbooks entirely."""
    texts: list[str] = []
    for ws in wb.worksheets:
        texts.extend(_flatten(ws))
    return texts


_BOUND_START = r"(?<![a-z0-9])"
_BOUND_END = r"(?![a-z0-9])"


def _bounded_contains(container: str, contained: str) -> bool:
    """True if `contained` (already lowercased) appears in `container` (already lowercased) at a
    position bounded by a non-alphanumeric character (or a string edge) on both sides -- so "1h"
    does not match inside "21h", but it does match "smith 21h dfit" via the standalone "21h"
    token, and "#"/spaces count as bounds ("Sn#63473", "2022_08_09" surroundings)."""
    if not contained:
        return False
    pattern = _BOUND_START + re.escape(contained) + _BOUND_END
    return re.search(pattern, container) is not None


def _sheet_match_length(hint_lc: str, candidate: str | None) -> int:
    """Length of the matched (contained) string if `hint_lc` (already lowercased/stripped) and
    `candidate` are a delimiter-bounded substring match of each other in either direction, else 0.
    Bounded so a short hint like "1H" doesn't false-match inside a longer token like "21H" (see
    `_bounded_contains`). The length is used by `_select_sheet` to prefer the most specific match
    when more than one sheet matches."""
    if not candidate or not hint_lc:
        return 0
    c = candidate.strip().lower()
    if not c:
        return 0
    if _bounded_contains(c, hint_lc):
        return len(hint_lc)
    if _bounded_contains(hint_lc, c):
        return len(c)
    return 0


def _pick_best_sheet_match(
    matches: list[tuple],
) -> tuple:
    """Given a non-empty list of `(worksheet, match_length)` pairs, pick the one with the longest
    match length; if more than one sheet matched, append a warning naming the chosen sheet so an
    ambiguous narrowing is never silent. Returns `(worksheet, warnings)`."""
    best_ws, _ = max(matches, key=lambda pair: pair[1])
    warns = []
    if len(matches) > 1:
        warns.append(
            f"{len(matches)} sheets matched well hint; using the longest match, "
            f"sheet {best_ws.title!r}"
        )
    return best_ws, warns


def _select_sheet(wb, well_hint: str | None):
    """Pick which worksheet `parse_questionnaire` reads: the real Strathcona questionnaire
    carries three sheets, one well each, sheet title = well name -- other corpus files on hand
    are single-sheet.

    A single-sheet workbook is unambiguous and `well_hint` isn't even consulted (today's
    behavior, unchanged). With more than one sheet: try matching `well_hint` against each
    sheet's own title first, then (if nothing matched) against each sheet's own "Well Name"
    answer cell -- both via `_sheet_match_length` (delimiter-bounded containment, either
    direction). If more than one sheet matches at either stage, the one with the longest
    matched text wins (see `_pick_best_sheet_match`), and a warning names the chosen sheet.

    Returns `(worksheet, warnings)`, where `worksheet` is `None` to mean "no single sheet was
    identified -- flatten the whole workbook instead" (see `_flatten_workbook`). That's the
    case for a missing hint entirely (``scripts/`` callers that pass none) or a hint that
    matched nothing: narrowing to `sheets[0]` in either case would silently drop any answer
    that actually lives on a later sheet (a regression for a cover/instructions sheet placed
    first), so only a *positive* hint match narrows to a single worksheet.
    """
    sheets = wb.worksheets
    if len(sheets) == 1:
        return sheets[0], []
    if well_hint:
        hint_lc = well_hint.strip().lower()
        title_matches = [
            (ws, n) for ws in sheets if (n := _sheet_match_length(hint_lc, ws.title)) > 0
        ]
        if title_matches:
            return _pick_best_sheet_match(title_matches)
        name_matches = []
        for ws in sheets:
            name = _extract_text(_build_blocks(_flatten(ws)), _WELL_NAME)
            n = _sheet_match_length(hint_lc, name)
            if n > 0:
                name_matches.append((ws, n))
        if name_matches:
            return _pick_best_sheet_match(name_matches)
        return None, [
            f"{len(sheets)} sheets found; none matched well hint {well_hint!r} -- reading "
            "the whole workbook"
        ]
    return None, [
        f"{len(sheets)} sheets found and no well hint given -- reading the whole workbook"
    ]


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


def _is_kgm3_unit(unit: str) -> bool:
    return "kg" in unit.lower()


def _interpret_density(value: float, unit: str) -> tuple[float | None, list[str]]:
    """Apply the ppg/SG/gradient/kg-m3 interpretation rule to a raw (value, unit) match. See
    module docstring."""
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
    if _is_kgm3_unit(unit):
        if _KGM3_MIN <= value <= _KGM3_MAX:
            return value * units.KGM3_TO_PPG, warns
        warns.append(
            f"density {value} kg/m3 is outside the expected {_KGM3_MIN:.0f}-{_KGM3_MAX:.0f} "
            "kg/m3 range; ignored"
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
    # No numeric density anywhere -- fall back to an exact, trimmed, whole-cell fluid-name match
    # (e.g. a cell that just says "Fresh Water", no number at all). A cell that names the fluid
    # AND carries a real number (e.g. "Fresh Water - 8.4 lbs/gal") never reaches this fallback:
    # the loop above already returned on that cell's own numeric match, and even if it hadn't,
    # the exact-whole-cell check below wouldn't match a cell with extra text anyway.
    for key in (_WELLBORE_FLUID, _PUMPED_FLUID):
        for text in blocks.get(key, []):
            if text.strip().lower() in _FRESH_WATER_NAMES:
                warns.append(
                    f"fluid named {text.strip()!r} with no density given; assumed "
                    f"{_FRESH_WATER_PPG} ppg (fresh water)"
                )
                return _FRESH_WATER_PPG, text, warns
    warns.append("no parseable fluid density found in questionnaire")
    return None, None, warns


# --------------------------------------------------------------------------------------------------
# TVD
# --------------------------------------------------------------------------------------------------
def _is_meters(text: str, number_end: int) -> bool:
    """True if the text immediately after a matched number's end index (``number_end``) reads
    as a meters unit suffix -- "mTVD"/"mKB"/"mMD", a bare "m", or "metre(s)"/"meter(s)",
    attached (no space) or spaced. Checked right after the number, not the whole cell, so a
    stray "m" elsewhere in the text (e.g. inside a well name) can't false-positive."""
    return bool(_METER_SUFFIX_RE.match(text[number_end:]))


def _is_attached_meter_m(low: str, m_idx: int) -> bool:
    """True if the "m" at `low[m_idx]` is a unit-suffix "m" attached to a PRECEDING number --
    immediately preceded by a digit ("3368mTVD"), or by whitespace that is itself preceded by a
    digit ("3368 mTVD") -- as opposed to an "m" that starts the token with no number before it at
    all ("mTVD: 3368", where "mTVD" is the label's own spelling of the unit, not a suffix on some
    earlier number). Only the former is a genuine unit-suffix occurrence; see
    `_last_label_tvd_index`, which uses this to decide whether an "m"-preceded "tvd" is a suffix
    to skip or a label to keep."""
    i = m_idx - 1
    while i >= 0 and low[i].isspace():
        i -= 1
    return i >= 0 and low[i].isdigit()


def _last_label_tvd_index(low: str) -> tuple[int, bool] | None:
    """`(index, is_m_label)` for the last occurrence of "tvd" in `low` (already lowercased) that
    is a genuine label occurrence rather than the unit-suffix tail of an attached meter token
    like "mtvd". A combined cell like "TVD: 3368 mTVD" has two "tvd" occurrences: the label at
    index 0, and the one inside "mTVD" a few characters later, whose "m" is attached to the
    preceding number ("3368 m...") -- a genuine unit suffix, skipped. But a cell whose LABEL
    itself is spelled "mTVD" (e.g. "mTVD: 3368", no number before the "m" at all) must not be
    skipped just because it's "m"-preceded: there `is_m_label` comes back True, telling the
    caller the label names the unit as meters, since no unit suffix will follow the number in
    that shape either ("mTVD: 3368" has no trailing "mTVD" of its own).

    Returns None if every "tvd" occurrence in the text is a genuine unit-suffix tail (or there is
    none at all), letting the caller fall back to the whole-cell path instead."""
    idx: int | None = None
    is_m_label = False
    search_from = 0
    while True:
        found = low.find("tvd", search_from)
        if found == -1:
            break
        if found == 0 or low[found - 1] != "m":
            idx, is_m_label = found, False
        elif not _is_attached_meter_m(low, found - 1):
            idx, is_m_label = found, True
        search_from = found + 1
    return (idx, is_m_label) if idx is not None else None


def _extract_tvd_number(text: str) -> tuple[float | None, bool]:
    """Number for a cell containing "tvd" -- the first number *after* the last genuine label
    occurrence of "tvd" (see `_last_label_tvd_index`), so a combined "MD 21833' / TVD 10929.8'"
    cell yields 10929.8, not the MD. Falls back to the LAST number in the whole cell if no label
    occurrence exists (e.g. a bare "3368 mTVD" with no label prefix -- every "tvd" there is the
    unit-suffix tail) or if nothing follows the label (e.g. a bare "TVD:" with no value).

    The unit is meters if any of three independent signals says so: the text right after the
    matched number reads as a meters suffix (``_is_meters``, e.g. "TVD: 3368 mTVD"); the label
    itself was spelled "mTVD" (`is_m_label`, e.g. "mTVD: 3368"); or a delimiter-bounded meter
    token sits in the gap *between* the label and the number (``_GAP_METER_TOKEN_RE``, e.g.
    "TVD (mKB): 3368" or "TVD in mKB = 3368"). The fallback takes the LAST match, not the first,
    so a bare two-number cell with no usable label still prefers the later (TVD, by template
    order) number over the earlier (MD) one."""
    label = _last_label_tvd_index(text.lower())
    if label is not None:
        idx, is_m_label = label
        after_text = text[idx + len("tvd"):]
        m = _NUMBER_RE.search(after_text)
        if m is not None:
            gap = after_text[: m.start()]
            is_meters = (
                is_m_label
                or _is_meters(after_text, m.end())
                or bool(_GAP_METER_TOKEN_RE.search(gap))
            )
            return float(m.group(0).replace(",", "")), is_meters
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return None, False
    m = matches[-1]
    return float(m.group(0).replace(",", "")), _is_meters(text, m.end())


def _extract_bare_footage(text: str) -> tuple[float, bool] | None:
    """Parse a cell whose *entire* content is a footage or meterage number (e.g. "15887'",
    "3368 mTVD"), else None. Returns (value, is_meters)."""
    m = _BARE_FOOTAGE_RE.fullmatch(text)
    if not m:
        return None
    value = float(m.group(1).replace(",", ""))
    unit = m.group("unit")
    is_meters = unit is not None and not unit.lower().startswith("ft")
    return value, is_meters


def _meters_to_ft(value: float, is_meters: bool) -> float:
    """Pure meters->feet conversion (mirroring io_load.PSI_PER_PPG_FT's convention of keeping a
    matching constant local to each module); no warning side effect. Called BEFORE the
    _TVD_MIN/_TVD_MAX range filter in both TVD extraction paths below; that ordering is the
    actual fix (a meters TVD like 3368 sits inside the feet-assumed 1000-25000 range and was
    silently accepted as feet before this)."""
    return value * units.M_TO_FT if is_meters else value


def _warn_meters_conversion(value: float, ft: float, source: str, warns: list[str]) -> None:
    """Append the "converted to N ft" note -- factored out so the bare-candidates fallback (see
    `_tvd_from_block`) can defer it until after the range filter, and only for the candidate
    actually used; warning about a conversion the filter then discards would be misleading."""
    warns.append(f"TVD value {value} m (from {source!r}) converted to {ft:.1f} ft")


def _to_ft(value: float, is_meters: bool, source: str, warns: list[str]) -> float:
    """`_meters_to_ft` plus an immediate conversion warning -- used by the labeled-TVD-cell path
    in `_tvd_from_block`."""
    ft = _meters_to_ft(value, is_meters)
    if is_meters:
        _warn_meters_conversion(value, ft, source, warns)
    return ft


def _tvd_from_block(texts: list[str]) -> tuple[float | None, str | None, list[str]]:
    warns: list[str] = []
    for text in texts:
        if "tvd" in text.lower():
            value, is_meters = _extract_tvd_number(text)
            if value is None:
                warns.append(f"TVD label found but no number in {text!r}")
                continue
            value = _to_ft(value, is_meters, text, warns)
            if not (_TVD_MIN <= value <= _TVD_MAX):
                warns.append(
                    f"TVD value {value} outside expected {_TVD_MIN:.0f}-{_TVD_MAX:.0f} ft range"
                )
                continue
            return value, text, warns

    # No labeled TVD cell in this block -- fall back to bare footage/meterage numbers, template
    # order MD then TVD (this is the Abraxas-style layout: two unlabeled rows under one question).
    # The conversion warning is deferred until after the range filter and emitted only for the
    # candidate actually returned as the TVD -- warning about every surviving candidate's
    # conversion (including the MD's) would misleadingly label the MD's conversion as a TVD one.
    bare = []
    for text in texts:
        parsed = _extract_bare_footage(text)
        if parsed is None:
            continue
        value, is_meters = parsed
        ft = _meters_to_ft(value, is_meters)
        if not (_TVD_MIN <= ft <= _TVD_MAX):
            continue
        bare.append((text, value, ft, is_meters))
    if len(bare) == 2:
        (md_text, _md_value, md_val, _md_is_meters) = bare[0]
        (tvd_text, tvd_value, tvd_val, tvd_is_meters) = bare[1]
        if tvd_is_meters:
            _warn_meters_conversion(tvd_value, tvd_val, tvd_text, warns)
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
def parse_questionnaire(xlsx_path: str, well_hint: str | None = None) -> QuestionnaireResult:
    """Parse `xlsx_path` for fluid density and TVD.

    `well_hint`, if given, picks which worksheet to read when the workbook has more than one
    sheet (see `_select_sheet`) -- the real Strathcona questionnaire carries three sheets, one
    well each. Ignored (and harmless) for the single-sheet layout every other corpus file on
    hand uses; `scripts/` callers that pass none get the whole-workbook flatten instead of
    guessing `sheets[0]` (see `_select_sheet`'s docstring).

    Per-field parsing never raises -- a malformed cell just adds a warning and leaves that field
    None. Only an unreadable workbook (e.g. a truncated/corrupt zip) raises; the caller is expected
    to swallow that too, since a bad questionnaire must never block CSV loading.
    """
    import openpyxl  # lazy: openpyxl costs ~0.4s to import, and store -> questionnaire is on the
    # app-startup path even when no questionnaire is ever opened.

    result = QuestionnaireResult(path=str(xlsx_path))
    wb = openpyxl.load_workbook(xlsx_path, data_only=True, read_only=True)
    try:
        ws, sheet_warns = _select_sheet(wb, well_hint)
        result.warnings.extend(sheet_warns)
        texts = _flatten(ws) if ws is not None else _flatten_workbook(wb)
        blocks = _build_blocks(texts)
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
