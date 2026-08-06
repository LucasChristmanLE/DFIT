"""One-off batch export: a well-location Excel sheet for every DFIT test in a folder-mode root.

The DFIT app only ever surfaces a well name (from the questionnaire) one test at a time, in the
interactive UI. This script walks the same test queue the app builds (`store.scan_root`, so the
row set matches the app's queue exactly -- same `test_id`s, same arbitrary-depth walk (`scan_root`
uses `os.walk`), same dedup rules), reads each test's questionnaire well name, resolves it against
two Colorado COGCC shapefile attribute tables (`Wells.dbf` and
`Directional_Bottomhole_Locations.dbf`), and writes one row per test to an .xlsx sheet with
surface/bottomhole lat-long, operator, API label, and a record of how (or whether) the match was
made.

`MatchSource` is one of `directional-exact`/`wells-exact` (normalized-name match against the
respective table), `directional-fuzzy`/`wells-fuzzy` (a close but not exact name match, see
`_fuzzy_resolve`), `directional-sibling`/`wells-sibling` (a fuzzy match whose best candidate is a
*different* well on the same pad -- a digit got substituted, not just inserted or deleted; see
`_digits_compatible`), `name-too-generic` (the normalized name has no run of >= 3 letters, so it
is a bare well designator rather than a lease-name token and cannot be matched safely; see
`resolve`), `no-questionnaire` (no questionnaire file for the test) / `questionnaire-error` (a
questionnaire file exists but `parse_questionnaire` raised -- kept distinct from
`no-questionnaire` so a bad file doesn't silently read the same as no file), `no-well-name`, or
`no-match`. A `-sibling` row reports surface lat/long (wells on a pad share a surface location)
but deliberately leaves `BtmhLat`/`BtmhLong` blank -- laterals from a shared pad go different
directions, so a sibling's bottomhole would be meaningless.

Why the DBF reader is inlined: the project venv has no `dbfread`/`pyshp`/`geopandas`, and
`requirements.txt` is pinned deliberately (see CLAUDE.md) -- this script does not get to add a
dependency. Only the `.dbf` attribute table is needed; the paired `.shp` geometry file is
irrelevant here because both source tables already carry lat/long as ordinary attribute columns
(`Wells.Latitude`/`Longitude`, `Directional...Lat`/`Long`). dBase III's `.dbf` layout is simple
enough (fixed header, fixed-width field descriptors, fixed-width records) that decoding it with
`struct` is a couple dozen lines, well short of a real dependency.

Run from the repo root with the project venv:

    C:\\Users\\LucasChristman\\.venvs\\dfit\\Scripts\\python.exe scripts\\well_locations.py

Defaults assume `C:\\DFIT Data` as the root, `<repo>/Well Locations` for the two .dbf files, and
write `<repo>/well_locations.xlsx`. All three are overridable; see `--help`.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator

import openpyxl
import pandas as pd

# `scripts/` sits above the `dfit_tool` package, not inside it, so the repo root has to be on
# sys.path before `from dfit_tool import ...` works when this file is run directly
# (`python scripts\well_locations.py`) rather than as an installed package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dfit_tool import questionnaire, store  # noqa: E402  (import after sys.path fixup)

DEFAULT_ROOT = r"C:\DFIT Data"
DEFAULT_DB_DIR = os.path.join(_REPO_ROOT, "Well Locations")
DEFAULT_OUT = os.path.join(_REPO_ROOT, "well_locations.xlsx")

COLUMNS = [
    "TestID", "WellName", "DBWellName", "API_Label", "Operator",
    "SurfLat", "SurfLong", "BtmhLat", "BtmhLong", "MatchSource", "MatchScore",
]

_SUFFIX_RE = re.compile(r" \(\d+ candidates\)$")

# A real well name normalizes to a lease-name token plus a designator (`FAIRVIEW2`, `COX5`,
# `CHU194`). A key with no run of >= 3 letters is a bare designator (`1BH`, `3CH`, `229HN`,
# `5C30M`) and cannot be matched safely: `wells_by_name` manufactures hundreds of keys this short
# or shorter from its `Well_Num + Well_Name` permutations, so such a key finds a confident-looking
# exact hit on an arbitrary well (real case: questionnaire `1BH` exact-matched a `Well_Num="1"` +
# `Well_Name="B & H"` record on an unrelated well ~50 miles away). `resolve` rejects these before
# any index lookup, exact or fuzzy.
_MIN_KEY_LEN = 4
_LETTER_RUN_RE = re.compile(r"[A-Z]{3}")


# --------------------------------------------------------------------------------------------------
# minimal dBase III (.dbf) attribute-table reader
# --------------------------------------------------------------------------------------------------
def _read_dbf(path: str, fields: set[str] | None = None) -> Iterator[dict[str, str]]:
    """Yield one dict per non-deleted record of a dBase III `.dbf` attribute table.

    Header layout: 32 bytes, with record count / header length / record length packed at offset 4
    as `<I H H`. Field descriptors follow, 32 bytes each, until a descriptor whose first byte is
    the 0x0D terminator (or the descriptor read comes back short/empty). Each descriptor's name is
    the first 11 bytes up to the first NUL, and its length is the byte at offset 16.

    Records start at `hlen` and are `rlen` bytes each; the first byte of a record is a delete flag
    (`"*"` for deleted, otherwise a space) and is not part of any field, so field offsets start at
    1. When `fields` is given, only those keys are included in the yielded dict, but every field is
    still stepped over so later fields' offsets stay correct.

    A short or empty read (EOF before `nrec` records, or a truncated final record) stops the loop
    rather than yielding a partial row or spinning the remaining iterations at EOF; a truncated
    record also prints a warning naming the file and record index, since that is a real data
    problem worth surfacing, not a normal EOF.
    """
    with open(path, "rb") as f:
        header = f.read(32)
        nrec, hlen, rlen = struct.unpack("<I H H", header[4:12])

        field_descs: list[tuple[str, int]] = []
        while True:
            desc = f.read(32)
            if not desc or desc[0] == 0x0D:
                break
            name = desc[:11].split(b"\x00")[0].decode("latin-1")
            length = desc[16]
            field_descs.append((name, length))

        f.seek(hlen)
        for i in range(nrec):
            record = f.read(rlen)
            if not record:
                break  # EOF: nrec overstates the file (e.g. a truncated download), not more data
            if len(record) < rlen:
                # A truncated final record: yielding it would produce a row with blank tail
                # fields, and a real error deserves a real warning rather than a silently
                # short row -- and there is no more data behind it, so stop instead of spinning
                # the remaining iterations at EOF.
                print(f"warning: {path!r} truncated at record {i} (short read)", file=sys.stderr)
                break
            if record[0:1] == b"*":
                continue
            row: dict[str, str] = {}
            offset = 1
            for name, length in field_descs:
                raw = record[offset:offset + length]
                offset += length
                if fields is None or name in fields:
                    row[name] = raw.decode("latin-1").strip()
            yield row


# --------------------------------------------------------------------------------------------------
# small parsing helpers
# --------------------------------------------------------------------------------------------------
def _norm(s: str | None) -> str:
    """The match key used everywhere: upper-case, punctuation/whitespace stripped."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def _num(s: str | None) -> float | None:
    """Parse a DBF numeric-as-string field to a float, or None if blank/unparseable."""
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _digits_compatible(a: str, b: str) -> bool:
    """True unless the two keys' digit sequences differ by a *substitution*.

    Digit insertions and deletions are benign: they are how one well gets written with or without
    leading zeros or a pad prefix (`08-59` vs `8-59`, `162HNX` vs `08-162HNX`). A replaced digit is
    not -- `BERRY IC 15-159HC` vs `BERRY IC 11-159HC` is a sibling well on the same pad, and it
    scores 0.929, above the fuzzy threshold, with no other candidate close enough for the margin
    guard to fire. So diff the two digit strings and report any `replace` op as incompatible.
    """
    da = "".join(c for c in a if c.isdigit())
    db = "".join(c for c in b if c.isdigit())
    ops = difflib.SequenceMatcher(None, da, db).get_opcodes()
    return not any(tag == "replace" for tag, *_ in ops)


def _api_key(api_label: str | None) -> str | None:
    """`Directional...API_Label` (dashed, with sidetrack suffix) -> `Wells.API` (8-char
    county+seq), by joining the county and sequence segments. Resolves for 38,992/38,992
    directional rows against `Wells.API`, verified against the real tables."""
    parts = (api_label or "").split("-")
    if len(parts) >= 3:
        return parts[1] + parts[2]
    return None


# --------------------------------------------------------------------------------------------------
# well index
# --------------------------------------------------------------------------------------------------
@dataclass
class WellIndex:
    """Both source tables, indexed for lookup by `resolve`. Built directly from dicts in tests
    (see tests/test_well_locations.py) so the 94 MB real .dbf files never need to be touched to
    exercise the matching logic; `load_index` is the file-reading wrapper used by `main`."""

    directional: dict[str, list[dict]]
    wells_by_api: dict[str, dict]
    wells_by_name: dict[str, list[dict]]
    dir_by_api: dict[str, list[dict]]
    _cache: dict[str, "Match"] = field(default_factory=dict)
    _all_keys: list[str] | None = None

    def all_keys(self) -> list[str]:
        """Every name key in either table, the candidate pool for fuzzy matching. Built once and
        held: the real tables give ~400k keys, and re-unioning them per unmatched name would cost
        more than the scoring does."""
        if self._all_keys is None:
            self._all_keys = list(set(self.directional) | set(self.wells_by_name))
        return self._all_keys


_DIRECTIONAL_FIELDS = {"API", "API_Label", "Operator", "Well_Name", "Lat", "Long"}
_WELLS_FIELDS = {
    "API", "API_Label", "Operator", "Well_Num", "Well_Name", "Well_Title", "Loc_Name",
    "Latitude", "Longitude",
}


def load_index(db_dir: str) -> WellIndex:
    """Read `Directional_Bottomhole_Locations.dbf` and `Wells.dbf` out of `db_dir` and build a
    `WellIndex` from them."""
    directional: dict[str, list[dict]] = {}
    dir_by_api: dict[str, list[dict]] = {}
    dir_path = os.path.join(db_dir, "Directional_Bottomhole_Locations.dbf")
    for rec in _read_dbf(dir_path, _DIRECTIONAL_FIELDS):
        key = _norm(rec.get("Well_Name"))
        if key:
            directional.setdefault(key, []).append(rec)
        api_key = _api_key(rec.get("API_Label"))
        if api_key:
            dir_by_api.setdefault(api_key, []).append(rec)

    wells_by_api: dict[str, dict] = {}
    wells_by_name: dict[str, list[dict]] = {}
    wells_path = os.path.join(db_dir, "Wells.dbf")
    for rec in _read_dbf(wells_path, _WELLS_FIELDS):
        api = rec.get("API")
        if api and api not in wells_by_api:
            wells_by_api[api] = rec
        well_name = rec.get("Well_Name", "")
        well_num = rec.get("Well_Num", "")
        for key in (
            _norm(rec.get("Well_Title")),
            _norm(well_name + well_num),
            _norm(well_num + well_name),
            _norm(rec.get("Loc_Name")),
        ):
            if not key:
                continue
            bucket = wells_by_name.setdefault(key, [])
            if rec not in bucket:
                bucket.append(rec)

    return WellIndex(directional=directional, wells_by_api=wells_by_api,
                      wells_by_name=wells_by_name, dir_by_api=dir_by_api)


# --------------------------------------------------------------------------------------------------
# matching
# --------------------------------------------------------------------------------------------------
@dataclass
class Match:
    db_well_name: str = ""
    api_label: str = ""
    operator: str = ""
    surf_lat: float | None = None
    surf_long: float | None = None
    btmh_lat: float | None = None
    btmh_long: float | None = None
    source: str = ""
    score: float | None = None

    @classmethod
    def empty(cls, source: str) -> "Match":
        """The no-match/no-questionnaire/no-well-name cases: only `source` is meaningful."""
        return cls(source=source)


def _pick_record(records: list[dict]) -> tuple[dict, str]:
    """When a key maps to several records, sort by API_Label and take the first. Returns the
    winning record and, if the records span more than one distinct API_Label, a
    " (N candidates)" suffix to append to the match source (no suffix if they're all the same
    API_Label, e.g. duplicate rows for the same well)."""
    distinct = sorted({r.get("API_Label", "") for r in records})
    winner = min(records, key=lambda r: r.get("API_Label", ""))
    suffix = f" ({len(distinct)} candidates)" if len(distinct) > 1 else ""
    return winner, suffix


_CANDIDATE_COUNT_RE = re.compile(r" \((\d+) candidates\)$")


def _combine_suffixes(a: str, b: str) -> str:
    """A row can be ambiguous twice over (the name-key bucket had >1 API_Label, and separately the
    `dir_by_api` sidetrack join had no exact API_Label match either) -- report the larger
    candidate count once rather than concatenating two ` (N candidates)` strings, so `_SUFFIX_RE`
    in `main`'s histogram still strips exactly one suffix."""
    def _n(s: str) -> int:
        m = _CANDIDATE_COUNT_RE.search(s)
        return int(m.group(1)) if m else 0
    return a if _n(a) >= _n(b) else b


def _from_directional(index: WellIndex, records: list[dict], source: str, score: float,
                       sibling: bool = False) -> Match:
    """`sibling=True` is the same-pad-sibling downgrade: the surface location still comes through
    the sibling's API link (`wells_by_api`), which is the whole point since the pad is shared, but
    the bottomhole -- this record's own `Lat`/`Long`, i.e. the sibling's, not the questionnaire
    well's -- is deliberately withheld."""
    rec, suffix = _pick_record(records)
    api_label = rec.get("API_Label", "")
    surf_lat = surf_long = None
    api_key = _api_key(api_label)
    if api_key:
        wells_rec = index.wells_by_api.get(api_key)
        if wells_rec is not None:
            surf_lat = _num(wells_rec.get("Latitude"))
            surf_long = _num(wells_rec.get("Longitude"))
    btmh_lat = btmh_long = None
    if not sibling:
        btmh_lat = _num(rec.get("Lat"))
        btmh_long = _num(rec.get("Long"))
    return Match(
        db_well_name=rec.get("Well_Name", ""),
        api_label=api_label,
        operator=rec.get("Operator", ""),
        surf_lat=surf_lat,
        surf_long=surf_long,
        btmh_lat=btmh_lat,
        btmh_long=btmh_long,
        source=source + suffix,
        score=score,
    )


def _from_wells(index: WellIndex, records: list[dict], source: str, score: float,
                sibling: bool = False) -> Match:
    """`sibling=True` withholds the bottomhole (looked up via `dir_by_api`, which would otherwise
    be the sibling's, not the questionnaire well's) while keeping the surface location, same
    rationale as `_from_directional`.

    One `_api_key` can carry several directional sidetrack records (1,018 do in the real tables,
    716 of those spread the reported bottomhole by more than 0.001 degrees), so an arbitrary
    `dir_recs[0]` silently reports whichever sidetrack happened to load first. Prefer the record
    whose own `API_Label` matches the wells record's `API_Label` exactly; only if none does, fall
    back to `_pick_record`'s "sort and flag" tie-break, and fold its ambiguity suffix into the
    match source so the row is visibly ambiguous rather than silently arbitrary."""
    rec, suffix = _pick_record(records)
    db_well_name = rec.get("Well_Title") or rec.get("Well_Name", "")
    btmh_lat = btmh_long = None
    if not sibling:
        dir_recs = index.dir_by_api.get(rec.get("API", ""))
        if dir_recs:
            api_label = rec.get("API_Label", "")
            exact = [d for d in dir_recs if d.get("API_Label", "") == api_label]
            if exact:
                dir_rec = exact[0]
            else:
                dir_rec, dir_suffix = _pick_record(dir_recs)
                suffix = _combine_suffixes(suffix, dir_suffix)
            btmh_lat = _num(dir_rec.get("Lat"))
            btmh_long = _num(dir_rec.get("Long"))
    return Match(
        db_well_name=db_well_name,
        api_label=rec.get("API_Label", ""),
        operator=rec.get("Operator", ""),
        surf_lat=_num(rec.get("Latitude")),
        surf_long=_num(rec.get("Longitude")),
        btmh_lat=btmh_lat,
        btmh_long=btmh_long,
        source=source + suffix,
        score=score,
    )


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _fuzzy_resolve(index: WellIndex, key: str) -> Match:
    """Acceptance rule: ratio >= 0.92 against the best candidate, margin over the runner-up
    >= 0.03, then `_digits_compatible` decides the shape of the result -- digit-compatible (no
    substituted digit) is a full match (`*-fuzzy`, both surface and bottomhole); digit-substituted
    is a same-pad sibling match (`*-sibling`, surface only, see `_from_directional`/`_from_wells`).
    """
    all_keys = index.all_keys()
    block = {k for k in all_keys if k[:3] == key[:3]}
    if not block:
        block = set(difflib.get_close_matches(key, all_keys, n=5, cutoff=0.85))
    if not block:
        return Match.empty("no-match")

    scored = sorted(((_ratio(key, k), k) for k in block), key=lambda t: -t[0])
    best_score, best_key = scored[0]
    if best_score < 0.92:
        return Match.empty("no-match")

    other_scores = [s for s, k in scored if k != best_key]
    second_best = other_scores[0] if other_scores else 0.0
    if best_score - second_best < 0.03:
        return Match.empty("no-match")

    score = round(best_score, 3)
    sibling = not _digits_compatible(key, best_key)
    if best_key in index.directional:
        source = "directional-sibling" if sibling else "directional-fuzzy"
        return _from_directional(index, index.directional[best_key], source, score, sibling=sibling)
    source = "wells-sibling" if sibling else "wells-fuzzy"
    return _from_wells(index, index.wells_by_name[best_key], source, score, sibling=sibling)


def resolve(index: WellIndex, name: str | None) -> Match:
    """Resolve a questionnaire well `name` against `index`: exact directional, then exact
    wells-by-name, then fuzzy, else no-match. A normalized key that is too short or has no
    3-letter run is rejected up front as `name-too-generic` (see the comment above
    `_MIN_KEY_LEN`) before any lookup runs. Results are memoized on the normalized key on
    `index._cache` (2,299 test rows collapse to ~207 unique names, so fuzzy scoring runs at most
    ~207 times)."""
    if name is None or not name.strip():
        return Match.empty("no-well-name")

    key = _norm(name)
    if len(key) < _MIN_KEY_LEN or not _LETTER_RUN_RE.search(key):
        return Match.empty("name-too-generic")

    cached = index._cache.get(key)
    if cached is not None:
        return cached

    if key in index.directional:
        match = _from_directional(index, index.directional[key], "directional-exact", 1.0)
    elif key in index.wells_by_name:
        match = _from_wells(index, index.wells_by_name[key], "wells-exact", 1.0)
    else:
        match = _fuzzy_resolve(index, key)

    index._cache[key] = match
    return match


# --------------------------------------------------------------------------------------------------
# row assembly
# --------------------------------------------------------------------------------------------------
def build_rows(root: str, index: WellIndex, progress: bool = True) -> tuple[list[dict], dict[str, int]]:
    """One row per `store.scan_root(root)` entry -- the exact same test queue the app builds.
    `list_tests` is deliberately not used here: it also loads every picks JSON and the log CSV,
    neither of which this script needs.

    Returns the rows and a small stats dict (`tests`, `questionnaires`) so `main` can report the
    questionnaire count without walking the tree a second time."""
    entries = store.scan_root(root)
    total = len(entries)
    qcache: dict[str, questionnaire.QuestionnaireResult | None] = {}
    rows: list[dict] = []

    for i, entry in enumerate(entries):
        if progress and total and (i % 100 == 0 or i == total - 1):
            print(f"  scanning {i + 1}/{total}...", file=sys.stderr)

        well_name: str | None = None
        if entry.questionnaire_path is None:
            match = Match.empty("no-questionnaire")
        else:
            qpath = entry.questionnaire_path
            if qpath not in qcache:
                try:
                    qcache[qpath] = questionnaire.parse_questionnaire(qpath)
                except Exception as e:  # a bad questionnaire must never abort the run
                    print(f"warning: failed to parse {qpath!r}: {e}", file=sys.stderr)
                    qcache[qpath] = None
            qres = qcache[qpath]
            if qres is None:
                # Only `parse_questionnaire` raising lands `qcache[qpath] = None` (see the
                # `except` above) -- a distinct source from "no questionnaire file", so a bad
                # file doesn't read as identical to a missing one in the printed histogram.
                match = Match.empty("questionnaire-error")
            else:
                well_name = qres.well_name
                match = resolve(index, well_name)

        rows.append({
            "TestID": entry.test_id,
            "WellName": well_name or "",
            "DBWellName": match.db_well_name or "",
            "API_Label": match.api_label or "",
            "Operator": match.operator or "",
            "SurfLat": match.surf_lat,
            "SurfLong": match.surf_long,
            "BtmhLat": match.btmh_lat,
            "BtmhLong": match.btmh_long,
            "MatchSource": match.source,
            "MatchScore": match.score,
        })

    stats = {
        "tests": total,
        "questionnaires": sum(1 for e in entries if e.questionnaire_path is not None),
    }
    return rows, stats


# --------------------------------------------------------------------------------------------------
# Excel output
# --------------------------------------------------------------------------------------------------
_COLUMN_WIDTHS = {
    "A": 70,  # TestID
    "B": 30,  # WellName
    "C": 30,  # DBWellName
    "D": 16,  # API_Label
    "E": 40,  # Operator
    "F": 12,  # SurfLat
    "G": 12,  # SurfLong
    "H": 12,  # BtmhLat
    "I": 12,  # BtmhLong
    "J": 16,  # MatchSource
    "K": 12,  # MatchScore
}


def write_xlsx(rows: list[dict], out_path: str) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_excel(out_path, index=False, sheet_name="Well Locations")

    wb = openpyxl.load_workbook(out_path)
    ws = wb["Well Locations"]
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col_letter, width in _COLUMN_WIDTHS.items():
        ws.column_dimensions[col_letter].width = width
    wb.save(out_path)


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a well-location Excel sheet for every DFIT test found under --root, "
                     "matched against the Wells/Directional COGCC shapefile attribute tables."
    )
    parser.add_argument("--root", default=DEFAULT_ROOT,
                         help=f"folder-mode root to scan (default: {DEFAULT_ROOT})")
    parser.add_argument("--db-dir", default=DEFAULT_DB_DIR,
                         help=f"folder holding Wells.dbf / Directional_Bottomhole_Locations.dbf "
                              f"(default: {DEFAULT_DB_DIR})")
    parser.add_argument("--out", default=DEFAULT_OUT,
                         help=f"output .xlsx path (default: {DEFAULT_OUT})")
    args = parser.parse_args()

    index = load_index(args.db_dir)
    rows, stats = build_rows(args.root, index)
    write_xlsx(rows, args.out)

    n_tests = stats["tests"]
    n_quest = stats["questionnaires"]
    n_names = sum(1 for r in rows if r["WellName"])

    histogram: Counter[str] = Counter()
    n_ambiguous = 0
    for r in rows:
        src = r["MatchSource"]
        if _SUFFIX_RE.search(src):
            n_ambiguous += 1
            src = _SUFFIX_RE.sub("", src)
        histogram[src] += 1

    print(f"Tests scanned: {n_tests}")
    print(f"Questionnaires found: {n_quest}")
    print(f"Well names found: {n_names}")
    print("MatchSource histogram:")
    for src, count in histogram.most_common():
        print(f"  {src}: {count}")
    print(f"  (rows with multiple API candidates: {n_ambiguous})")
    print(args.out)


if __name__ == "__main__":
    main()
