"""Unit tests for scripts/well_locations.py: the inlined DBF reader, name normalization, and the
directional/wells/fuzzy matching precedence. Headless, and deliberately never touches the real
94 MB `Well Locations/*.dbf` files or `C:\\DFIT Data` -- every `WellIndex` here is built directly
from small dicts standing in for DBF records.

`scripts/` is not a package (mirrors how the script itself reaches into `dfit_tool`), so the repo
root and the scripts directory are added to sys.path here, the same fixup `well_locations.py`
does for `dfit_tool`.
"""

from __future__ import annotations

import os
import struct
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import well_locations  # noqa: E402


# --------------------------------------------------------------------------------------------------
# _norm
# --------------------------------------------------------------------------------------------------
def test_norm_collapses_punctuation_case_hash_and_spacing():
    a = well_locations._norm("Carson Federal #3TFH")
    b = well_locations._norm("carson federal 3tfh")
    assert a == b == "CARSONFEDERAL3TFH"


def test_norm_none_and_blank():
    assert well_locations._norm(None) == ""
    assert well_locations._norm("   ") == ""


# --------------------------------------------------------------------------------------------------
# _read_dbf
# --------------------------------------------------------------------------------------------------
def _dbf_bytes(fields: list[tuple[str, int]], records: list[dict]) -> bytes:
    """Build a minimal dBase III .dbf file as bytes: a 32-byte header, one 32-byte field
    descriptor per entry in `fields`, a 0x0D terminator, then one fixed-width record per entry in
    `records` (each record dict maps field name -> value; `_deleted=True` sets the delete flag)."""
    rlen = 1 + sum(length for _, length in fields)
    header = bytearray(32)
    header[0] = 0x03  # version
    header[4:8] = struct.pack("<I", len(records))
    header[8:10] = struct.pack("<H", 32 + 32 * len(fields) + 1)  # hlen
    header[10:12] = struct.pack("<H", rlen)
    out = bytes(header)

    for name, length in fields:
        desc = bytearray(32)
        name_bytes = name.encode("latin-1")[:11]
        desc[0:len(name_bytes)] = name_bytes
        desc[11] = ord("C")
        desc[16] = length
        out += bytes(desc)
    out += b"\x0D"

    for rec in records:
        line = bytearray()
        line.append(0x2A if rec.get("_deleted") else 0x20)
        for name, length in fields:
            value = str(rec.get(name, ""))
            raw = value.encode("latin-1")[:length]
            raw = raw + b" " * (length - len(raw))
            line += raw
        out += bytes(line)
    out += b"\x1A"  # EOF marker
    return bytes(out)


def test_read_dbf_round_trip(tmp_path):
    fields = [("NAME", 10), ("API", 8)]
    records = [
        {"NAME": "Well A", "API": "12345678"},
        {"NAME": "Deleted", "API": "99999999", "_deleted": True},
        {"NAME": "  Well B  ", "API": "87654321"},
    ]
    path = tmp_path / "sample.dbf"
    path.write_bytes(_dbf_bytes(fields, records))

    rows = list(well_locations._read_dbf(str(path)))
    assert rows == [
        {"NAME": "Well A", "API": "12345678"},
        {"NAME": "Well B", "API": "87654321"},
    ]


def test_read_dbf_fields_whitelist(tmp_path):
    fields = [("NAME", 10), ("API", 8)]
    records = [{"NAME": "Well A", "API": "12345678"}]
    path = tmp_path / "sample.dbf"
    path.write_bytes(_dbf_bytes(fields, records))

    rows = list(well_locations._read_dbf(str(path), fields={"API"}))
    assert rows == [{"API": "12345678"}]


def test_read_dbf_truncated_final_record_stops_and_warns(tmp_path, capsys):
    """FIX E: a short final read must break (not yield a partial row, not spin the remaining
    header-declared records at EOF), and must warn on stderr naming the file and record index."""
    fields = [("NAME", 10), ("API", 8)]
    records = [
        {"NAME": "Well A", "API": "12345678"},
        {"NAME": "Well B", "API": "87654321"},
    ]
    full = _dbf_bytes(fields, records)
    # Header still claims 2 records; keep all of record 0's bytes plus only 3 of record 1's
    # `rlen` bytes, so the second (index-1) record read comes back short.
    hlen = 32 + 32 * len(fields) + 1
    rlen = 1 + sum(length for _, length in fields)
    truncated = full[:hlen + rlen + 3]
    path = tmp_path / "sample.dbf"
    path.write_bytes(truncated)

    rows = list(well_locations._read_dbf(str(path)))
    assert rows == [{"NAME": "Well A", "API": "12345678"}]  # the partial second record is dropped

    err = capsys.readouterr().err
    assert "sample.dbf" in err  # names the file (message uses !r, so no bare path substring match)
    assert "record 1" in err  # index of the truncated record


def test_read_dbf_fully_empty_read_breaks_not_continues(tmp_path):
    """A read that returns nothing at all (EOF before `nrec` records) must stop the loop too, not
    just the short-but-nonempty case."""
    fields = [("NAME", 10), ("API", 8)]
    records = [{"NAME": "Well A", "API": "12345678"}]
    full = _dbf_bytes(fields, records)
    hlen = 32 + 32 * len(fields) + 1
    rlen = 1 + sum(length for _, length in fields)
    # Bump the header's declared record count to 5 but keep only 1 record's worth of bytes.
    full = bytearray(full)
    struct.pack_into("<I", full, 4, 5)
    truncated = bytes(full[:hlen + rlen])
    path = tmp_path / "sample.dbf"
    path.write_bytes(truncated)

    rows = list(well_locations._read_dbf(str(path)))
    assert rows == [{"NAME": "Well A", "API": "12345678"}]


# --------------------------------------------------------------------------------------------------
# resolve: exact-match precedence
# --------------------------------------------------------------------------------------------------
def _index(directional=None, wells_by_api=None, wells_by_name=None, dir_by_api=None):
    return well_locations.WellIndex(
        directional=directional or {},
        wells_by_api=wells_by_api or {},
        wells_by_name=wells_by_name or {},
        dir_by_api=dir_by_api or {},
    )


def test_resolve_directional_exact_beats_wells_exact_for_same_key():
    dir_rec = {
        "API": "99900001", "API_Label": "05-999-00001-00", "Operator": "DirOp",
        "Well_Name": "Shared Well", "Lat": "41.000", "Long": "-105.000",
    }
    wells_rec = {
        "API": "99900002", "API_Label": "05-999-00002-00", "Operator": "WellsOp",
        "Well_Num": "", "Well_Name": "Shared Well", "Well_Title": "Shared Well",
        "Loc_Name": "", "Latitude": "41.100", "Longitude": "-105.100",
    }
    index = _index(
        directional={"SHAREDWELL": [dir_rec]},
        wells_by_name={"SHAREDWELL": [wells_rec]},
    )

    match = well_locations.resolve(index, "Shared Well")
    assert match.source == "directional-exact"
    assert match.operator == "DirOp"
    assert match.db_well_name == "Shared Well"


def test_resolve_wells_only_fills_surf_and_leaves_btmh_none():
    wells_rec = {
        "API": "55500001", "API_Label": "05-555-00001-00", "Operator": "OpX",
        "Well_Num": "5H", "Well_Name": "Wells Only", "Well_Title": "Wells Only 5H",
        "Loc_Name": "", "Latitude": "39.500", "Longitude": "-103.500",
    }
    index = _index(wells_by_name={"WELLSONLY5H": [wells_rec]})

    match = well_locations.resolve(index, "Wells Only 5H")
    assert match.source == "wells-exact"
    assert match.surf_lat == 39.5
    assert match.surf_long == -103.5
    assert match.btmh_lat is None
    assert match.btmh_long is None


def test_resolve_wells_exact_fills_btmh_via_dir_by_api_join():
    """FIX C: `_from_wells`'s bottomhole join keys `dir_by_api` on the wells record's own `API`
    field, which `load_index` populates from `_api_key(directional.API_Label)` -- NOT from the
    directional record's own 10-char `API` column, a different key space. Give the directional
    record a bogus `API` so a regression to keying on it would fail to find the bottomhole."""
    wells_rec = {
        "API": "12345670", "API_Label": "05-123-45670-01", "Operator": "Acme Op",
        "Well_Num": "1H", "Well_Name": "Well A", "Well_Title": "Well A 1H",
        "Loc_Name": "", "Latitude": "40.7000", "Longitude": "-104.8200",
    }
    dir_rec = {
        "API": "0099999999",  # deliberately NOT "12345670" -- must never be the join key
        "API_Label": "05-123-45670-01", "Operator": "Acme Op",
        "Well_Name": "Well A", "Lat": "40.7100", "Long": "-104.8300",
    }
    index = _index(
        wells_by_name={"WELLA1H": [wells_rec]},
        dir_by_api={"12345670": [dir_rec]},  # keyed as load_index would: _api_key(API_Label)
    )

    match = well_locations.resolve(index, "Well A 1H")
    assert match.source == "wells-exact"
    assert match.btmh_lat == 40.71
    assert match.btmh_long == -104.83


def test_resolve_directional_fills_btmh_and_surf_via_api_link():
    dir_rec = {
        "API": "12345670", "API_Label": "05-123-45670-00", "Operator": "Acme Op",
        "Well_Name": "Well A", "Lat": "40.7100", "Long": "-104.8300",
    }
    wells_rec = {
        "API": "12345670", "API_Label": "05-123-45670-00", "Operator": "Acme Op",
        "Well_Num": "1H", "Well_Name": "Well A", "Well_Title": "Well A 1H",
        "Loc_Name": "", "Latitude": "40.7000", "Longitude": "-104.8200",
    }
    index = _index(
        directional={"WELLA": [dir_rec]},
        wells_by_api={"12345670": wells_rec},
    )

    match = well_locations.resolve(index, "Well A")
    assert match.source == "directional-exact"
    assert match.btmh_lat == 40.71
    assert match.btmh_long == -104.83
    assert match.surf_lat == 40.7
    assert match.surf_long == -104.82


# --------------------------------------------------------------------------------------------------
# resolve: fuzzy threshold and margin
# --------------------------------------------------------------------------------------------------
def test_resolve_fuzzy_single_close_candidate_matches():
    dir_rec = {
        "API": "11100001", "API_Label": "05-111-00001-00", "Operator": "FuzzyOp",
        "Well_Name": "Carson Federal 3TFHX", "Lat": "40.0", "Long": "-104.0",
    }
    index = _index(directional={"CARSONFEDERAL3TFHX": [dir_rec]})

    match = well_locations.resolve(index, "Carson Federal 3TFH")
    assert match.source.endswith("-fuzzy")
    assert match.source == "directional-fuzzy"
    assert match.score is not None and match.score >= 0.92


def test_resolve_fuzzy_two_near_tied_candidates_is_no_match():
    rec1 = {
        "API": "22200001", "API_Label": "05-222-00001-00", "Operator": "OpA",
        "Well_Name": "Carson Federal 3TFHX", "Lat": "40.0", "Long": "-104.0",
    }
    rec2 = {
        "API": "22200002", "API_Label": "05-222-00002-00", "Operator": "OpB",
        "Well_Name": "Carson Federal 39TFH", "Lat": "40.1", "Long": "-104.1",
    }
    index = _index(directional={
        "CARSONFEDERAL3TFHX": [rec1],
        "CARSONFEDERAL39TFH": [rec2],
    })

    match = well_locations.resolve(index, "Carson Federal 3TFH")
    assert match.source == "no-match"


def test_resolve_fuzzy_below_threshold_is_no_match():
    dir_rec = {
        "API": "33300001", "API_Label": "05-333-00001-00", "Operator": "OpC",
        "Well_Name": "Carson Federal", "Lat": "40.0", "Long": "-104.0",
    }
    index = _index(directional={"CARSONFEDERAL": [dir_rec]})

    match = well_locations.resolve(index, "Carson Federal 3TFH")
    assert match.source == "no-match"


# --------------------------------------------------------------------------------------------------
# _digits_compatible
# --------------------------------------------------------------------------------------------------
def test_digits_compatible_true_for_insertions_and_deletions():
    cases = [
        ("Allred Fed 08-59 30-31-06", "ALLRED FED 8-59 30-31-6"),
        ("Brant LE 162HNX", "Brant LE 08-162HNX"),
        ("Critter Creek 248 2412", "Critter Creek 248-2412H"),
        ("State Massive 1CH", "State Massive 1H"),
    ]
    for a, b in cases:
        assert well_locations._digits_compatible(
            well_locations._norm(a), well_locations._norm(b)
        ), f"{a!r} vs {b!r} should be digit-compatible"


def test_digits_compatible_false_for_substitution():
    a = well_locations._norm("Berry IC 15-159HC")
    b = well_locations._norm("BERRY IC 11-159HC")
    assert not well_locations._digits_compatible(a, b)


def test_digits_compatible_true_for_no_digits_at_all():
    a = well_locations._norm("Foo Bar Well")
    b = well_locations._norm("FOOBARWELL")
    assert well_locations._digits_compatible(a, b)


# --------------------------------------------------------------------------------------------------
# resolve: fuzzy sibling downgrade (digit-substituted match)
# --------------------------------------------------------------------------------------------------
def test_resolve_fuzzy_digit_substitution_downgrades_to_directional_sibling():
    dir_rec = {
        "API": "09100001", "API_Label": "05-091-00001-00", "Operator": "SiblingOp",
        "Well_Name": "BERRY IC 11-159HC", "Lat": "40.500", "Long": "-104.500",
    }
    wells_rec = {
        "API": "09100001", "API_Label": "05-091-00001-00", "Operator": "SiblingOp",
        "Well_Num": "", "Well_Name": "BERRY IC 11-159HC", "Well_Title": "BERRY IC 11-159HC",
        "Loc_Name": "", "Latitude": "40.400", "Longitude": "-104.400",
    }
    index = _index(
        directional={"BERRYIC11159HC": [dir_rec]},
        wells_by_api={"09100001": wells_rec},
    )

    key = well_locations._norm("Berry IC 15-159HC")
    best_key = well_locations._norm("BERRY IC 11-159HC")
    raw_ratio = well_locations._ratio(key, best_key)
    assert raw_ratio >= 0.92  # proves the downgrade, not the threshold, produced the sibling result

    match = well_locations.resolve(index, "Berry IC 15-159HC")
    assert match.source == "directional-sibling"
    assert match.surf_lat == 40.4
    assert match.surf_long == -104.4
    assert match.btmh_lat is None
    assert match.btmh_long is None
    assert match.score is not None and abs(match.score - 0.929) < 0.001


def test_resolve_fuzzy_digit_substitution_downgrades_to_wells_sibling():
    wells_rec = {
        "API": "09100002", "API_Label": "05-091-00002-00", "Operator": "SibOp2",
        "Well_Num": "", "Well_Name": "BERRY IC 11-159HC", "Well_Title": "BERRY IC 11-159HC",
        "Loc_Name": "", "Latitude": "41.000", "Longitude": "-105.000",
    }
    dir_rec_for_api = {
        "API": "09100002", "API_Label": "05-091-00002-00", "Operator": "SibOp2",
        "Well_Name": "BERRY IC 11-159HC", "Lat": "41.999", "Long": "-105.999",
    }
    index = _index(
        wells_by_name={"BERRYIC11159HC": [wells_rec]},
        dir_by_api={"09100002": [dir_rec_for_api]},
    )

    match = well_locations.resolve(index, "Berry IC 15-159HC")
    assert match.source == "wells-sibling"
    assert match.surf_lat == 41.0
    assert match.surf_long == -105.0
    assert match.btmh_lat is None
    assert match.btmh_long is None


# --------------------------------------------------------------------------------------------------
# resolve: ambiguity (multiple distinct API_Labels under one key)
# --------------------------------------------------------------------------------------------------
def test_resolve_ambiguous_key_appends_candidate_count_and_picks_first_api():
    rec_high = {
        "API": "00100002", "API_Label": "05-001-00002-00", "Operator": "OpHigh",
        "Well_Name": "Ambig Well", "Lat": "40.2", "Long": "-104.2",
    }
    rec_low = {
        "API": "00100001", "API_Label": "05-001-00001-00", "Operator": "OpLow",
        "Well_Name": "Ambig Well", "Lat": "40.1", "Long": "-104.1",
    }
    index = _index(directional={"AMBIGWELL": [rec_high, rec_low]})

    match = well_locations.resolve(index, "Ambig Well")
    assert match.source == "directional-exact (2 candidates)"
    assert match.operator == "OpLow"
    assert match.api_label == "05-001-00001-00"


# --------------------------------------------------------------------------------------------------
# _from_wells: sidetrack preference (FIX B)
# --------------------------------------------------------------------------------------------------
def test_from_wells_prefers_dir_record_matching_wells_api_label_exactly():
    """Two directional sidetrack records share one `_api_key`, with different `API_Label`s and
    different coordinates. The one whose `API_Label` matches the wells record's exactly must win,
    regardless of list order (`dir_recs[0]` would silently pick the wrong one here)."""
    wells_rec = {
        "API": "55500001", "API_Label": "05-555-00001-01", "Operator": "OpX",
        "Well_Num": "", "Well_Name": "Sidetrack Well", "Well_Title": "Sidetrack Well",
        "Loc_Name": "", "Latitude": "39.000", "Longitude": "-103.000",
    }
    dir_rec_wrong = {
        "API": "55500001", "API_Label": "05-555-00001-00", "Operator": "OpX",
        "Well_Name": "Sidetrack Well", "Lat": "42.000", "Long": "-106.000",
    }
    dir_rec_right = {
        "API": "55500001", "API_Label": "05-555-00001-01", "Operator": "OpX",
        "Well_Name": "Sidetrack Well", "Lat": "41.000", "Long": "-105.000",
    }
    index = _index(
        wells_by_name={"SIDETRACKWELL": [wells_rec]},
        dir_by_api={"55500001": [dir_rec_wrong, dir_rec_right]},  # wrong one listed first
    )

    match = well_locations.resolve(index, "Sidetrack Well")
    assert match.source == "wells-exact"  # exact API_Label match: no ambiguity suffix
    assert match.btmh_lat == 41.0
    assert match.btmh_long == -105.0


# --------------------------------------------------------------------------------------------------
# resolve: name-too-generic guard (FIX A)
# --------------------------------------------------------------------------------------------------
def test_resolve_rejects_bare_designator_even_when_it_would_exact_match():
    """Real case: questionnaire `1BH` normalized to a `wells_by_name` key manufactured from an
    unrelated well's `Well_Num="1"` + `Well_Name="B & H"`. The guard must reject the key before
    the lookup runs, not just when there happens to be no candidate."""
    wrong_well = {
        "API": "12314136", "API_Label": "05-123-14136-00", "Operator": "PDC Energy",
        "Well_Num": "1", "Well_Name": "B & H", "Well_Title": "B & H 1",
        "Loc_Name": "", "Latitude": "40.000", "Longitude": "-104.000",
    }
    index = _index(wells_by_name={"1BH": [wrong_well]})

    match = well_locations.resolve(index, "1BH")
    assert match.source == "name-too-generic"
    assert match.api_label == ""


def test_resolve_rejects_short_key_with_no_letter_run():
    index = _index(wells_by_name={"5C30M": [{"API_Label": "05-001-00001-00"}]})
    assert well_locations.resolve(index, "5C30M").source == "name-too-generic"


def test_resolve_accepts_short_key_with_three_letter_run():
    """`Cox 5` -> `COX5`: 4 characters, but the 3-letter run `COX` makes it a real lease-name
    token plus a designator, not a bare designator -- the guard must let it through."""
    dir_rec = {
        "API": "00700001", "API_Label": "05-007-00001-00", "Operator": "CoxOp",
        "Well_Name": "Cox 5", "Lat": "40.000", "Long": "-104.000",
    }
    index = _index(directional={"COX5": [dir_rec]})

    match = well_locations.resolve(index, "Cox 5")
    assert match.source == "directional-exact"


def test_resolve_no_well_name_checked_before_name_too_generic():
    """A blank/None name must still report `no-well-name`, not `name-too-generic` -- the guard
    only applies once there is a name to normalize."""
    index = _index()
    assert well_locations.resolve(index, None).source == "no-well-name"
    assert well_locations.resolve(index, "").source == "no-well-name"


# --------------------------------------------------------------------------------------------------
# resolve: no well name
# --------------------------------------------------------------------------------------------------
def test_resolve_no_well_name_for_none_and_blank():
    index = _index()
    assert well_locations.resolve(index, None).source == "no-well-name"
    assert well_locations.resolve(index, "   ").source == "no-well-name"
    assert well_locations.resolve(index, "").source == "no-well-name"


# --------------------------------------------------------------------------------------------------
# build_rows: questionnaire-error vs no-questionnaire (FIX D)
# --------------------------------------------------------------------------------------------------
def test_build_rows_questionnaire_parse_error_gets_its_own_source(monkeypatch, tmp_path, capsys):
    """A questionnaire file that exists but fails to parse must report `questionnaire-error`, not
    `no-questionnaire` (that source is reserved for a test with no questionnaire file at all), and
    the stderr warning must still fire."""
    qpath = str(tmp_path / "t1_questionnaire.xlsx")
    entry = well_locations.store.TestEntry(
        test_id="t1", folder=str(tmp_path), csv_path=str(tmp_path / "t1.csv"),
        questionnaire_path=qpath,
    )
    monkeypatch.setattr(well_locations.store, "scan_root", lambda root: [entry])

    def _raise(path):
        raise ValueError("boom")
    monkeypatch.setattr(well_locations.questionnaire, "parse_questionnaire", _raise)

    index = _index()
    rows, stats = well_locations.build_rows(str(tmp_path), index, progress=False)

    assert rows[0]["MatchSource"] == "questionnaire-error"
    err = capsys.readouterr().err
    assert "t1_questionnaire.xlsx" in err and "boom" in err


def test_build_rows_missing_questionnaire_file_is_no_questionnaire(monkeypatch, tmp_path):
    """A test with no questionnaire file at all is a distinct case from a bad one."""
    entry = well_locations.store.TestEntry(
        test_id="t1", folder=str(tmp_path), csv_path=str(tmp_path / "t1.csv"),
        questionnaire_path=None,
    )
    monkeypatch.setattr(well_locations.store, "scan_root", lambda root: [entry])

    index = _index()
    rows, stats = well_locations.build_rows(str(tmp_path), index, progress=False)

    assert rows[0]["MatchSource"] == "no-questionnaire"
