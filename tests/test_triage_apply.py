"""Unit tests for scripts/triage/apply.py: destination-path construction and the move/copy plan.

Headless (no matplotlib, no Tkinter) and never touches the real `C:\\DFIT Data` corpus -- every
`FolderScan`/`FileFeatures`/`Ledger` here is built directly, under `tmp_path`, the same way
`tests/test_well_locations.py` builds `WellIndex` directly from dicts instead of reading the real
`.dbf` files.

`scripts/` is not a package (mirrors `well_locations.py`/`test_well_locations.py`'s own fixup),
so the repo root and the scripts directory are added to `sys.path` here.

`triage.basins.basin_for` is monkeypatched in most tests so this file exercises `apply.py`'s own
logic (destination construction, collision handling, move planning, execution) rather than the
content of the real formation/customer tables in `triage/basins.py`.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from triage import apply as apply_mod  # noqa: E402
from triage.basins import DEFAULT_BASIN  # noqa: E402
from triage.features import FileFeatures, FolderScan  # noqa: E402
from triage.ledger import FolderDecision, Ledger, group_files_sig  # noqa: E402


def _fixed_basin(monkeypatch, basin: str = "TestBasin", source: str = "customer") -> None:
    """Replace `basins.basin_for` with a constant answer, decoupling these tests from the real
    (and evolving) formation/customer tables."""
    monkeypatch.setattr(apply_mod.basins, "basin_for", lambda formation, customer: (basin, source))


def _decide(ledger: Ledger, scan: FolderScan, keeps: list[str], status: str = "decided") -> None:
    """`ledger.set` with `files_sig` computed from `scan`'s own current file set (FIX 2 --
    decision fingerprint guard), so `plan_moves`/`plan_warnings` treat the decision as CURRENT
    rather than stale -- the normal case every `plan_moves` test below wants, unless it is
    specifically exercising staleness itself."""
    ledger.set(scan.rel, keeps, status, files_sig=group_files_sig(f.sig for f in scan.files))


# --------------------------------------------------------------------------------------------------
# well_folder_name
# --------------------------------------------------------------------------------------------------
def test_well_folder_name_questionnaire_name_wins(tmp_path):
    scan = FolderScan(folder=str(tmp_path / "some_folder"), rel="C/some_folder",
                       well_name="Actual Well")
    assert apply_mod.well_folder_name(scan) == "Actual Well"


def test_well_folder_name_blank_falls_back_to_folder_basename(tmp_path):
    scan = FolderScan(folder=str(tmp_path / "Folder Name Here"), rel="C/Folder Name Here",
                       well_name="")
    assert apply_mod.well_folder_name(scan) == "Folder Name Here"


def test_well_folder_name_collapses_whitespace_runs(tmp_path):
    scan = FolderScan(folder=str(tmp_path / "x"), rel="C/x", well_name="  Foo   Well  ")
    assert apply_mod.well_folder_name(scan) == "Foo Well"


def test_well_folder_name_sanitizes_forbidden_chars_and_trailing_dot(tmp_path):
    scan = FolderScan(folder=str(tmp_path / "x"), rel="C/x", well_name='Foo/Bar: "Well".')
    name = apply_mod.well_folder_name(scan)
    for ch in '<>:"/\\|?*':
        assert ch not in name
    assert not name.endswith(".")
    assert not name.endswith(" ")


# --------------------------------------------------------------------------------------------------
# FIX 2: a well_name that sanitizes to "" must never make well_folder_name return "" -- it must
# fall back to the folder's own basename (and, if that is also unusable, to a placeholder derived
# from `rel`, never an empty string).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("well_name", ["???", "...", "   ", '<>:"/\\|?*'])
def test_well_folder_name_unusable_well_name_falls_back_to_folder_basename(tmp_path, well_name):
    scan = FolderScan(folder=str(tmp_path / "Some Well Folder"), rel="Cust/Some Well Folder",
                       well_name=well_name)
    name = apply_mod.well_folder_name(scan)
    assert name != ""
    assert name == "Some Well Folder"


def test_well_folder_name_well_name_and_folder_basename_both_unusable_falls_back_to_rel(tmp_path):
    # The folder itself is named "???" too, so the folder-basename fallback is also unusable --
    # the next fallback is a placeholder built from the scan's `rel` (its full relative path, not
    # just the leaf, which would just repeat the unusable folder basename).
    scan = FolderScan(folder=str(tmp_path / "CustomerA" / "???"), rel="CustomerA/???",
                       well_name="???")
    name = apply_mod.well_folder_name(scan)
    assert name != ""
    assert name == "CustomerA_"


def test_well_folder_name_everything_unusable_falls_back_to_fixed_placeholder():
    scan = FolderScan(folder="???", rel="???", well_name="???")
    name = apply_mod.well_folder_name(scan)
    assert name == "_unnamed_well"


# --------------------------------------------------------------------------------------------------
# destination_dir
# --------------------------------------------------------------------------------------------------
def test_destination_dir_formation_wins_over_customer(monkeypatch, tmp_path):
    def fake_basin_for(formation, customer):
        if formation:
            return (f"Basin-{formation}", "formation")
        if customer:
            return (f"Basin-{customer}", "customer")
        return (DEFAULT_BASIN, "default")

    monkeypatch.setattr(apply_mod.basins, "basin_for", fake_basin_for)

    out_root = str(tmp_path / "out")
    scan = FolderScan(folder=str(tmp_path / "CustA" / "Well1"), rel="CustA/Well1",
                       well_name="Well A", formation="Niobrara")
    dest_dir, basin, source = apply_mod.destination_dir(scan, out_root, str(tmp_path))

    assert basin == "Basin-Niobrara"
    assert source == "formation"
    assert dest_dir == os.path.join(out_root, "Basin-Niobrara", "Well A")


def test_destination_dir_blank_formation_falls_back_to_customer(monkeypatch, tmp_path):
    def fake_basin_for(formation, customer):
        if formation:
            return (f"Basin-{formation}", "formation")
        if customer:
            return (f"Basin-{customer}", "customer")
        return (DEFAULT_BASIN, "default")

    monkeypatch.setattr(apply_mod.basins, "basin_for", fake_basin_for)

    out_root = str(tmp_path / "out")
    scan = FolderScan(folder=str(tmp_path / "CustB" / "Well2"), rel="CustB/Well2",
                       well_name="Well B", formation="")
    dest_dir, basin, source = apply_mod.destination_dir(scan, out_root, str(tmp_path))

    assert basin == "Basin-CustB"
    assert source == "customer"
    assert dest_dir == os.path.join(out_root, "Basin-CustB", "Well B")


def test_destination_dir_neither_formation_nor_customer_is_default_basin(monkeypatch, tmp_path):
    def fake_basin_for(formation, customer):
        if formation:
            return (f"Basin-{formation}", "formation")
        if customer:
            return (f"Basin-{customer}", "customer")
        return (DEFAULT_BASIN, "default")

    monkeypatch.setattr(apply_mod.basins, "basin_for", fake_basin_for)

    out_root = str(tmp_path / "out")
    scan = FolderScan(folder=str(tmp_path / "Well3"), rel="", well_name="Well C", formation="")
    dest_dir, basin, source = apply_mod.destination_dir(scan, out_root, str(tmp_path))

    assert basin == DEFAULT_BASIN
    assert source == "default"
    assert dest_dir == os.path.join(out_root, DEFAULT_BASIN, "Well C")


def test_destination_dir_passes_first_rel_segment_as_customer(monkeypatch, tmp_path):
    seen = []

    def fake_basin_for(formation, customer):
        seen.append((formation, customer))
        return ("SomeBasin", "customer")

    monkeypatch.setattr(apply_mod.basins, "basin_for", fake_basin_for)

    scan = FolderScan(folder=str(tmp_path / "CustA" / "Sub" / "Well1"),
                       rel="CustA/Sub/Well1", well_name="Well A", formation="Codell")
    apply_mod.destination_dir(scan, str(tmp_path / "out"), str(tmp_path))

    assert seen == [("Codell", "CustA")]


# --------------------------------------------------------------------------------------------------
# plan_moves
# --------------------------------------------------------------------------------------------------
def _write(path, data=b"data"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)


def test_plan_moves_decided_folder_keep_questionnaire_and_quarantine(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    keeper = _write(folder / "keeper.dbs", b"keep")
    other = _write(folder / "other.csv", b"other")
    quest = _write(folder / "questionnaire.xlsx", b"quest")

    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        questionnaire_path=quest,
        files=[
            FileFeatures(path=keeper, folder=str(folder), size_bytes=4, sig="a",
                         verdict="likely_dfit"),
            FileFeatures(path=other, folder=str(folder), size_bytes=5, sig="b",
                         verdict="too_short"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [keeper])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)

    by_kind = {m.kind: m for m in moves}
    assert set(by_kind) == {"keep", "questionnaire", "quarantine"}

    keep_mv = by_kind["keep"]
    assert keep_mv.src == keeper
    assert keep_mv.dst == os.path.join(out_root, "TestBasin", "Well One", "keeper.dbs")
    assert keep_mv.skipped == ""

    quest_mv = by_kind["questionnaire"]
    assert quest_mv.src == quest
    assert quest_mv.dst == os.path.join(out_root, "TestBasin", "Well One", "questionnaire.xlsx")
    assert quest_mv.skipped == ""

    quarantine_mv = by_kind["quarantine"]
    assert quarantine_mv.src == other
    assert quarantine_mv.dst == os.path.join(str(root), "_quarantine", "CustomerA", "Well1",
                                              "other.csv")
    assert quarantine_mv.skipped == ""


def test_plan_moves_two_keeps_one_folder_share_dest_dir_no_skip(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "test1.dbs", b"1")
    f2 = _write(folder / "test2.dbs", b"2")

    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="likely_dfit"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="likely_dfit"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1, f2])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    keep_moves = [m for m in moves if m.kind == "keep"]

    assert len(keep_moves) == 2
    assert all(m.skipped == "" for m in keep_moves)
    dest_dirs = {os.path.dirname(m.dst) for m in keep_moves}
    assert dest_dirs == {os.path.join(out_root, "TestBasin", "Well One")}
    assert {m.dst for m in keep_moves} == {
        os.path.join(out_root, "TestBasin", "Well One", "test1.dbs"),
        os.path.join(out_root, "TestBasin", "Well One", "test2.dbs"),
    }


def test_plan_moves_undecided_and_unsure_folders_produce_zero_moves(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder1 = root / "CustomerA" / "Well1"
    f1 = _write(folder1 / "a.dbs", b"1")
    scan1 = FolderScan(folder=str(folder1), rel="CustomerA/Well1",
                        files=[FileFeatures(path=f1, folder=str(folder1), size_bytes=1, sig="a",
                                             verdict="likely_dfit")])

    folder2 = root / "CustomerB" / "Well2"
    f2 = _write(folder2 / "b.dbs", b"2")
    scan2 = FolderScan(folder=str(folder2), rel="CustomerB/Well2",
                        files=[FileFeatures(path=f2, folder=str(folder2), size_bytes=1, sig="b",
                                             verdict="likely_dfit")])

    ledger = Ledger.load(str(root))
    ledger.set(scan2.rel, [f2], "unsure")
    # scan1's decision is left completely undecided (never `.set`).

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan1, scan2], ledger)
    assert moves == []


def test_plan_moves_none_folder_quarantines_all_no_questionnaire_copy(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.csv", b"2")
    quest = _write(folder / "q.xlsx", b"q")

    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", questionnaire_path=quest,
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="flat"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="flat"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [], "none")

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)

    assert len(moves) == 2
    assert all(m.kind == "quarantine" for m in moves)
    assert {m.src for m in moves} == {f1, f2}


# --------------------------------------------------------------------------------------------------
# FIX 2 (decision fingerprint guard): a decided group whose files_sig no longer matches its
# current file set -- or has none at all (legacy) -- is skipped by plan_moves and warned about by
# plan_warnings, never silently planned.
# --------------------------------------------------------------------------------------------------
def test_plan_moves_stale_fingerprint_mismatch_produces_zero_moves(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.dbs", b"2")  # a file that "arrived" after the decision was made
    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="likely_dfit"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="likely_dfit"),
        ],
    )
    ledger = Ledger.load(str(root))
    # Decision recorded against a fingerprint computed from only f1's signature -- stale relative
    # to the scan's actual (2-file) current set.
    ledger.set(scan.rel, [f1], "decided", files_sig=group_files_sig(["a"]))

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)
    assert moves == []


def test_plan_moves_legacy_decision_no_files_sig_produces_zero_moves(monkeypatch, tmp_path):
    """A decision recorded before this fingerprint existed (`files_sig == ""`, the default) must
    be treated exactly like a mismatch -- never silently trusted just because it happens to be
    the only decision on file for this `rel`."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    ledger.set(scan.rel, [f1], "decided")  # no files_sig at all

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)
    assert moves == []


def test_plan_moves_matching_fingerprint_plans_normally(monkeypatch, tmp_path):
    """The normal case, pinned directly: a decision whose `files_sig` matches the group's current
    fingerprint is planned exactly like before FIX 2 existed."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)
    keep_moves = [m for m in moves if m.kind == "keep"]
    assert len(keep_moves) == 1
    assert keep_moves[0].src == f1


def test_plan_warnings_stale_decision_reports_and_matching_does_not(monkeypatch, tmp_path):
    """MUTATION: disable the fingerprint comparison in `_exclusion_reason` (always treat as
    current) -> the stale-decision line disappears from `plan_warnings`' output and this test
    fails."""
    _fixed_basin(monkeypatch)
    root = tmp_path

    stale_folder = root / "CustomerA" / "Stale"
    fs1 = _write(stale_folder / "a.dbs", b"1")
    fs2 = _write(stale_folder / "b.dbs", b"2")
    stale_scan = FolderScan(
        folder=str(stale_folder), rel="CustomerA/Stale", well_name="Stale Well",
        files=[
            FileFeatures(path=fs1, folder=str(stale_folder), size_bytes=1, sig="a",
                         verdict="likely_dfit"),
            FileFeatures(path=fs2, folder=str(stale_folder), size_bytes=1, sig="b",
                         verdict="likely_dfit"),
        ],
    )

    fresh_folder = root / "CustomerA" / "Fresh"
    ff1 = _write(fresh_folder / "c.dbs", b"3")
    fresh_scan = FolderScan(
        folder=str(fresh_folder), rel="CustomerA/Fresh", well_name="Fresh Well",
        files=[FileFeatures(path=ff1, folder=str(fresh_folder), size_bytes=1, sig="c",
                             verdict="likely_dfit")],
    )

    ledger = Ledger.load(str(root))
    ledger.set(stale_scan.rel, [fs1], "decided", files_sig=group_files_sig(["a"]))  # stale
    _decide(ledger, fresh_scan, [ff1])  # current

    warnings = apply_mod.plan_warnings([stale_scan, fresh_scan], ledger)

    assert len(warnings) == 1
    assert warnings[0].rel == "CustomerA/Stale"
    assert warnings[0].category == "stale_decision"
    assert "stale decision, re-review required" in warnings[0].message


def test_plan_warnings_decision_referencing_unknown_group(tmp_path):
    """A ledger `rel` with a real decision that matches no scan at all in the current scan list
    (e.g. a re-scan where the group disappeared entirely) is reported as `"unknown_group"`, not
    silently ignored."""
    ledger = Ledger.load(str(tmp_path))
    ledger.decisions["CustomerA/Gone"] = FolderDecision(
        rel="CustomerA/Gone", keeps=["/data/a.dbs"], status="decided", files_sig="somehash",
    )

    warnings = apply_mod.plan_warnings([], ledger)

    assert len(warnings) == 1
    assert warnings[0].rel == "CustomerA/Gone"
    assert warnings[0].category == "unknown_group"
    assert "decision references unknown group" in warnings[0].message


def test_plan_warnings_undecided_and_unsure_produce_no_warning(tmp_path):
    """An undecided or unsure folder is reported elsewhere (as "skipped (undecided/unsure)"),
    never by `plan_warnings` -- it has nothing stale or ambiguous to say about a folder nobody
    has reviewed yet."""
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    ledger.set(scan.rel, [f1], "unsure")

    assert apply_mod.plan_warnings([scan], ledger) == []


# --------------------------------------------------------------------------------------------------
# FIX 3: a group whose subtree holds more than one distinct well (n_wells != 1) is never planned
# as keep/quarantine moves -- it is left untouched and flagged with a warning line instead.
# --------------------------------------------------------------------------------------------------
def test_plan_moves_ambiguous_well_produces_zero_moves(monkeypatch, tmp_path):
    """MUTATION: remove the `n_wells != 1` guard in `_exclusion_reason` -> this test fails (the
    ambiguous group's files get planned as keep/quarantine moves instead of being left alone)."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "Bijou_1A_and_1B"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.dbs", b"2")
    scan = FolderScan(
        folder=str(folder), rel="Bijou_1A_and_1B", n_wells=2,
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="likely_dfit"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="likely_dfit"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])  # even a CURRENT, fingerprint-matching decision must not be planned

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)
    assert moves == []
    # Nothing on disk was touched, let alone moved or quarantined.
    assert os.path.exists(f1)
    assert os.path.exists(f2)


def test_plan_warnings_ambiguous_well_reports_n_wells(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "Bijou_1A_and_1B"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(
        folder=str(folder), rel="Bijou_1A_and_1B", n_wells=3,
        files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                             verdict="likely_dfit")],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    warnings = apply_mod.plan_warnings([scan], ledger)

    assert len(warnings) == 1
    assert warnings[0].category == "ambiguous_well"
    assert warnings[0].rel == scan.rel
    assert "n_wells=3" in warnings[0].message
    assert "manual handling required" in warnings[0].message


def test_plan_warnings_ambiguous_well_checked_before_fingerprint(monkeypatch, tmp_path):
    """An ambiguous well is reported as such even when its decision is ALSO stale -- the
    ambiguous-well guard is checked first and unconditionally, per `_exclusion_reason`'s
    docstring, so a folder never gets both warnings or the wrong one."""
    root = tmp_path
    folder = root / "Bijou_1A_and_1B"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(
        folder=str(folder), rel="Bijou_1A_and_1B", n_wells=2,
        files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                             verdict="likely_dfit")],
    )
    ledger = Ledger.load(str(root))
    ledger.set(scan.rel, [f1], "decided")  # legacy AND ambiguous

    warnings = apply_mod.plan_warnings([scan], ledger)
    assert len(warnings) == 1
    assert warnings[0].category == "ambiguous_well"


# --------------------------------------------------------------------------------------------------
# FIX 5 (third review round): pin the "none"-folder provenance behavior as deliberate, not a gap.
# --------------------------------------------------------------------------------------------------
def test_execute_none_folder_creates_no_destination_dir_and_no_provenance(monkeypatch, tmp_path):
    """A folder decided `"none"` has an empty `keeps`, so no `Move` targets its would-be well
    folder at all -- every file goes to `_quarantine` instead. Even when a caller (mirroring
    `_cmd_apply`, which builds one `provenance_for` record per reviewed folder regardless of
    decision) supplies a provenance record keyed by that folder's `destination_dir`, `execute`
    must never create that directory and must never write a `_provenance.json` into it --
    `_finalize_provenance` is simply never invoked for a directory no move's `dst` ever
    resolves to. Nothing is lost (both files land in `_quarantine`); the well folder itself must
    just never come into existence."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.csv", b"2")
    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="flat"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="flat"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [], "none")

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    assert all(m.kind == "quarantine" for m in moves)

    # A caller building a provenance record for this folder regardless of its decision (as
    # `_cmd_apply` does) -- the would-be well folder still gets a `provenance_for` record even
    # though nothing will ever be delivered there.
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    provenance = {dest_dir: [apply_mod.provenance_for(scan, [], basin, basin_source, well)]}

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    assert performed == 2  # both files quarantined
    assert failed == []
    assert prov_failed == []  # never attempted, so never reported as a failure either
    assert not os.path.exists(dest_dir)  # the well folder itself was never created
    assert not os.path.exists(os.path.join(dest_dir, "_provenance.json"))
    assert os.path.exists(os.path.join(root, "_quarantine", "CustomerA", "Well1", "a.dbs"))
    assert os.path.exists(os.path.join(root, "_quarantine", "CustomerA", "Well1", "b.csv"))


def test_plan_moves_existing_destination_file_is_skipped(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = root / "out"
    # well_name is blank, so the destination well folder falls back to the folder's basename
    # ("Well1") -- pre-create that destination file so plan_moves finds it already occupied.
    dest_dir = out_root / "TestBasin" / "Well1"
    _write(dest_dir / "a.dbs", b"already here")

    moves = apply_mod.plan_moves(str(root), str(out_root), [scan], ledger)

    assert len(moves) == 1
    assert moves[0].skipped == "destination exists"
    assert moves[0].dst == str(dest_dir / "a.dbs")


def test_plan_moves_same_basename_different_folders_disambiguates_by_folder_name(
        monkeypatch, tmp_path):
    """Real case: `Lucero Energy\\Tahu 2TF2H 1st DFIT` and `...\\Tahu 2TF2H 2nd DFIT` are two
    separate DFIT tests on the same well, both resolving to one well destination, and both
    happen to keep a file with the same basename. The second must be disambiguated by prefixing
    its source folder's basename -- never skipped, never overwritten."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder1 = root / "Lucero Energy" / "Tahu 2TF2H 1st DFIT"
    folder2 = root / "Lucero Energy" / "Tahu 2TF2H 2nd DFIT"
    f1 = _write(folder1 / "pumping.csv", b"1")
    f2 = _write(folder2 / "pumping.csv", b"2")

    scan1 = FolderScan(folder=str(folder1), rel="Lucero Energy/Tahu 2TF2H 1st DFIT",
                        well_name="Tahu 2TF2H",
                        files=[FileFeatures(path=f1, folder=str(folder1), size_bytes=1, sig="a",
                                             verdict="likely_dfit")])
    scan2 = FolderScan(folder=str(folder2), rel="Lucero Energy/Tahu 2TF2H 2nd DFIT",
                        well_name="Tahu 2TF2H",
                        files=[FileFeatures(path=f2, folder=str(folder2), size_bytes=1, sig="b",
                                             verdict="likely_dfit")])

    ledger = Ledger.load(str(root))
    _decide(ledger, scan1, [f1])
    _decide(ledger, scan2, [f2])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan1, scan2], ledger)

    assert len(moves) == 2
    assert all(m.skipped == "" for m in moves)
    by_src = {m.src: m.dst for m in moves}
    dest_dir = os.path.join(out_root, "TestBasin", "Tahu 2TF2H")
    assert by_src[f1] == os.path.join(dest_dir, "pumping.csv")
    assert by_src[f2] == os.path.join(dest_dir, "Tahu 2TF2H 2nd DFIT - pumping.csv")


def test_plan_moves_disambiguated_name_also_taken_gets_numeric_suffix(monkeypatch, tmp_path):
    """A third folder that happens to share its basename with the second folder above forces the
    disambiguated name to collide too -- covered by a " (2)" numeric suffix, still never
    skipped."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder1 = root / "Lucero Energy" / "Tahu 2TF2H 1st DFIT"
    folder2 = root / "Lucero Energy" / "Tahu 2TF2H 2nd DFIT"
    folder3 = root / "OtherCustomer" / "Tahu 2TF2H 2nd DFIT"  # same basename as folder2
    f1 = _write(folder1 / "pumping.csv", b"1")
    f2 = _write(folder2 / "pumping.csv", b"2")
    f3 = _write(folder3 / "pumping.csv", b"3")

    scans = [
        FolderScan(folder=str(folder1), rel="Lucero Energy/Tahu 2TF2H 1st DFIT",
                   well_name="Tahu 2TF2H",
                   files=[FileFeatures(path=f1, folder=str(folder1), size_bytes=1, sig="a",
                                        verdict="likely_dfit")]),
        FolderScan(folder=str(folder2), rel="Lucero Energy/Tahu 2TF2H 2nd DFIT",
                   well_name="Tahu 2TF2H",
                   files=[FileFeatures(path=f2, folder=str(folder2), size_bytes=1, sig="b",
                                        verdict="likely_dfit")]),
        FolderScan(folder=str(folder3), rel="OtherCustomer/Tahu 2TF2H 2nd DFIT",
                   well_name="Tahu 2TF2H",
                   files=[FileFeatures(path=f3, folder=str(folder3), size_bytes=1, sig="c",
                                        verdict="likely_dfit")]),
    ]
    ledger = Ledger.load(str(root))
    for scan, f in zip(scans, (f1, f2, f3)):
        _decide(ledger, scan, [f])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, scans, ledger)

    assert len(moves) == 3
    assert all(m.skipped == "" for m in moves)
    by_src = {m.src: m.dst for m in moves}
    dest_dir = os.path.join(out_root, "TestBasin", "Tahu 2TF2H")
    assert by_src[f1] == os.path.join(dest_dir, "pumping.csv")
    assert by_src[f2] == os.path.join(dest_dir, "Tahu 2TF2H 2nd DFIT - pumping.csv")
    assert by_src[f3] == os.path.join(dest_dir, "Tahu 2TF2H 2nd DFIT - pumping (2).csv")
    # every planned dst is unique -- nothing silently overwrites anything else
    assert len(set(by_src.values())) == 3


def test_plan_moves_identical_questionnaire_source_emitted_once(monkeypatch, tmp_path):
    """Two folders under one well can share the exact same questionnaire file. That must not be
    planned as a collision (disambiguated) or duplicated -- it's the same (src, dst) pair, so it
    is emitted once."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder1 = root / "Cust" / "Well 1st"
    folder2 = root / "Cust" / "Well 2nd"
    shared_quest = _write(root / "Cust" / "questionnaire.xlsx", b"q")
    f1 = _write(folder1 / "a.dbs", b"1")
    f2 = _write(folder2 / "b.dbs", b"2")

    scan1 = FolderScan(folder=str(folder1), rel="Cust/Well 1st", well_name="Well",
                        questionnaire_path=shared_quest,
                        files=[FileFeatures(path=f1, folder=str(folder1), size_bytes=1, sig="a",
                                             verdict="likely_dfit")])
    scan2 = FolderScan(folder=str(folder2), rel="Cust/Well 2nd", well_name="Well",
                        questionnaire_path=shared_quest,
                        files=[FileFeatures(path=f2, folder=str(folder2), size_bytes=1, sig="b",
                                             verdict="likely_dfit")])

    ledger = Ledger.load(str(root))
    _decide(ledger, scan1, [f1])
    _decide(ledger, scan2, [f2])

    moves = apply_mod.plan_moves(str(root), str(root / "out"), [scan1, scan2], ledger)
    quest_moves = [m for m in moves if m.kind == "questionnaire"]

    assert len(quest_moves) == 1
    assert quest_moves[0].src == shared_quest
    assert quest_moves[0].skipped == ""


def test_plan_moves_is_dry_run_no_filesystem_changes(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    keeper = _write(folder / "keeper.dbs", b"keep")
    other = _write(folder / "other.csv", b"other")

    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=keeper, folder=str(folder), size_bytes=4, sig="a",
                         verdict="likely_dfit"),
            FileFeatures(path=other, folder=str(folder), size_bytes=5, sig="b",
                         verdict="too_short"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [keeper])  # writes decisions.json -- do this before snapshotting

    def _snapshot(base):
        out = {}
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                p = os.path.join(dirpath, name)
                out[p] = os.path.getsize(p)
        return out

    before = _snapshot(str(root))
    apply_mod.plan_moves(str(root), str(root / "out"), [scan], ledger)
    after = _snapshot(str(root))

    assert before == after


# --------------------------------------------------------------------------------------------------
# execute
# --------------------------------------------------------------------------------------------------
def test_execute_performs_moves_writes_provenance_and_never_deletes(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch, basin="TestBasin", source="customer")
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    keeper = _write(folder / "keeper.dbs", b"keep")
    other = _write(folder / "other.csv", b"other")
    quest = _write(folder / "questionnaire.xlsx", b"quest")

    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        questionnaire_path=quest,
        files=[
            FileFeatures(path=keeper, folder=str(folder), size_bytes=4, sig="a",
                         verdict="likely_dfit"),
            FileFeatures(path=other, folder=str(folder), size_bytes=5, sig="b",
                         verdict="too_short"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [keeper])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    # provenance is keyed by dest_dir -> a LIST of per-source-folder records (FIX 2), even when
    # (as here) only one folder contributes to this destination.
    provenance = {
        dest_dir: [apply_mod.provenance_for(scan, [keeper], basin, basin_source, well)],
    }

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    assert performed == 3  # keep + questionnaire + quarantine
    assert skipped == 0
    assert failed == []
    assert prov_failed == []

    dest_keeper = os.path.join(dest_dir, "keeper.dbs")
    assert os.path.exists(dest_keeper)
    assert not os.path.exists(keeper)  # moved: original gone

    dest_quest = os.path.join(dest_dir, "questionnaire.xlsx")
    assert os.path.exists(dest_quest)
    assert os.path.exists(quest)  # copied: original still present

    quarantine_path = os.path.join(str(root), "_quarantine", "CustomerA", "Well1", "other.csv")
    assert os.path.exists(quarantine_path)
    assert not os.path.exists(other)  # moved: original gone

    prov_path = os.path.join(dest_dir, "_provenance.json")
    assert os.path.exists(prov_path)
    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["sources"]) == 1
    record = data["sources"][0]
    # FIX 3: "keeps" carries the actual delivery outcome, not the ledger's bare intent.
    assert record["keeps"] == [{"path": keeper, "status": "delivered", "dst": dest_keeper}]
    assert record["basin_source"] == "customer"
    assert record["basin"] == "TestBasin"

    leftover_tmp = [p for p in _walk_all(root) if p.endswith(".tmp")]
    assert leftover_tmp == []


def _walk_all(base):
    for dirpath, _dirs, files in os.walk(str(base)):
        for name in files:
            yield os.path.join(dirpath, name)


def test_execute_skips_moves_marked_skipped(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = str(root / "out")
    dest_dir = os.path.join(out_root, "TestBasin", "Well One")
    _write(tmp_path / "out" / "TestBasin" / "Well One" / "a.dbs", b"already here")

    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    assert moves[0].skipped == "destination exists"

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, {})

    assert performed == 0
    assert skipped == 1
    assert failed == []
    assert prov_failed == []
    assert os.path.exists(f1)  # never touched, let alone deleted
    with open(os.path.join(dest_dir, "a.dbs"), "rb") as fh:
        assert fh.read() == b"already here"  # never overwritten


# --------------------------------------------------------------------------------------------------
# FIX 3 (third review round): two regression guards for a destination whose ONLY keep move is
# skipped -- the prior test above passes `provenance={}`, so it cannot catch either bug.
# --------------------------------------------------------------------------------------------------
def test_execute_skipped_only_destination_still_gets_provenance_written(monkeypatch, tmp_path):
    """B1: the per-directory countdown (`remaining`) must include skipped moves, not just
    performed ones -- mutation: `Counter(os.path.dirname(mv.dst) for mv in moves if not
    mv.skipped)`. A destination whose only keep move is `skipped="destination exists"` would
    then decrement the countdown for a directory that was never counted in the first place
    (`remaining[dst_dir] -= 1` on a `Counter` with no entry for that key goes to -1, never 0),
    so `_finalize_provenance` never runs and no `_provenance.json` is written at all -- even
    though a provenance record was supplied for it."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = str(root / "out")
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    _write(tmp_path / "out" / "TestBasin" / "Well One" / "a.dbs", b"already here")

    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    assert len(moves) == 1
    assert moves[0].skipped == "destination exists"

    provenance = {dest_dir: [apply_mod.provenance_for(scan, [f1], basin, basin_source, well)]}
    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    assert performed == 0
    assert skipped == 1
    assert failed == []
    assert prov_failed == []  # never even attempted a write under the mutation -- no failure, no file
    assert os.path.exists(os.path.join(dest_dir, "_provenance.json"))


def test_execute_skipped_keep_status_is_skipped_not_unknown(monkeypatch, tmp_path):
    """B3: a skipped `"keep"` move must record `outcome_by_src[mv.src] = {"status": "skipped",
    "reason": ...}` -- mutation: delete that block, and the keep silently degrades to
    `{"path": ..., "status": "unknown"}` via `_annotate_keeps`'s fallback."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = str(root / "out")
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    _write(tmp_path / "out" / "TestBasin" / "Well One" / "a.dbs", b"already here")

    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    provenance = {dest_dir: [apply_mod.provenance_for(scan, [f1], basin, basin_source, well)]}
    apply_mod.execute(moves, provenance)

    with open(os.path.join(dest_dir, "_provenance.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    keep = data["sources"][0]["keeps"][0]
    assert keep["status"] == "skipped"
    assert keep["reason"] == "destination exists"


# --------------------------------------------------------------------------------------------------
# FIX 1: a per-move failure must not abandon the rest of the plan, and provenance must still be
# written for every destination directory execute actually touched.
# --------------------------------------------------------------------------------------------------
def test_execute_continues_past_failing_move_and_writes_provenance_for_touched_dirs(
        monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.dbs", b"2")
    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="likely_dfit"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="likely_dfit"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1, f2])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    provenance = {
        dest_dir: [apply_mod.provenance_for(scan, [f1, f2], basin, basin_source, well)],
    }

    # Simulate a move raising -- e.g. a file another process (Excel/OneDrive/Defender) holds
    # open -- for f1 specifically, while f2's move goes through normally.
    real_move = apply_mod.shutil.move

    def flaky_move(src, dst):
        if src == f1:
            raise PermissionError("simulated: file in use")
        return real_move(src, dst)

    monkeypatch.setattr(apply_mod.shutil, "move", flaky_move)

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    assert performed == 1  # f2's move still happened
    assert skipped == 0
    assert len(failed) == 1
    failed_move, err = failed[0]
    assert failed_move.src == f1
    assert "PermissionError" in err
    assert os.path.exists(f1)  # the failed move never touched its source
    assert os.path.exists(os.path.join(dest_dir, "b.dbs"))  # the surviving move completed
    assert prov_failed == []  # the provenance write itself succeeded

    # The provenance record for the touched directory exists despite the move failure, and its
    # "keeps" reflect what ACTUALLY happened (FIX 3), not the ledger's bare intent: f1 failed to
    # move (still sitting at its source, with the error recorded) while f2 was delivered.
    prov_path = os.path.join(dest_dir, "_provenance.json")
    assert os.path.exists(prov_path)
    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)
    keeps = data["sources"][0]["keeps"]
    keeps_by_path = {k["path"]: k for k in keeps}
    assert keeps_by_path[f1]["status"] == "failed"
    assert "PermissionError" in keeps_by_path[f1]["error"]
    assert keeps_by_path[f2]["status"] == "delivered"
    assert keeps_by_path[f2]["dst"] == os.path.join(dest_dir, "b.dbs")


# --------------------------------------------------------------------------------------------------
# FIX 2: two decided folders resolving to one destination directory must not collapse to one
# source's provenance -- both contributing folders' records must survive.
# --------------------------------------------------------------------------------------------------
def test_provenance_for_two_folders_sharing_a_dest_dir_names_both_sources(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder1 = root / "Cust" / "Tahu 1st DFIT"
    folder2 = root / "Cust" / "Tahu 2nd DFIT"
    f1 = _write(folder1 / "a.dbs", b"1")
    f2 = _write(folder2 / "b.dbs", b"2")

    scan1 = FolderScan(folder=str(folder1), rel="Cust/Tahu 1st DFIT", well_name="Tahu 2TF2H",
                        files=[FileFeatures(path=f1, folder=str(folder1), size_bytes=1, sig="a",
                                             verdict="likely_dfit")])
    scan2 = FolderScan(folder=str(folder2), rel="Cust/Tahu 2nd DFIT", well_name="Tahu 2TF2H",
                        files=[FileFeatures(path=f2, folder=str(folder2), size_bytes=1, sig="b",
                                             verdict="likely_dfit")])

    ledger = Ledger.load(str(root))
    _decide(ledger, scan1, [f1])
    _decide(ledger, scan2, [f2])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan1, scan2], ledger)

    dest_dir1, basin1, source1 = apply_mod.destination_dir(scan1, out_root, str(root))
    dest_dir2, basin2, source2 = apply_mod.destination_dir(scan2, out_root, str(root))
    assert dest_dir1 == dest_dir2  # same well name -> same destination directory

    well = apply_mod.well_folder_name(scan1)
    provenance: dict[str, list] = {}
    for scan, keeps, basin, basin_source in (
        (scan1, [f1], basin1, source1),
        (scan2, [f2], basin2, source2),
    ):
        provenance.setdefault(dest_dir1, []).append(
            apply_mod.provenance_for(scan, keeps, basin, basin_source, well))

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)
    assert failed == []
    assert prov_failed == []

    prov_path = os.path.join(dest_dir1, "_provenance.json")
    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)

    rels = {rec["rel"] for rec in data["sources"]}
    assert rels == {"Cust/Tahu 1st DFIT", "Cust/Tahu 2nd DFIT"}
    keeps_by_rel = {rec["rel"]: rec["keeps"] for rec in data["sources"]}
    dest_f1 = os.path.join(dest_dir1, "a.dbs")
    dest_f2 = os.path.join(dest_dir1, "b.dbs")
    assert keeps_by_rel["Cust/Tahu 1st DFIT"] == [{"path": f1, "status": "delivered", "dst": dest_f1}]
    assert keeps_by_rel["Cust/Tahu 2nd DFIT"] == [{"path": f2, "status": "delivered", "dst": dest_f2}]


# --------------------------------------------------------------------------------------------------
# FIX 1: a directory whose provenance write itself fails gets retried once more before giving up,
# and its dst_dir is reported in the returned `prov_failed` list -- never a silent gap.
# --------------------------------------------------------------------------------------------------
def test_execute_retries_provenance_write_once_and_succeeds(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    provenance = {dest_dir: [apply_mod.provenance_for(scan, [f1], basin, basin_source, well)]}

    real_write = apply_mod._write_provenance_atomic
    calls = []

    def flaky_write(dest, data):
        calls.append(dest)
        if len(calls) == 1:
            raise PermissionError("simulated: transient lock")
        return real_write(dest, data)

    monkeypatch.setattr(apply_mod, "_write_provenance_atomic", flaky_write)

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    assert performed == 1
    assert failed == []
    assert prov_failed == []  # the retry succeeded
    assert len(calls) == 2  # one failed attempt, one retry
    assert os.path.exists(os.path.join(dest_dir, "_provenance.json"))


def test_execute_persistent_provenance_write_failure_is_reported_not_silent(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    f2 = _write(folder / "b.dbs", b"2")
    scan = FolderScan(
        folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
        files=[
            FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a", verdict="likely_dfit"),
            FileFeatures(path=f2, folder=str(folder), size_bytes=1, sig="b", verdict="likely_dfit"),
        ],
    )
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1, f2])

    out_root = str(root / "out")
    moves = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)
    provenance = {dest_dir: [apply_mod.provenance_for(scan, [f1, f2], basin, basin_source, well)]}

    def always_fails(dest, data):
        raise PermissionError("simulated: disk full")

    monkeypatch.setattr(apply_mod, "_write_provenance_atomic", always_fails)

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)

    # Both files were still delivered -- a doomed provenance write must not abandon the plan.
    assert performed == 2
    assert failed == []
    assert not os.path.exists(os.path.join(dest_dir, "_provenance.json"))
    assert len(prov_failed) == 1
    reported_dir, err = prov_failed[0]
    assert reported_dir == dest_dir
    assert "PermissionError" in err


# --------------------------------------------------------------------------------------------------
# FIX 4: a second `execute` run into a destination directory a prior run already populated must
# merge, not overwrite -- both runs' source records must survive.
# --------------------------------------------------------------------------------------------------
def test_execute_second_run_into_same_dest_dir_preserves_first_runs_provenance(monkeypatch, tmp_path):
    _fixed_basin(monkeypatch)
    root = tmp_path

    # Run 1: folder A's keeper lands in the well destination.
    folder_a = root / "Cust" / "Well A DFIT"
    fa = _write(folder_a / "a.dbs", b"a")
    scan_a = FolderScan(folder=str(folder_a), rel="Cust/Well A DFIT", well_name="Well One",
                        files=[FileFeatures(path=fa, folder=str(folder_a), size_bytes=1, sig="a",
                                             verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan_a, [fa])

    out_root = str(root / "out")
    moves_a = apply_mod.plan_moves(str(root), out_root, [scan_a], ledger)
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan_a, out_root, str(root))
    well = apply_mod.well_folder_name(scan_a)
    provenance_a = {
        dest_dir: [apply_mod.provenance_for(scan_a, [fa], basin, basin_source, well)],
    }
    performed, skipped, failed, prov_failed = apply_mod.execute(moves_a, provenance_a)
    assert failed == [] and prov_failed == []

    # Run 2 (a later, separate `apply --commit` invocation): folder B is reviewed afterward and
    # resolves to the SAME well destination.
    folder_b = root / "Cust" / "Well B DFIT"
    fb = _write(folder_b / "b.dbs", b"b")
    scan_b = FolderScan(folder=str(folder_b), rel="Cust/Well B DFIT", well_name="Well One",
                        files=[FileFeatures(path=fb, folder=str(folder_b), size_bytes=1, sig="b",
                                             verdict="likely_dfit")])
    _decide(ledger, scan_b, [fb])
    moves_b = apply_mod.plan_moves(str(root), out_root, [scan_b], ledger)
    provenance_b = {
        dest_dir: [apply_mod.provenance_for(scan_b, [fb], basin, basin_source, well)],
    }
    performed, skipped, failed, prov_failed = apply_mod.execute(moves_b, provenance_b)
    assert failed == [] and prov_failed == []

    prov_path = os.path.join(dest_dir, "_provenance.json")
    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)

    rels = {rec["rel"] for rec in data["sources"]}
    assert rels == {"Cust/Well A DFIT", "Cust/Well B DFIT"}  # both runs' records survive


# --------------------------------------------------------------------------------------------------
# FIX 2 (third review round): a stale `"failed"` provenance record for a `rel` must not permanently
# outlive a later run that successfully retries and delivers that same file.
# --------------------------------------------------------------------------------------------------
def test_execute_retry_after_failed_move_upgrades_provenance_to_delivered(monkeypatch, tmp_path):
    """Run 1's move raises (a locked file); `_provenance.json` correctly records `"failed"` for
    it. Run 2 (a later, separate `apply --commit` invocation, same `rel`, same ledger decision --
    the retry workflow `write_failures_file` exists to support) succeeds. The OLD dedup-by-`rel`
    rule left the on-disk record as-is once a `rel` was already present, so it stayed `"failed"`
    forever even though the file is now sitting at its destination. The fix must upgrade it to
    `"delivered"` with the real destination path, and `sources` must still carry exactly one
    record for this `rel` (not two, one stale and one fresh)."""
    _fixed_basin(monkeypatch)
    root = tmp_path
    folder = root / "CustomerA" / "Well1"
    f1 = _write(folder / "a.dbs", b"1")
    scan = FolderScan(folder=str(folder), rel="CustomerA/Well1", well_name="Well One",
                       files=[FileFeatures(path=f1, folder=str(folder), size_bytes=1, sig="a",
                                            verdict="likely_dfit")])
    ledger = Ledger.load(str(root))
    _decide(ledger, scan, [f1])

    out_root = str(root / "out")
    dest_dir, basin, basin_source = apply_mod.destination_dir(scan, out_root, str(root))
    well = apply_mod.well_folder_name(scan)

    # Run 1: the move raises (simulated locked file); f1 never actually moves.
    moves_1 = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    provenance_1 = {dest_dir: [apply_mod.provenance_for(scan, [f1], basin, basin_source, well)]}

    real_move = apply_mod.shutil.move

    def flaky_move(src, dst):
        if src == f1:
            raise PermissionError("simulated: file in use")
        return real_move(src, dst)

    monkeypatch.setattr(apply_mod.shutil, "move", flaky_move)
    performed, skipped, failed, prov_failed = apply_mod.execute(moves_1, provenance_1)
    assert performed == 0
    assert len(failed) == 1
    assert os.path.exists(f1)  # never moved -- still sitting at its source

    prov_path = os.path.join(dest_dir, "_provenance.json")
    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["sources"][0]["keeps"][0]["status"] == "failed"

    # Run 2: a later, separate `apply --commit` invocation retries the same reviewed decision.
    # The move now succeeds (no more flaky_move).
    monkeypatch.setattr(apply_mod.shutil, "move", real_move)
    moves_2 = apply_mod.plan_moves(str(root), out_root, [scan], ledger)
    provenance_2 = {dest_dir: [apply_mod.provenance_for(scan, [f1], basin, basin_source, well)]}
    performed, skipped, failed, prov_failed = apply_mod.execute(moves_2, provenance_2)
    assert performed == 1
    assert failed == []
    assert not os.path.exists(f1)  # delivered this time
    dest_f1 = os.path.join(dest_dir, "a.dbs")
    assert os.path.exists(dest_f1)

    with open(prov_path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Exactly one record for this rel -- not a stale one plus a fresh one.
    matching = [rec for rec in data["sources"] if rec["rel"] == "CustomerA/Well1"]
    assert len(matching) == 1
    keeps = matching[0]["keeps"]
    assert len(keeps) == 1
    assert keeps[0]["path"] == f1
    assert keeps[0]["status"] == "delivered"
    assert keeps[0]["dst"] == dest_f1


# --------------------------------------------------------------------------------------------------
# FIX 3: the failed list is also persisted to disk, not just printed, so a partial migration
# leaves a durable record.
# --------------------------------------------------------------------------------------------------
def test_write_failures_file_persists_moves_and_provenance_failures(tmp_path):
    root = str(tmp_path)
    move = apply_mod.Move(src="C:\\src\\a.dbs", dst="C:\\out\\Basin\\Well\\a.dbs", kind="keep")
    failed = [(move, "PermissionError: simulated")]
    prov_failed = [("C:\\out\\Basin\\Well", "PermissionError: disk full")]

    path = apply_mod.write_failures_file(root, failed, prov_failed)

    assert path == os.path.join(root, "_triage", "apply_failures.json")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert len(data["moves_failed"]) == 1
    assert data["moves_failed"][0]["src"] == move.src
    assert data["moves_failed"][0]["dst"] == move.dst
    assert data["moves_failed"][0]["error"] == "PermissionError: simulated"
    assert len(data["provenance_failed"]) == 1
    assert data["provenance_failed"][0]["dest_dir"] == "C:\\out\\Basin\\Well"


def test_write_failures_file_returns_none_and_writes_nothing_when_no_failures(tmp_path):
    root = str(tmp_path)
    path = apply_mod.write_failures_file(root, [], [])
    assert path is None
    assert not os.path.exists(os.path.join(root, "_triage", "apply_failures.json"))


# --------------------------------------------------------------------------------------------------
# validate_out_not_inside_root
# --------------------------------------------------------------------------------------------------
def test_validate_out_not_inside_root_rejects_out_equal_to_root(tmp_path):
    with pytest.raises(ValueError):
        apply_mod.validate_out_not_inside_root(str(tmp_path), str(tmp_path))


def test_validate_out_not_inside_root_rejects_out_nested_in_root(tmp_path):
    with pytest.raises(ValueError):
        apply_mod.validate_out_not_inside_root(str(tmp_path), str(tmp_path / "Organized"))


def test_validate_out_not_inside_root_allows_sibling_out(tmp_path):
    apply_mod.validate_out_not_inside_root(str(tmp_path / "root"), str(tmp_path / "out"))
