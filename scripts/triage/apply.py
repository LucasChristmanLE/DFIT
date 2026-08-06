"""Phase 3 of DFIT triage: turn a reviewed ledger into a move/copy plan, then execute it.

Pure path/plan logic plus filesystem execution -- no tkinter here (`review_app.py` is the only
file in this package allowed to import it).

`plan_moves` is dry: it never touches disk beyond `os.path.exists` checks used to detect a
pre-existing destination or to find a free disambiguated name. `execute` is the only function
that moves, copies, or writes anything, and it never deletes: keeper files and quarantine
candidates are `shutil.move`d (so nothing doubles disk usage), the questionnaire is
`shutil.copy2`d (one questionnaire can serve several folders), and each destination well folder
gets a `_provenance.json` written atomically -- the same temp-file-plus-`os.replace` contract as
`dfit_tool/store.py:save_picks_for`.

Two different kinds of "the target is already spoken for" are handled differently:

- A `dst` that already exists **on disk before this run** is left alone: the `Move` is marked
  `skipped="destination exists"` and `execute` never touches it. This is the one case where a
  keeper is silently not delivered -- it is treated as evidence a previous `apply --commit` (or a
  hand-placed file) already put something there.
- A `dst` that two different `src` files **within this same plan** would both land on (same
  well, same original basename -- e.g. two DFIT tests on one well each named their pumping-stage
  file the same thing) is never a skip and never an overwrite: the second src is disambiguated by
  prefixing its source folder's basename onto the filename, and if that is also taken, a
  ``" (2)"``, ``" (3)"``, ... suffix is added before the extension. An identical `(src, dst)` pair
  seen twice (the same questionnaire file feeding two folders under one well) is emitted only
  once rather than disambiguated or duplicated.

Because two *different* source folders can resolve to the same destination well folder (same
well name, different DFIT-test folders -- see `well_folder_name`'s docstring and the "Tahu"
example in the tests), a destination directory's `_provenance.json` is not one flat record: it
is `{"sources": [record, record, ...]}`, one record per contributing source folder, each carrying
its own `folder`/`rel`/`keeps`/`files`/basin decision (see `provenance_for`). `execute` finalizes
(writes) each destination directory's provenance once every move destined for that directory has
been processed -- not on first touch, before anything about that directory's moves is known --
so the `keeps` written are the *actual* delivery outcome (`"delivered"`/`"failed"`/`"skipped"`,
not the ledger's bare intent) rather than a record of what was merely planned. A directory whose
provenance write itself fails is retried once more before being reported in `execute`'s returned
`prov_failed` list, so files delivered into a directory always come with a loud signal if no
provenance record could be produced for it (never a silent gap). Writing also merges with
whatever `_provenance.json` a *previous* `apply --commit` run already left in that directory
(grouped by source folder `rel`), so a second run into an already-populated well folder appends
rather than erasing the first run's record. A `rel` seen in both the prior run and this one
(e.g. retrying after a move failed last time) is merged at the per-kept-file level, not left
stale: `delivered` beats `failed`/`skipped`, which beat `unknown`, and a newer `failed` never
overwrites an older `delivered` -- see `_merge_provenance_sources`/`_merge_keep_records`.

Two more guards keep `plan_moves` from silently misfiling a group, beyond the plain undecided/
unsure skip -- see `_exclusion_reason`/`plan_warnings` for the details:

- A group whose subtree holds more than one distinct well (`FolderScan.n_wells != 1`) is never
  planned as keep/questionnaire/quarantine moves at all -- there is no single correct well to file
  it under, so its files are left exactly where they are.
- A decided group whose ledger decision was recorded against a DIFFERENT file set than the one
  `scan_folders` currently reports for it (a `files_sig` fingerprint mismatch, including a legacy
  decision from before that fingerprint existed) is likewise never planned -- the decision is
  stale and must not be silently applied to files no human ever reviewed in this shape.

Both are reported by `plan_warnings`, one line per excluded folder, so `dfit_triage.py` can print
and count them alongside `plan_moves`' own move list -- neither silent-skip reason goes
unreported to the human running `apply`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone

from triage import atomic, basins
from triage.features import FolderScan
from triage.ledger import Ledger, group_files_sig

# Characters Windows forbids in a path component, plus a trailing dot/space -- a questionnaire
# well name is free text and can carry any of these.
_FORBIDDEN_CHARS_RE = re.compile(r'[<>:"/\\|?*]')
_WS_RE = re.compile(r"\s+")


@dataclass
class Move:
    src: str
    dst: str
    kind: str          # "keep" | "questionnaire" | "quarantine"
    skipped: str = ""  # non-empty means this move will not be performed, and says why


def _sanitize_component(name: str) -> str:
    """Collapse whitespace runs to one space and strip the result; strip characters Windows
    forbids in a path component; strip any trailing dot or space. Returns `""` if nothing
    survives -- a questionnaire well name is free text and can be e.g. `"???"` or `"..."`,
    which reduce to nothing once sanitized."""
    name = (name or "").strip()
    name = _WS_RE.sub(" ", name).strip()
    name = _FORBIDDEN_CHARS_RE.sub("", name)
    name = name.rstrip(". ")
    return name


def well_folder_name(scan: FolderScan) -> str:
    """`scan.well_name` when it sanitizes to something non-blank, else the folder's basename,
    else a placeholder derived from `scan.rel`, else a fixed literal. Never returns `""`: an
    empty well folder name would make `destination_dir` return a path ending in a separator
    (e.g. `C:\\out\\DJ\\`), which does not match `os.path.dirname(mv.dst)` for the provenance
    lookup and, worse, would land keeper files directly in the basin folder instead of a well
    folder, mixing wells together (FIX 2). A blank or all-forbidden-characters `well_name`
    (`"???"`, `"..."`, `"   "`) is exactly the case this guards -- it falls through to the
    folder's own basename, which is itself sanitized in case the folder is *also* named
    something like `"???"`."""
    sanitized = _sanitize_component(scan.well_name)
    if sanitized:
        return sanitized
    sanitized = _sanitize_component(os.path.basename(scan.folder))
    if sanitized:
        return sanitized
    # Both the well name and the folder's own basename are unusable. Fall back to the scan's
    # `rel` (its full relative path, not just the leaf -- which usually equals the folder
    # basename just tried and so would add nothing), flattened to one component so it still
    # carries some identifying information (e.g. the customer segment) even if the leaf alone
    # is forbidden-characters-only.
    flattened_rel = re.sub(r"[\\/]+", "_", scan.rel or "")
    sanitized = _sanitize_component(flattened_rel)
    if sanitized:
        return sanitized
    return "_unnamed_well"


def _customer(rel: str) -> str:
    """The first path segment of a scan's `rel`, which -- like `store.TestEntry.test_id` -- may
    be forward-slash-separated regardless of platform, so split on either separator."""
    if not rel:
        return ""
    return re.split(r"[\\/]", rel)[0]


def validate_out_not_inside_root(root: str, out_root: str) -> None:
    """Raise `ValueError` if `out_root` is `root` itself, or nested inside it. Otherwise the new
    Basin/Well/ tree would land inside the very tree being migrated, so the next `scan`/`apply`
    over `root` would see (and could re-migrate) its own output."""
    root_abs = os.path.normcase(os.path.abspath(root))
    out_abs = os.path.normcase(os.path.abspath(out_root))
    if out_abs == root_abs or out_abs.startswith(root_abs + os.sep):
        raise ValueError(
            f"--out ({out_root!r}) must not be --root ({root!r}) or a subdirectory of it -- "
            f"the new tree would land inside the tree being migrated."
        )


def destination_dir(scan: FolderScan, out_root: str, root: str) -> tuple[str, str, str]:
    """Where a scan's keepers land: `<out_root>/<basin>/<well folder name>`. `root` is unused by
    the path math itself (the customer comes from `scan.rel`, already root-relative) but is kept
    in the signature per the contract, matching `plan_moves`'s parameter list."""
    customer = _customer(scan.rel)
    basin, basin_source = basins.basin_for(scan.formation, customer)
    well = well_folder_name(scan)
    dir_path = os.path.join(out_root, basin, well)
    return dir_path, basin, basin_source


def _disambiguate(dst: str, src: str, registry: dict[str, str]) -> str:
    """`dst` is already claimed by a different `src`. Prefix the source folder's basename onto
    the filename; if that is also taken (on disk, or claimed by yet another src), append
    " (2)", " (3)", ... before the extension until a free name is found. If the disambiguated
    name turns out to already be claimed by this exact `src`, reuse it (the same file reached
    this dst by two routes)."""
    dst_dir = os.path.dirname(dst)
    base, ext = os.path.splitext(os.path.basename(dst))
    folder_name = os.path.basename(os.path.dirname(src))
    prefixed = f"{folder_name} - {base}"
    candidate = os.path.join(dst_dir, f"{prefixed}{ext}")
    n = 2
    while True:
        claimant = registry.get(candidate)
        if claimant == src:
            return candidate
        if claimant is None and not os.path.exists(candidate):
            return candidate
        candidate = os.path.join(dst_dir, f"{prefixed} ({n}){ext}")
        n += 1


def _place(src: str, dst: str, kind: str, registry: dict[str, str], moves: list[Move]) -> None:
    """Append the `Move` for one (src, dst) pair to `moves`, resolving any collision first.

    A `dst` pre-existing on disk before this run is skipped outright. A `dst` already claimed
    within this plan by a *different* src is disambiguated (never skipped, never overwritten).
    A `dst` already claimed by this exact src is a no-op (the pair is emitted only once)."""
    if os.path.exists(dst):
        moves.append(Move(src=src, dst=dst, kind=kind, skipped="destination exists"))
        return
    claimant = registry.get(dst)
    if claimant == src:
        return  # identical (src, dst) already planned -- emit once
    if claimant is not None:
        dst = _disambiguate(dst, src, registry)
    registry[dst] = src
    moves.append(Move(src=src, dst=dst, kind=kind))


def _exclusion_reason(scan: FolderScan, decision) -> str | None:
    """Which of the two group-level planning guards excludes `scan` from being planned as
    keep/questionnaire/quarantine moves, given its ledger `decision` -- `None` if neither applies
    (the group is plannable normally). Shared by `plan_moves` (which just skips the group) and
    `plan_warnings` (which reports WHY, for the human running `apply`) so the two can never
    disagree about which groups are excluded.

    - `"ambiguous_well"` (this round's FIX 3): `scan.n_wells != 1` -- a group whose subtree holds
      more than one distinct well (or, for `n_wells == 0`, none at all -- reachable only under
      `require_questionnaire=False`, which `apply` never runs against) has no single correct well
      to file its keepers under; `well_folder_name`/`destination_dir` would otherwise silently
      pick one basin/well guess for a folder that might belong to several. Checked FIRST and
      unconditionally: an ambiguous well is excluded regardless of what its ledger decision says,
      fingerprint match or not -- there is no group identity to even check a fingerprint against.
    - `"stale_decision"` (this round's FIX 2): the decision's `files_sig` doesn't match the
      group's CURRENT fingerprint (`group_files_sig`) -- including a legacy decision with no
      `files_sig` at all, which can never match. The exact same staleness `review_app.py` treats
      as "resurface for review"; here it means never silently applying this decision to files no
      human reviewed in this shape."""
    if scan.n_wells != 1:
        return "ambiguous_well"
    current_sig = group_files_sig(f.sig for f in scan.files)
    if not decision.files_sig or decision.files_sig != current_sig:
        return "stale_decision"
    return None


def plan_moves(root: str, out_root: str, scans: list[FolderScan], ledger: Ledger) -> list[Move]:
    """The full move/copy plan for every reviewed folder.

    Folders whose ledger decision is undecided (`""`) or `"unsure"` are skipped entirely -- no
    `Move` is emitted for them at all, decided or not. `dfit_triage.py` reports those counts to
    the user; this function does not, since it has no notion of "skipped folder" beyond simply
    not visiting one.

    A decided folder is ALSO skipped -- no moves planned, files left exactly where they are,
    neither filed nor quarantined -- when `_exclusion_reason` finds it ambiguous (`n_wells != 1`)
    or its decision stale (a `files_sig` mismatch, including a legacy decision with none at all).
    `plan_warnings` is the human-readable explanation of every such exclusion; call it alongside
    this function so neither silent-skip reason goes unreported.
    """
    moves: list[Move] = []
    registry: dict[str, str] = {}  # dst -> the src that claimed it, across the whole plan

    for scan in scans:
        decision = ledger.get(scan.rel)
        if decision.status in ("", "unsure"):
            continue
        if _exclusion_reason(scan, decision) is not None:
            continue  # ambiguous well or stale/mismatched decision -- see plan_warnings

        dest_dir, _basin, _basin_source = destination_dir(scan, out_root, root)
        keeps = decision.keeps

        for src in keeps:
            dst = os.path.join(dest_dir, os.path.basename(src))
            _place(src, dst, "keep", registry, moves)

        if keeps and scan.questionnaire_path:
            dst = os.path.join(dest_dir, os.path.basename(scan.questionnaire_path))
            _place(scan.questionnaire_path, dst, "questionnaire", registry, moves)

        for feat in scan.files:
            if feat.path in keeps:
                continue
            rel_to_root = os.path.relpath(feat.path, root)
            dst = os.path.join(root, "_quarantine", rel_to_root)
            _place(feat.path, dst, "quarantine", registry, moves)

    return moves


@dataclass
class PlanWarning:
    rel: str
    category: str   # "ambiguous_well" | "stale_decision" | "unknown_group"
    message: str    # ready to print, prefixed with `rel` (except "unknown_group", see below)


def plan_warnings(scans: list[FolderScan], ledger: Ledger) -> list[PlanWarning]:
    """One `PlanWarning` per folder `plan_moves` silently excludes for a reason beyond plain
    undecided/unsure -- this round's FIX 3 (`scan.n_wells != 1`) and FIX 2 (a decided group whose
    current file fingerprint no longer matches the ledger's recorded one, or a legacy decision
    with no fingerprint at all). Also covers the case `_exclusion_reason` can't, because it has no
    matching `scan` to even check: a ledger `rel` with a real decision that matches no group in
    `scans` at all (`"unknown_group"`) -- e.g. a re-scan whose grouping semantics changed enough
    that a `rel` disappeared entirely, not just grew or shrank.

    `dfit_triage.py` prints these alongside `plan_moves`'s own move list and counts them into the
    dry-run/`--commit` summaries, so neither silent-skip reason goes unreported to the human
    running `apply`. Order: `scans` order first (ambiguous-well/stale-decision), then leftover
    ledger-only `rel`s (unknown-group), in `ledger.decisions`' own iteration order."""
    warnings: list[PlanWarning] = []
    scan_by_rel = {s.rel: s for s in scans}

    for scan in scans:
        decision = ledger.get(scan.rel)
        if decision.status in ("", "unsure"):
            continue
        reason = _exclusion_reason(scan, decision)
        if reason == "ambiguous_well":
            warnings.append(PlanWarning(
                rel=scan.rel, category="ambiguous_well",
                message=f"{scan.rel}: ambiguous well (n_wells={scan.n_wells}): "
                        f"manual handling required",
            ))
        elif reason == "stale_decision":
            warnings.append(PlanWarning(
                rel=scan.rel, category="stale_decision",
                message=f"{scan.rel}: stale decision, re-review required",
            ))

    for rel, decision in ledger.decisions.items():
        if decision.status in ("", "unsure"):
            continue
        if rel not in scan_by_rel:
            warnings.append(PlanWarning(
                rel=rel, category="unknown_group",
                message=f"{rel}: decision references unknown group",
            ))

    return warnings


def provenance_for(scan: FolderScan, keeps: list[str], basin: str, basin_source: str,
                    well: str) -> dict:
    """A JSON-able record of how one folder's keepers were INTENDED to end up at their
    destination: the original folder, every file's absolute path with its verdict and full
    feature set (each via `dataclasses.asdict`, which already carries `path` and `verdict`),
    the ledger's kept paths (still plain strings here -- `execute` replaces each with its actual
    delivery outcome before writing, see `_annotate_keeps` and FIX 3 in this module's
    docstring), the questionnaire path, the resolved well/basin and the basin's source, and an
    ISO timestamp of when this was computed.

    This is ONE source folder's record. A destination directory can be fed by more than one
    source folder (two DFIT-test folders on the same well); the caller collects one of these per
    contributing scan into a list, keyed by destination directory, and passes that to `execute`,
    which writes them out together as `{"sources": [...]}` -- see this module's docstring."""
    return {
        "folder": scan.folder,
        "rel": scan.rel,
        "well": well,
        "basin": basin,
        "basin_source": basin_source,
        "questionnaire_path": scan.questionnaire_path,
        "keeps": list(keeps),
        "files": [dataclasses.asdict(f) for f in scan.files],
        "triaged_at": datetime.now(timezone.utc).isoformat(),
    }


def _annotate_keeps(record: dict, outcome_by_src: dict[str, dict]) -> dict:
    """Return a shallow copy of one `provenance_for` record with its `"keeps"` field -- plain
    source paths -- replaced by each path's actual delivery outcome (FIX 3): `{"path":
    ..., "status": "delivered", "dst": ...}`, `{"path": ..., "status": "failed", "error": ...}`,
    or `{"path": ..., "status": "skipped", "reason": ...}` (the pre-existing-destination case).
    A keep path with no recorded outcome (should not happen -- every `keeps` entry corresponds
    to exactly one `Move` of kind `"keep"` built by `plan_moves`) gets `"status": "unknown"`
    rather than raising, so a caller passing a hand-built record can never crash `execute`."""
    annotated = dict(record)
    annotated["keeps"] = [
        outcome_by_src.get(path, {"path": path, "status": "unknown"})
        for path in record.get("keeps", [])
    ]
    return annotated


# FIX 2 (third review round): precedence for merging two "keeps" records for the SAME (rel, src
# path) across runs -- higher wins; a tie keeps whichever side is passed as the "new" outcome in
# `_merge_keep_records`. `delivered` is definitive and final: nothing beats it, and nothing lower
# is allowed to silently replace it (a stale on-disk "failed" must not survive a later successful
# retry, but an on-disk "delivered" must never be knocked back down to "failed" or "skipped" by a
# later run that never touched that file again). `failed`/`skipped` are equally definitive-for-now
# outcomes of an actual attempt, both outranking `unknown` (a keep with no recorded outcome at
# all, which should not happen in practice -- see `_annotate_keeps`).
_KEEP_STATUS_RANK = {"unknown": 0, "failed": 1, "skipped": 1, "delivered": 2}


def _merge_keep_records(old_keeps: list[dict], new_keeps: list[dict]) -> list[dict]:
    """Merge two `"keeps"` lists (each `{"path": ..., "status": ..., ...}`) belonging to the SAME
    source folder's provenance record -- one already on disk, one from the current run -- picking
    per source path whichever status ranks higher per `_KEEP_STATUS_RANK`. A tie (same rank on
    both sides, e.g. `failed` on disk and `failed` again this run) keeps the NEW side, since it
    reflects the more recent attempt's error message/outcome. Preserves `old_keeps`' order, with
    any path only the new side has appended at the end."""
    by_path = {k.get("path"): dict(k) for k in old_keeps}
    order = [k.get("path") for k in old_keeps]
    for k in new_keeps:
        path = k.get("path")
        if path not in by_path:
            by_path[path] = dict(k)
            order.append(path)
            continue
        old_rank = _KEEP_STATUS_RANK.get(by_path[path].get("status"), 0)
        new_rank = _KEEP_STATUS_RANK.get(k.get("status"), 0)
        if new_rank >= old_rank:
            by_path[path] = dict(k)
    return [by_path[p] for p in order]


def _merge_provenance_sources(dest_dir: str, new_records: list[dict]) -> list[dict]:
    """Combine `new_records` (this run's contribution to `dest_dir`) with whatever
    `_provenance.json` already sits there from a previous `apply --commit` run (FIX 4), so a
    second run into a well folder a prior run already populated -- e.g. reviewing more folders
    that resolve to a well already migrated -- appends rather than erasing that prior run's
    record. Grouped by source folder `rel`: a `rel` not already on disk is appended outright, in
    the order it arrives.

    A `rel` present on BOTH sides (FIX 2 -- e.g. a retried `apply --commit` after a previous
    run's move failed) is merged, not left stale and not blindly overwritten: the new record's
    own top-level fields (files/verdicts/basin -- whatever this run recomputed) replace the old
    ones, but its `"keeps"` list is merged per source path against the old record's `"keeps"` via
    `_merge_keep_records`, so a `"failed"` entry that a later run actually delivered becomes
    `"delivered"` instead of permanently reading `"failed"` -- see `_KEEP_STATUS_RANK` for the
    full precedence (`delivered` beats `failed`/`skipped`, which beat `unknown`; a newer `failed`
    never overwrites an older `delivered`).

    A missing, empty, or corrupt existing file is treated as "nothing to merge" rather than
    raising -- the write that follows is what matters, not the read."""
    path = os.path.join(dest_dir, "_provenance.json")
    existing: list[dict] = []
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                existing = json.load(fh).get("sources", [])
        except Exception:
            existing = []
    by_rel = {rec.get("rel"): rec for rec in existing}
    merged = list(existing)
    for rec in new_records:
        rel = rec.get("rel")
        old_rec = by_rel.get(rel)
        if old_rec is None:
            merged.append(rec)
            by_rel[rel] = rec
            continue
        merged_rec = dict(rec)
        merged_rec["keeps"] = _merge_keep_records(old_rec.get("keeps", []), rec.get("keeps", []))
        merged[merged.index(old_rec)] = merged_rec
        by_rel[rel] = merged_rec
    return merged


def _write_provenance_atomic(dest_dir: str, data: dict) -> None:
    """Write `<dest_dir>/_provenance.json` atomically: a temp file in the same directory, then
    `atomic.replace_with_retry` onto the final path -- same contract as
    `dfit_tool/store.py:save_picks_for`, retried on Windows' transient post-write
    `PermissionError` (see `atomic.py`)."""
    path = os.path.join(dest_dir, "_provenance.json")
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".provenance_", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        atomic.replace_with_retry(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def _finalize_provenance(
    dst_dir: str,
    provenance: dict[str, list[dict]],
    outcome_by_src: dict[str, dict],
    prov_failed: list[tuple[str, str]],
) -> None:
    """Called once `dst_dir` has seen its last move (FIX 1 + FIX 3): annotate each contributing
    record's `keeps` with the real outcome, merge with any pre-existing `_provenance.json` (FIX
    4), and write. A `dst_dir` with no provenance records at all (e.g. the `_quarantine` tree)
    is a no-op. If the write raises, it is retried exactly once more before giving up -- enough
    to survive a single transient failure without hammering a genuinely doomed write (a full
    disk) on every directory -- and a `dst_dir` that still has no provenance after both attempts
    is appended to `prov_failed` so `execute`'s caller can report it loudly rather than the
    directory silently ending up with files delivered and no provenance record at all.

    The inverse case (FIX 5, third review round) is deliberate and different: a folder the
    ledger decided `"none"` ("no DFIT here") has an empty `keeps`, so `plan_moves` emits no
    `"keep"` move for it at all -- every one of its files goes to `_quarantine` instead -- and
    that folder's `destination_dir` (its would-be well folder) therefore never appears as any
    `Move.dst`'s directory. This function is never even CALLED for that directory (not called-
    with-no-records, as the `_quarantine` case above is): the `remaining` countdown in `execute`
    has no entry for it, so it never reaches zero there, and whatever `provenance_for` record a
    caller built for that folder (`_cmd_apply` builds one per reviewed folder, decided or not)
    is silently never written. Nothing is lost from the filesystem -- no files move, none are
    quarantined into a phantom well folder -- and creating an empty well-folder directory just
    to hold a `_provenance.json` for a folder with nothing in it would be worse. This is kept as
    the accepted behavior, not treated as a gap to close."""
    records = provenance.get(dst_dir)
    if not records:
        return
    annotated = [_annotate_keeps(rec, outcome_by_src) for rec in records]
    merged = _merge_provenance_sources(dst_dir, annotated)
    for attempt in range(2):
        try:
            _write_provenance_atomic(dst_dir, {"sources": merged})
            return
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    prov_failed.append((dst_dir, last_err))


def execute(
    moves: list[Move], provenance: dict[str, list[dict]]
) -> tuple[int, int, list[tuple[Move, str]], list[tuple[str, str]]]:
    """Perform every non-skipped move. `keep`/`quarantine` are `shutil.move`; `questionnaire` is
    `shutil.copy2` so the original still serves any other folder that needs it. Never deletes
    anything on its own initiative.

    A per-move failure -- a locked file (Excel/OneDrive/Defender holding a questionnaire open), a
    path exceeding Windows MAX_PATH, a full destination disk, a ledger keep that a later re-scan
    renamed -- is caught, recorded in the returned `failed` list as `(Move, error string)`, and
    does NOT abort the run: the remaining moves still execute, so one bad file never leaves the
    tree half-migrated with the rest silently un-attempted.

    `provenance` is keyed by destination directory; each entry is the LIST of every contributing
    source folder's record (one per `provenance_for` call -- a destination well folder can be fed
    by more than one source folder, e.g. two DFIT tests on one well, see this module's
    docstring), written together as `{"sources": [...]}`. Each destination directory's
    provenance is finalized (annotated with real outcomes, merged with any prior run's record,
    and written) via `_finalize_provenance` once every move destined for that directory --
    skipped ones included -- has been processed, tracked by a per-directory countdown computed
    up front from `moves` itself. This is why the outcome recorded for each kept file is accurate
    rather than a snapshot of intent: at finalize time every move for that directory (this run's
    contribution to it, at least) is already known. Dirs with no provenance entry (e.g. the
    `_quarantine` tree) get none. Returns `(performed, skipped, failed, prov_failed)`, where
    `prov_failed` is `[(dest_dir, error string), ...]` for every directory that received files
    but still has no provenance record after `_finalize_provenance`'s one retry -- see that
    function's docstring."""
    performed = 0
    skipped = 0
    failed: list[tuple[Move, str]] = []
    prov_failed: list[tuple[str, str]] = []
    outcome_by_src: dict[str, dict] = {}

    # How many moves (skipped ones included -- a skipped move is still the last word on that
    # file) still need to be processed for each destination directory, so we know exactly when
    # we've seen the last one and can finalize that directory's provenance.
    remaining = Counter(os.path.dirname(mv.dst) for mv in moves)

    for mv in moves:
        dst_dir = os.path.dirname(mv.dst)
        if mv.skipped:
            skipped += 1
            if mv.kind == "keep":
                outcome_by_src[mv.src] = {
                    "path": mv.src, "status": "skipped", "reason": mv.skipped,
                }
        else:
            try:
                os.makedirs(dst_dir, exist_ok=True)
                if mv.kind == "questionnaire":
                    shutil.copy2(mv.src, mv.dst)
                else:
                    shutil.move(mv.src, mv.dst)
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                failed.append((mv, err))
                if mv.kind == "keep":
                    outcome_by_src[mv.src] = {"path": mv.src, "status": "failed", "error": err}
            else:
                performed += 1
                if mv.kind == "keep":
                    outcome_by_src[mv.src] = {"path": mv.src, "status": "delivered", "dst": mv.dst}

        remaining[dst_dir] -= 1
        if remaining[dst_dir] == 0:
            _finalize_provenance(dst_dir, provenance, outcome_by_src, prov_failed)

    return performed, skipped, failed, prov_failed


def write_failures_file(
    root: str,
    failed: list[tuple[Move, str]],
    prov_failed: list[tuple[str, str]] | None = None,
) -> str | None:
    """Persist `execute`'s `failed` and `prov_failed` lists to
    `<root>/_triage/apply_failures.json`, atomically, so a partial migration of a large corpus
    leaves a durable record of what did not make it rather than only the stderr lines a console
    scrolls away (FIX 3). Returns the path written, or `None` (writes nothing) if both lists are
    empty."""
    prov_failed = prov_failed or []
    if not failed and not prov_failed:
        return None
    triage_dir = os.path.join(root, "_triage")
    os.makedirs(triage_dir, exist_ok=True)
    path = os.path.join(triage_dir, "apply_failures.json")
    data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "moves_failed": [
            {"kind": mv.kind, "src": mv.src, "dst": mv.dst, "error": err} for mv, err in failed
        ],
        "provenance_failed": [
            {"dest_dir": dest_dir, "error": err} for dest_dir, err in prov_failed
        ],
    }
    fd, tmp_path = tempfile.mkstemp(dir=triage_dir, prefix=".apply_failures_", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        atomic.replace_with_retry(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path
