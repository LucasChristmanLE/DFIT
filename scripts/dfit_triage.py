"""DFIT data triage: classify, review, then move -- three phases, three subcommands.

The DFIT data tree has far more data *files* than real tests: nothing records which file in a
well folder is the one worth interpreting, so `dfit_tool.store.scan_root` makes one queue entry
per data-file stem instead of one per test. This tool reorganizes a root into a `Basin/Well/`
tree holding one keeper file (plus its questionnaire and a provenance record) per well folder, so
that fan-out disappears. No script decides on its own verdict -- a human looks at a rendered plot
of every file in a folder and picks the keeper(s); a folder can hold more than one real test, so
the decision is a *set* of keepers, not a single choice.

    python scripts\\dfit_triage.py scan   --root "C:\\DFIT Data" [--all-folders] [--limit N] [--force]
    python scripts\\dfit_triage.py review --root "C:\\DFIT Data"
    python scripts\\dfit_triage.py apply   --root "C:\\DFIT Data" --out "C:\\DFIT Organized" [--commit]

**scan** walks the root (`dfit_tool.store.scan_root`, folder-grouped), collapses byte-identical
files, extracts measured features per remaining file (rows, duration, pressure/rate channel
guesses, shut-in detection, drop, decline fraction) and a suggested verdict, pre-renders one PNG
per file, and caches everything to `<root>/_triage/features.json`. By default it is filtered to
folders that have a questionnaire (`--all-folders` lifts that filter); `--limit N` caps how many
folders are scanned (touching only those folders' files, not the whole root -- see
`features.scan_folders`'s docstring), for a quick smoke run. Because a limited scan is
necessarily incomplete, `--limit` refuses to overwrite an existing `features.json` unless
`--force` is also given; an unlimited scan always overwrites freely.

**review** opens a Tkinter window over the cached scan, one folder per screen, and lets a human
pick the keeper(s) (or say "no DFIT here" / "unsure"). Decisions land in
`<root>/_triage/decisions.json`, written atomically after every folder, so quitting mid-way and
relaunching resumes right where it left off.

**apply** turns the reviewed ledger into a move/copy plan and prints it. Nothing on disk is
touched unless `--commit` is also given -- **dry-run is the default**. Folders the ledger never
reached a decision on ("" or "unsure") are skipped entirely and reported separately, never
partially applied. A decided folder can ALSO be excluded from the plan and reported as a warning
line instead: an ambiguous well (`n_wells != 1`, several distinct wells' questionnaires under one
group -- no single well to file it under) or a stale decision (its recorded file fingerprint no
longer matches the group's current file set, including a legacy decision from before that
fingerprint existed -- see `triage.apply.plan_warnings`); both cases leave every file in that
folder completely untouched, neither filed nor quarantined, and are counted separately in the
dry-run/`--commit` summary. With `--commit`, each destination well folder gets a
`_provenance.json` recording the actual delivery outcome of every kept file (not just what was
planned), merged with any earlier `apply --commit` run's record for that folder rather than
overwriting it. Any move or provenance-write failure is printed to stderr AND persisted to
`<root>/_triage/apply_failures.json` (the path is printed), so a partial migration of a large
corpus leaves a durable record of what did not make it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

# `scripts/` sits above the `dfit_tool` package, and `scripts/triage/` is a plain (namespace)
# package next to this file -- neither is importable from a bare `python scripts\dfit_triage.py`
# invocation without both the repo root and the scripts dir on sys.path. Same fixup
# `scripts/well_locations.py` and `tests/test_well_locations.py` use for `dfit_tool`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------------------------------
# scan
# --------------------------------------------------------------------------------------------------
def _cmd_scan(args: argparse.Namespace) -> None:
    from triage import features, figure

    if args.limit is not None and not args.force:
        try:
            existing = features.load_scan(args.root)
        except Exception:
            existing = None  # missing or corrupt: nothing to protect, proceed as normal
        if existing:  # an empty scan ([] or None) holds nothing worth protecting either
            existing_path = os.path.join(features.triage_dir(args.root), "features.json")
            print(
                f"{existing_path!r} already holds a scan of {len(existing)} folder(s); a "
                f"--limit {args.limit} run would overwrite it with only {args.limit}. Pass "
                f"--force to overwrite anyway, or drop --limit to do a full rescan.",
                file=sys.stderr,
            )
            raise SystemExit(1)

    start = time.time()

    def _progress(done: int, total: int, rel: str) -> None:
        # features.scan_folders calls this once per folder, after extracting that folder's files.
        print(f"  [{done}/{total}] {rel}", file=sys.stderr)

    scans = features.scan_folders(
        args.root,
        require_questionnaire=not args.all_folders,
        limit=args.limit,
        progress=_progress,
    )

    # The scan's features are the whole point of a run that can take tens of minutes -- save them
    # BEFORE spending any more time rendering PNGs, so a render failure (or a Ctrl-C during
    # rendering) can never cost the scan itself (FIX 3).
    json_path = features.save_scan(args.root, scans)

    n_files = 0
    n_dupes = 0
    n_rendered = 0
    n_render_failures = 0
    n_suggested = 0
    verdicts: Counter[str] = Counter()

    os.makedirs(features.png_dir(args.root), exist_ok=True)
    for scan in scans:
        if scan.suggested:
            n_suggested += 1
        for feat in scan.files:
            n_files += 1
            verdicts[feat.verdict] += 1
            if feat.verdict == "duplicate":
                n_dupes += 1
                continue
            out_path = features.png_path_for(args.root, feat)
            if os.path.exists(out_path):
                continue  # a re-scan is cheap: never re-render an existing PNG
            try:
                figure.render_file_png(feat, out_path)
                n_rendered += 1
            except Exception as e:
                # render_file_png already guards load/plot failures with its own error panel;
                # this is the belt-and-suspenders guard for the one path that isn't guarded --
                # its own fallback `fig.savefig` -- so one bad render never costs the rest.
                n_render_failures += 1
                print(f"  render failed for {feat.path!r}: {type(e).__name__}: {e}",
                      file=sys.stderr)

    elapsed = time.time() - start

    print(f"Folders: {len(scans)}")
    print(f"Files: {n_files}")
    print(f"Duplicates collapsed: {n_dupes}")
    print("Verdict histogram:")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict}: {count}")
    print(f"Folders with a suggested keeper: {n_suggested}")
    print(f"PNGs rendered this run: {n_rendered}")
    if n_render_failures:
        print(f"PNG render failures: {n_render_failures}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(json_path)


# --------------------------------------------------------------------------------------------------
# review
# --------------------------------------------------------------------------------------------------
def _cmd_review(args: argparse.Namespace) -> None:
    from triage import features

    features_path = os.path.join(features.triage_dir(args.root), "features.json")
    if not os.path.exists(features_path):
        print(f"No features.json found under {features.triage_dir(args.root)!r} -- run `scan` "
              f"first.", file=sys.stderr)
        raise SystemExit(1)

    from triage import review_app
    review_app.main(args.root)


# --------------------------------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------------------------------
def _cmd_apply(args: argparse.Namespace) -> None:
    from triage import apply as apply_mod
    from triage import features
    from triage.ledger import Ledger

    try:
        apply_mod.validate_out_not_inside_root(args.root, args.out)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)

    features_path = os.path.join(features.triage_dir(args.root), "features.json")
    if not os.path.exists(features_path):
        print(f"No features.json found under {features.triage_dir(args.root)!r} -- run `scan` "
              f"first.", file=sys.stderr)
        raise SystemExit(1)

    scans = features.load_scan(args.root)
    ledger = Ledger.load(args.root)

    moves = apply_mod.plan_moves(args.root, args.out, scans, ledger)
    # A decided folder that plan_moves silently excluded -- an ambiguous well (n_wells != 1) or a
    # decision whose files_sig no longer matches its current file set -- plus a ledger `rel` that
    # matches no scan at all. `excluded_rels` is exactly the set plan_moves also skipped, so the
    # provenance loop below never disagrees with what actually got planned.
    warnings = apply_mod.plan_warnings(scans, ledger)
    excluded_rels = {w.rel for w in warnings if w.category in ("ambiguous_well", "stale_decision")}
    n_ambiguous = sum(1 for w in warnings if w.category == "ambiguous_well")
    n_stale_or_unknown = sum(1 for w in warnings if w.category in ("stale_decision", "unknown_group"))

    n_skipped_folders = 0
    # Keyed by destination dir, one LIST entry per contributing source folder -- two different
    # source folders can resolve to the same destination well folder (FIX 2, third review round),
    # so this must never collapse to one record per dest_dir.
    provenance: dict[str, list[dict]] = {}
    for scan in scans:
        decision = ledger.get(scan.rel)
        if decision.status in ("", "unsure"):
            n_skipped_folders += 1
            continue
        if scan.rel in excluded_rels:
            continue  # ambiguous well or stale decision -- reported via `warnings`, never filed
        dest_dir, basin, basin_source = apply_mod.destination_dir(scan, args.out, args.root)
        well = apply_mod.well_folder_name(scan)
        record = apply_mod.provenance_for(scan, decision.keeps, basin, basin_source, well)
        provenance.setdefault(dest_dir, []).append(record)

    by_dest: dict[str, list] = {}
    for mv in moves:
        by_dest.setdefault(os.path.dirname(mv.dst), []).append(mv)

    for dest_dir in sorted(by_dest):
        print(dest_dir)
        for mv in by_dest[dest_dir]:
            note = f"  [SKIP: {mv.skipped}]" if mv.skipped else ""
            print(f"  ({mv.kind}) {mv.src} -> {os.path.basename(mv.dst)}{note}")

    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  {w.message}")

    n_performed_candidates = sum(1 for mv in moves if not mv.skipped)
    n_skipped_moves = sum(1 for mv in moves if mv.skipped)
    print()
    print(f"Folders skipped (undecided/unsure): {n_skipped_folders}")
    print(f"Folders skipped (ambiguous well, n_wells != 1): {n_ambiguous}")
    print(f"Folders skipped (stale decision / unknown group): {n_stale_or_unknown}")
    print(f"Moves planned: {len(moves)} "
          f"({n_performed_candidates} to perform, {n_skipped_moves} skipped)")

    if not args.commit:
        print("Dry run -- no changes made. Pass --commit to apply this plan.")
        return

    performed, skipped, failed, prov_failed = apply_mod.execute(moves, provenance)
    print(f"Committed: {performed} performed, {skipped} skipped, {len(failed)} failed")
    if prov_failed:
        print(f"PROVENANCE FAILURES ({len(prov_failed)}) -- these directories received files "
              f"but have no _provenance.json:", file=sys.stderr)
        for dest_dir, err in prov_failed:
            print(f"  {dest_dir}: {err}", file=sys.stderr)
    if failed:
        print(f"FAILURES ({len(failed)}) -- these moves did NOT complete, everything else did:",
              file=sys.stderr)
        for mv, err in failed:
            print(f"  ({mv.kind}) {mv.src} -> {mv.dst}: {err}", file=sys.stderr)
    failures_path = apply_mod.write_failures_file(args.root, failed, prov_failed)
    if failures_path:
        print(f"Failure details written to {failures_path}")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dfit_triage.py",
        description="Classify, review, then move DFIT data files into a Basin/Well/ tree.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="measure features and pre-render plots for a root")
    p_scan.add_argument("--root", required=True, help="folder-mode root to scan")
    p_scan.add_argument("--all-folders", action="store_true",
                         help="include folders with no questionnaire (default: filtered to "
                              "folders that have one)")
    p_scan.add_argument("--limit", type=int, default=None, help="cap the number of folders scanned")
    p_scan.add_argument("--force", action="store_true",
                         help="with --limit, overwrite an existing features.json anyway "
                              "(default: refuse, to avoid clobbering a full scan)")
    p_scan.set_defaults(func=_cmd_scan)

    p_review = sub.add_parser("review", help="open the Tkinter review window over a cached scan")
    p_review.add_argument("--root", required=True, help="folder-mode root previously `scan`ned")
    p_review.set_defaults(func=_cmd_review)

    p_apply = sub.add_parser("apply", help="plan (and, with --commit, perform) the reorganize")
    p_apply.add_argument("--root", required=True, help="folder-mode root previously `scan`ned "
                                                         "and `review`ed")
    p_apply.add_argument("--out", required=True, help="destination root for the Basin/Well/ tree")
    p_apply.add_argument("--commit", action="store_true",
                          help="actually perform the plan (default: dry-run, print the plan only)")
    p_apply.set_defaults(func=_cmd_apply)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
