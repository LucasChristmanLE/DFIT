"""Per-file measured features and per-folder scan results for DFIT data triage.

``extract`` loads one data file and measures the handful of numbers a human reviewer needs to
judge "is this the interpretable DFIT file, or a duplicate / a raw gauge log / something that
won't load at all": duration, shut-in instant, post-shut-in span, pressure drop, decline shape,
and a suggested (never decided) ``verdict``. ``scan_folders`` walks a folder-mode root
(``dfit_tool.store.scan_root``) and groups entries by **well root**, not by ``entry.folder``
directly -- see ``scan_folders``' docstring for the walk-up rule and why it doesn't repeat the
old grouping-by-questionnaire-directory failure -- dedups byte-identical files *within* each
group, and extracts features for everything else, caching the extraction by content signature so
a file that happens to be byte-identical to one in a different group (a real corpus case -- the
same raw export copied into two separate DFIT-test folders on the same well) is only parsed once.

Pure Python + numpy + pandas (via ``dfit_tool.io_load``) -- no matplotlib, no Tkinter, so this
module is fully unit-testable headless, matching ``dfit_tool/store.py``'s layering.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np

# `scripts/triage/` sits two levels below the repo root, which itself sits above `dfit_tool` --
# same sys.path fixup `scripts/well_locations.py` does, so this module works whether it's
# imported through an entry-point script that already fixed up sys.path, or directly (e.g. from
# a test that only added `scripts/` to sys.path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dfit_tool import io_load, interpret, questionnaire, store  # noqa: E402

from .atomic import replace_with_retry  # noqa: E402

VERDICTS = (
    "likely_dfit", "too_short", "flat", "short_falloff", "noisy", "no_pressure", "load_error",
    "duplicate",
)

# --------------------------------------------------------------------------------------------------
# verdict thresholds
# --------------------------------------------------------------------------------------------------
_MIN_PRESSURE_SAMPLES = 50
_MIN_DURATION_HR = 1.0
_FLAT_DROP_PSI = 50.0
_MIN_POST_SHUTIN_HR = 4.0
_MIN_DECLINE_FRACTION = 0.6
_MIN_DECLINE_SAMPLES = 10  # decline_fraction needs more post-shut-in samples than this, else None

_KPA_PMAX_THRESHOLD = 40000.0  # p_max above this -> unit_guess "kPa" (real files peak ~100,000 kPa)
_PSI_PMAX_THRESHOLD = 100.0    # p_max above this (and below the kPa threshold) -> unit_guess "psi"

# 1 psi = 6.895 kPa. The drop/flat thresholds above are stated in psi; when unit_guess is "kPa"
# they're scaled by this constant instead of maintaining a second, parallel set of kPa constants.
_PSI_TO_KPA = 6.895

# signature(): head+tail blake2b, matching the measured-cheap approach (3.4 s over the full
# 2,464-file corpus). Tail is only read for files bigger than twice the head/tail chunk size, so
# a small file is never read twice.
_SIG_CHUNK_BYTES = 262144
_SIG_TAIL_MIN_SIZE = 524288

# FIX 3: png_path_for keys the cached PNG filename on this in addition to the content signature.
# Bump this whenever render_file_png's plotted output changes OR the FileFeatures fields it
# annotates onto the panel change (duration_hr, verdict, trailing_dropped, etc.) -- otherwise a
# stale PNG rendered under the old logic keeps being reused by `os.path.exists` after a re-scan,
# showing a panel that disagrees with the fresh features.json numbers next to it.
#
# Bumped to 3 for FIX 4's nonfinite_time_dropped annotation line (third review round). An earlier
# pass left this pinned at 2 on the premise that the line is conditional on the value being
# non-zero, so "most" panels would render byte-identical. That premise was checked against the
# real corpus and was wrong: 14 of 30 sampled CSVs (47%) have at least one unparseable timestamp,
# and CSVs are 588 of the 1,237 scanned files, so roughly 275 of 1,237 files (~22%) carry a
# non-zero nonfinite_time_dropped -- not ~1%. Worse, `features.json` predates this field entirely,
# so `load_scan` defaults every already-cached file to 0, and `review_app.py` only ever displays
# the cached PNG grid (never a raw `FileFeatures` number) -- so the annotation is the ONLY channel
# by which a human learns samples were dropped, and it can only reach the screen by invalidating
# every cached PNG so the next scan re-renders them all. Keep bumping this whenever
# render_file_png's output or the features feeding its annotation change.
PNG_RENDER_VERSION = 4


# --------------------------------------------------------------------------------------------------
# data shapes
# --------------------------------------------------------------------------------------------------
@dataclass
class FileFeatures:
    path: str                      # absolute
    folder: str                    # absolute dir containing it
    size_bytes: int
    sig: str                       # "<size>:<blake2b-hex>"
    dup_of: str | None = None      # path of the representative when this is a byte-duplicate
    # Paths of files in OTHER folders sharing this file's signature -- purely informational (a
    # real corpus case: the same raw gauge export copied into two separate DFIT-test folders on
    # the same well). Never affects verdict or `suggested`; see scan_folders' dedup rule.
    same_bytes_as: list[str] = field(default_factory=list)
    rows: int | None = None
    duration_hr: float | None = None
    # Samples trimmed off the end by `monotonic_prefix` before any measurement -- e.g. real DBS
    # files carry a block of `idx == 0` padding rows at the end, which resets `t_s` to 0 and would
    # otherwise corrupt duration_hr/post_shutin_hr/drop/decline_fraction. 0 when nothing was
    # trimmed (the common case for a well-formed CSV).
    trailing_dropped: int = 0
    # FIX 4 (third review round): samples dropped BEFORE monotonic_prefix ever runs, because their
    # timestamp didn't parse (NaN t_s -- FIX 1's masking step). Counted separately from
    # trailing_dropped (which only counts monotonic-truncation, not this earlier NaN-timestamp
    # drop) so a file whose datetime column is largely unparseable doesn't silently report
    # confident numbers measured on a fraction of its rows with nothing on screen saying so. 0 when
    # every timestamp parsed (the common case).
    nonfinite_time_dropped: int = 0
    pressure_col: str | None = None
    rate_col: str | None = None
    unit_guess: str = ""           # "psi" | "kPa" | ""
    p_max: float | None = None
    p_at_shutin: float | None = None
    shutin_source: str = ""        # "rate" | "pressure-peak" | ""
    injection_min: float | None = None
    post_shutin_hr: float | None = None
    drop: float | None = None
    decline_fraction: float | None = None
    verdict: str = ""
    load_error: str = ""           # "" when the load succeeded


@dataclass
class FolderScan:
    folder: str                    # absolute; the WELL ROOT (see scan_folders), not necessarily
                                    # any single entry's own immediate directory
    rel: str                       # relative to root, forward slashes
    well_name: str = ""            # from questionnaire, "" if none
    formation: str = ""            # from questionnaire, "" if none
    questionnaire_path: str = ""   # "" if none
    # Distinct well names found in this group's own subtree (see scan_folders). 1 is the normal
    # case; >1 means this well root actually holds more than one distinct well's questionnaire
    # (an ambiguous folder review_app.py flags rather than silently picking one) and 0 is only
    # possible with require_questionnaire=False (no questionnaire anywhere in the subtree).
    # Defaults to 1 so a features.json written before this field existed loads without crashing
    # (load_scan filters to known fields; a missing key just falls back to this default).
    n_wells: int = 1
    files: list[FileFeatures] = field(default_factory=list)
    suggested: list[str] = field(default_factory=list)   # paths pre-selected as keepers


# --------------------------------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------------------------------
def triage_dir(root: str) -> str:
    return os.path.join(root, "_triage")


def png_dir(root: str) -> str:
    return os.path.join(root, "_triage", "png")


def png_path_for(root: str, feat: FileFeatures) -> str:
    """A stable PNG path for `feat`, named from its content signature so byte-identical files
    (including the cross-folder case) share one cached render. Also keyed on
    `PNG_RENDER_VERSION`, so bumping that constant invalidates every previously cached PNG (the
    filename simply no longer matches anything on disk) instead of a stale pre-fix render being
    silently reused next to fresh, disagreeing numbers -- see FIX 3."""
    safe = feat.sig.replace(":", "_").replace(os.sep, "_")
    return os.path.join(png_dir(root), f"{safe}_v{PNG_RENDER_VERSION}.png")


# --------------------------------------------------------------------------------------------------
# signature
# --------------------------------------------------------------------------------------------------
def signature(path: str) -> str:
    """`"<size>:<blake2b-hex>"` of the first/last `_SIG_CHUNK_BYTES` of `path` (tail only read
    when the file exceeds `_SIG_TAIL_MIN_SIZE`). Cheap enough to run over the whole corpus
    (measured 3.4 s / 2,464 files) while still being a real content signature, not just a size
    check -- two files of the same size with different content essentially never collide."""
    size = os.path.getsize(path)
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        h.update(f.read(_SIG_CHUNK_BYTES))
        if size > _SIG_TAIL_MIN_SIZE:
            f.seek(-_SIG_CHUNK_BYTES, os.SEEK_END)
            h.update(f.read(_SIG_CHUNK_BYTES))
    return f"{size}:{h.hexdigest()}"


# --------------------------------------------------------------------------------------------------
# monotonic prefix
# --------------------------------------------------------------------------------------------------
def monotonic_prefix(t_s: np.ndarray) -> slice:
    """The slice selecting the genuinely usable span of `t_s`, which callers must have already
    stripped of non-finite samples (FIX 1 -- a NaN timestamp is missing data, not a record
    boundary, and comparing it against anything is always False, so this function must never be
    handed one: `extract` and `figure.render_file_png` both mask `t_s` finite BEFORE calling
    this).

    Contract (FIX 6): given `t_s` already stripped of non-finite samples (the caller's
    responsibility, see above), this returns the longest non-decreasing prefix -- computed
    genuinely by walking forward to the first decrease, never a reversed or negative-step slice.
    This replaced the old "last index equal to the running max" shortcut, which let interior
    non-monotonicity (a dip back down, then a further rise) pass through untouched.

    `io_load` builds DBS time as ``rec["idx"] * interval_min * 60``, and real DBS files carry a
    block of ``idx == 0`` padding rows at the END, so `t_s` is not monotonic there and its last
    element is 0.0 rather than the record's true end. Trusting `t_s[-1]` as the end of the record
    silently corrupts every duration/span measurement (and, in a plot, draws a spurious
    return-to-zero segment). Both `extract` and `figure.render_file_png` truncate through this one
    helper so the numbers and the plot can never disagree.

    - All-monotonic input is returned unchanged.
    - Trailing padding (or any interior dip) truncates the prefix at the sample right before the
      first decrease -- not at the last occurrence of the global max.
    - A single-sample or empty array is a no-op (an all-zero array is trivially non-decreasing
      throughout, so it is also returned unchanged).

    NOT handled here (third review round, FIX 1), and updated since: a record that is decreasing
    across most or even all of its steps -- a scrambled merge of two out-of-order channel streams,
    say -- is still truncated the same as any other non-monotonic record, collapsing to a short
    (possibly 1-sample) prefix and typically reading `too_short`. An earlier version of this
    function special-cased a "mostly decreasing" record as reversed and returned the reversed
    view; that was removed rather than hardened further. But the specific case this docstring used
    to call out -- a genuinely, wholly reverse-chronological export -- is no longer one this
    function needs to handle: `io_load`'s FIX D now reverses that shape upstream, in `load_csv`
    itself, before `extract` (and therefore this function) ever sees `t_s`. The earlier "zero
    corpus files reach this path" claim was also wrong on its own terms -- it was measured by
    looking for negative `duration_hr`/`post_shutin_hr`, which a reversed file never produces (it
    truncates to a short prefix, not a negative span). The real signature was silent truncation to
    `rows=1`: 2 Civitas Bijou SignalFire files hit exactly that before FIX D, and now recover
    396,682 rows between them once `load_csv` reverses them first. This module's `verdict` only
    orders/pre-highlights panels for a human reviewer -- it gates nothing, every file stays visible
    and selectable regardless of verdict -- so a record that somehow still reaches this function
    reversed (e.g. a shape FIX D's "wholly descending" check doesn't catch) collapses to a short
    prefix and reads `too_short`, which is the intended fallback behavior, not a bug."""
    n = len(t_s)
    if n <= 1:
        return slice(0, n)
    diffs = np.diff(t_s)
    decreasing = diffs < 0
    if not decreasing.any():
        return slice(0, n)
    first_decrease = int(np.argmax(decreasing))  # index into diffs of the first t_s[i+1] < t_s[i]
    return slice(0, first_decrease + 1)


# --------------------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------------------
def extract(path: str) -> FileFeatures:
    """Load `path` and measure its `FileFeatures`. Never raises: any failure (unreadable file,
    unparseable format, no recognizable pressure channel) lands in `load_error`/`verdict`
    instead."""
    feat = FileFeatures(path=path, folder=os.path.dirname(path), size_bytes=0, sig="")
    try:
        feat.size_bytes = os.path.getsize(path)
        feat.sig = signature(path)

        td = io_load.load(path)
        guess = io_load.suggest_channels(td.columns)
        pressure_col = guess.get("pressure")
        if not pressure_col:
            feat.verdict = "no_pressure"
            return feat

        p_all = td.column(pressure_col)
        finite = np.isfinite(p_all)
        if int(finite.sum()) < _MIN_PRESSURE_SAMPLES:
            feat.pressure_col = pressure_col
            feat.verdict = "no_pressure"
            return feat

        feat.pressure_col = pressure_col
        feat.rate_col = guess.get("rate")

        t_s_full = np.asarray(td.t_s, dtype=float)
        p_clean_full = np.where(finite, p_all, np.nan)  # NaN non-finite so nan* reductions ignore

        # FIX 1: a NaN timestamp is missing data, not a record boundary. `io_load.load_csv` keeps
        # NaT rows as long as at least one datetime in the file parsed, so `t_s` can be NaN at the
        # very first sample, in the middle, or scattered throughout -- and NaN never compares
        # equal (or less/greater) to anything, so handing it to `monotonic_prefix` either raised
        # (the old running-max check) or silently truncated the record at the first gap. Drop
        # those samples here, in the same masking step that already turned non-finite pressure
        # into NaN above, BEFORE `monotonic_prefix` ever sees the array.
        finite_t = np.isfinite(t_s_full)
        t_s_nonan = t_s_full[finite_t]
        p_clean_nonan = p_clean_full[finite_t]
        feat.nonfinite_time_dropped = len(t_s_full) - len(t_s_nonan)  # FIX 4: surfaced, not silent

        # Truncate to the genuinely usable span of the NaN-free `t_s` BEFORE any measurement --
        # see monotonic_prefix's docstring. Everything below reads only this view.
        mono = monotonic_prefix(t_s_nonan)
        t_s = t_s_nonan[mono]
        p_clean = p_clean_nonan[mono]
        feat.trailing_dropped = len(t_s_nonan) - len(t_s)

        # FIX 5: rows counts the finite-pressure samples actually surviving both the NaN-timestamp
        # drop and the monotonic-prefix truncation above, not the pre-truncation full array --
        # every other measurement below is computed on this same truncated view, and `rows` must
        # agree with them rather than counting trimmed padding.
        feat.rows = int(np.isfinite(p_clean).sum())

        feat.duration_hr = float((t_s[-1] - t_s[0]) / 3600.0) if len(t_s) else 0.0
        # Defensive clamp (FIX 1): `monotonic_prefix` is contracted to hand back a non-decreasing
        # `t_s`, but a negative duration is nonsense that must never reach a review panel even if
        # that contract is ever violated by a future change to this function -- clamp here rather
        # than trust the caller.
        feat.duration_hr = max(0.0, feat.duration_hr)
        feat.p_max = float(np.nanmax(p_clean)) if np.isfinite(p_clean).any() else 0.0

        if feat.p_max > _KPA_PMAX_THRESHOLD:
            feat.unit_guess = "kPa"
        elif feat.p_max > _PSI_PMAX_THRESHOLD:
            feat.unit_guess = "psi"
        else:
            feat.unit_guess = ""

        # Shut-in: prefer the rate channel's main-injection window; fall back to the pressure
        # peak when there's no rate channel, or `suggest_injection_window` finds no active rate
        # (a gauge-only file, or a rate column that never actually pumped). rate/volume go through
        # the same `finite_t` mask then `mono` slice as t_s/p_clean, so the indices they yield
        # line up.
        shutin_idx: int | None = None
        if feat.rate_col:
            rate = td.column(feat.rate_col)[finite_t][mono]
            volume_col = guess.get("volume")
            volume = td.column(volume_col)[finite_t][mono] if volume_col else None
            try:
                start_idx, shutin_idx = interpret.suggest_injection_window(rate, volume=volume)
                feat.shutin_source = "rate"
                feat.injection_min = float((t_s[shutin_idx] - t_s[start_idx]) / 60.0)
            except ValueError:
                shutin_idx = None

        if shutin_idx is None:
            feat.shutin_source = "pressure-peak"
            feat.injection_min = None
            shutin_idx = int(np.nanargmax(p_clean))

        p_shutin = p_clean[shutin_idx]
        feat.p_at_shutin = float(p_shutin) if np.isfinite(p_shutin) else None
        feat.post_shutin_hr = float((t_s[-1] - t_s[shutin_idx]) / 3600.0)
        # Same defensive clamp as duration_hr above (FIX 1) -- never negative, regardless of what
        # monotonic_prefix or the shut-in index resolution hands back.
        feat.post_shutin_hr = max(0.0, feat.post_shutin_hr)

        post = p_clean[shutin_idx:]
        post_finite = post[np.isfinite(post)]
        if feat.p_at_shutin is not None and post_finite.size:
            feat.drop = feat.p_at_shutin - float(np.min(post_finite))
        if post_finite.size > _MIN_DECLINE_SAMPLES:
            diffs = np.diff(post_finite)
            feat.decline_fraction = float(np.mean(diffs <= 0))

        scale = _PSI_TO_KPA if feat.unit_guess == "kPa" else 1.0
        flat_threshold = _FLAT_DROP_PSI * scale

        if feat.duration_hr < _MIN_DURATION_HR:
            feat.verdict = "too_short"
        elif feat.drop is None or feat.drop < flat_threshold:
            feat.verdict = "flat"
        elif feat.post_shutin_hr is None or feat.post_shutin_hr < _MIN_POST_SHUTIN_HR:
            feat.verdict = "short_falloff"
        elif feat.decline_fraction is None or feat.decline_fraction < _MIN_DECLINE_FRACTION:
            feat.verdict = "noisy"
        else:
            feat.verdict = "likely_dfit"

    except Exception as e:  # a single bad file must never abort a folder/corpus scan
        feat.load_error = f"{type(e).__name__}: {e}"
        feat.verdict = "load_error"

    return feat


# --------------------------------------------------------------------------------------------------
# well-root grouping
# --------------------------------------------------------------------------------------------------
# The blind spot this section fixes: a well's files often aren't all in one immediate directory --
# e.g. a well folder holding a data file directly, plus a `Customer Data\` subfolder with the
# questionnaire and more data, plus an `OLD Files\` subfolder with still more data. Grouping by
# `entry.folder` (the old rule) put each of those in its own `FolderScan`, and the `OLD Files`/
# `Customer Data` ones had no questionnaire attached (`find_questionnaire`'s walk only checks a
# file's own directory then its immediate parent), so `require_questionnaire=True` silently
# dropped them -- 165+ entries corpus-wide.
#
# Well-root grouping fixes this by walking a well's ENTIRE subtree, not just one directory's
# parent, while still applying the *same* anti-merge principle the old `entry.folder` rule was
# reverted to protect: two distinct wells must never land on one review page. The guard here is a
# COUNT of distinct well names in a candidate parent's subtree, not merely whether a questionnaire
# is *reachable* from it -- which is exactly the distinction that keeps this from repeating the old
# failure. The old bug (grouping by `os.path.dirname(entry.questionnaire_path)`) merged Bonanza
# Creek's 7 separate wells because they all inherited the SAME single customer-level questionnaire
# through `find_questionnaire`'s upward walk -- reachability said "yes" for all 7 with no way to
# tell them apart. Here, a customer directory whose subtree holds several DIFFERENT wells' own
# questionnaires reports a name count > 1 for that directory, so the walk-up below stops at each
# well's own subfolder rather than continuing up into the shared customer directory -- the
# ambiguity is visible as a number, not hidden behind "found *a* questionnaire somewhere above".
# Two real corpus shapes this measures the same way but must resolve differently:
#   - BKH HDU 9-11AH (Black Hills): the well folder, its `Customer Data\` and `OLD Files\`
#     subfolders each hold exactly one well's data/questionnaire -- one name, all the way up to
#     (but not including) the `Black Hills` customer directory, which holds >1. Walk-up merges
#     the three into one group at the well folder.
#   - A customer directory with several well subfolders, each with ITS OWN questionnaire: that
#     directory's subtree reports >1 names immediately, so the walk-up from any one well subfolder
#     stops there -- never merges into the customer level. This is the case the old fix protected
#     and this rule protects it too, just by counting instead of by directory identity.
# A parse failure or unreadable questionnaire is treated as a private, unique "name" (see
# `_questionnaire_well_names`) rather than "no name" or "the same name as any other unreadable
# one" -- either of those alternatives could let two folders that could NOT be shown to be the same
# well merge anyway, which this rule is built to never do.
def _walk_dirs_and_questionnaires(root: str) -> tuple[list[str], list[str]]:
    """One `os.walk(root)` pass returning `(dirs, quest_paths)` -- every directory visited
    (normalized) and every questionnaire file's absolute path (per the exact filename predicate
    `find_questionnaire` itself uses, `questionnaire.is_questionnaire_filename`, so this can never
    disagree with what `store.scan_root`'s per-entry `find_questionnaire` calls would find). Skips
    this module's own `_triage` output directory (same pruning `save_scan`'s sibling `png_dir`
    lives under) so a stray earlier scan's artifacts are never mistaken for corpus questionnaires.

    FIX 5 (perf): `_all_questionnaire_paths` and `_subtree_well_names` each used to walk the whole
    tree on their own -- one call site (`scan_folders`) needed both, so that was two full walks of
    a corpus that can be tens of thousands of directories. Sharing this one walk halves that cost
    with identical results, since both walks were pruning and visiting exactly the same tree."""
    dirs: list[str] = []
    quest_paths: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "_triage"]
        dirs.append(os.path.normpath(dirpath))
        for name in filenames:
            if questionnaire.is_questionnaire_filename(name):
                quest_paths.append(os.path.join(dirpath, name))
    return dirs, quest_paths


def _all_questionnaire_paths(root: str) -> list[str]:
    """Every questionnaire file anywhere under `root` (any depth) -- see
    `_walk_dirs_and_questionnaires`, which this delegates to. Kept as its own function (rather than
    inlining the walk here) since nothing about its contract changed, only how it gets its answer;
    a caller that also needs the visited-directories list should call
    `_walk_dirs_and_questionnaires` directly instead, the way `scan_folders` does, to avoid a
    second walk."""
    _dirs, quest_paths = _walk_dirs_and_questionnaires(root)
    return quest_paths


# FIX 5 (perf): populated as a side effect of the REAL `_questionnaire_well_names` (below) every
# time it runs, keyed by questionnaire path, holding each path's already-parsed
# `questionnaire.parse_questionnaire` result (or `None` for a path that failed to parse). Rebuilt
# from scratch (cleared, then repopulated) on every call, so it only ever reflects the most recent
# run -- never an accumulating, unbounded cache across an entire process's lifetime. Its only
# reader is `scan_folders`'s per-group well_name/formation lookup, which reuses a path's parse
# instead of calling `parse_questionnaire` on it a second time. When `_questionnaire_well_names`
# is monkeypatched (as several tests in this repo do, to avoid authoring real `.xlsx` fixtures for
# every grouping shape), this cache is simply never touched and the per-group lookup falls back to
# parsing fresh -- a pure performance cache, never load-bearing for correctness, so a test that
# bypasses it changes nothing about what a test asserts.
_LAST_QUESTIONNAIRE_PARSE_CACHE: dict[str, object | None] = {}


def _questionnaire_well_names(paths: list[str]) -> dict[str, str]:
    """Normalized well name for each questionnaire path in `paths` (stripped, casefolded,
    internal whitespace collapsed to one space) -- factored out so tests can monkeypatch it
    instead of authoring real `.xlsx` fixtures for every grouping shape.

    A parse failure (bad/corrupt workbook), an exception of any kind, or a missing/empty
    `well_name` field all yield a private sentinel, `f"?{path}"`, unique to that one path. This
    is deliberately conservative (FEATURE 1, point 2): a sentinel can never equal another path's
    sentinel or any real parsed name, so an unreadable or nameless questionnaire can only ever
    ADD a distinct name to its directory's count -- it can never cause two folders to look like
    the same well and merge.

    As a side effect (FIX 5, perf), each path's parsed `QuestionnaireResult` (or `None` on parse
    failure) is stashed in `_LAST_QUESTIONNAIRE_PARSE_CACHE`, keyed by path, so `scan_folders`'s
    later per-group well_name/formation lookup can reuse it instead of re-parsing the same file."""
    names: dict[str, str] = {}
    _LAST_QUESTIONNAIRE_PARSE_CACHE.clear()
    for path in paths:
        name = None
        qres = None
        try:
            qres = questionnaire.parse_questionnaire(path)
            raw = qres.well_name
            if raw:
                normalized = " ".join(raw.strip().casefold().split())
                name = normalized or None
        except Exception:
            name = None
            qres = None
        _LAST_QUESTIONNAIRE_PARSE_CACHE[path] = qres
        names[path] = name if name is not None else f"?{path}"
    return names


def _subtree_well_names(
    root: str, quest_paths: list[str], dirs: list[str] | None = None
) -> dict[str, set[str]]:
    """`{directory: {distinct well names anywhere in that directory's subtree}}`, for every
    directory `os.walk(root)` visits (`_triage` pruned, same as `_all_questionnaire_paths`).
    `dirs`, if given, is used as-is instead of walking `root` again (FIX 5, perf) -- pass the same
    list `_walk_dirs_and_questionnaires` already returned alongside `quest_paths`, the way
    `scan_folders` does; a caller with no such list yet (or a direct/standalone call) gets one
    walked here, matching this function's original contract.

    Computed bottom-up: each directory starts with just its OWN immediate questionnaires' names,
    then -- processing deepest directories first -- merges into its parent. By the time a
    directory's own turn to merge upward comes, every one of its descendants has already merged
    into it, so its set is complete (every name anywhere beneath it), not just its own."""
    names_by_path = _questionnaire_well_names(quest_paths)

    if dirs is None:
        dirs, _quest_paths = _walk_dirs_and_questionnaires(root)

    subtree: dict[str, set[str]] = {d: set() for d in dirs}
    for path, name in names_by_path.items():
        d = os.path.normpath(os.path.dirname(path))
        subtree.setdefault(d, set()).add(name)

    for d in sorted(dirs, key=lambda p: p.count(os.sep), reverse=True):
        parent = os.path.normpath(os.path.dirname(d))
        if parent != d and parent in subtree:
            subtree[parent] |= subtree[d]

    return subtree


def _is_under(path: str, ancestor: str) -> bool:
    """True if `path` equals `ancestor`, or is nested anywhere below it."""
    path = os.path.normpath(path)
    ancestor = os.path.normpath(ancestor)
    if path == ancestor:
        return True
    rel = os.path.relpath(path, ancestor)
    return rel != os.pardir and not rel.startswith(os.pardir + os.sep)


def _is_strictly_under(path: str, ancestor: str) -> bool:
    """True if `path` is nested below `ancestor` but is not `ancestor` itself."""
    path = os.path.normpath(path)
    ancestor = os.path.normpath(ancestor)
    return path != ancestor and _is_under(path, ancestor)


def _well_root(entry_folder: str, root: str, subtree_names: dict[str, set[str]]) -> str:
    """`entry_folder`'s well-root group directory: a best-tracking climb up from `entry_folder`,
    stopping the instant it reaches an unambiguous verdict.

    `best` starts at `entry_folder` and only ever advances to a parent whose subtree holds
    EXACTLY one distinct well name (per `subtree_names`, see `_subtree_well_names`) -- that parent
    becomes the new `best` AND the climb continues from there, since an ancestor further up might
    still turn out to be one-name too (see the BKH walk-up-through-several-1-name-levels case in
    this module's docstring). A parent with ZERO names (nothing under it says anything either way)
    is climbed straight through WITHOUT updating `best` -- this is the fix over the old rule, which
    used to stop (and settle for `entry_folder` itself) at the first such parent: a data file three
    directories below a well folder, with the well's only questionnaire off in a sibling subfolder,
    used to group at that file's own immediate parent, find no questionnaire, and get silently
    dropped under `require_questionnaire=True` -- the exact data-loss class this whole feature
    exists to fix. The climb stops for good, keeping whatever `best` has accumulated so far, the
    first time it reaches a parent with MORE than one name (a genuinely different, unrelated well
    up there -- the anti-merge guard) or a parent that is `root` itself or beyond (a group can never
    be promoted to be the whole scanned root).

    Consequence worth being explicit about: deep-nested data with no questionnaire of its own now
    merges into a single-name well root even when several 0-name directories separate them (e.g.
    `Well1\\Raw Data\\CSVs\\deep.csv`, with the well's only questionnaire in a sibling
    `Well1\\Customer Data\\`) -- intentional, see `scan_folders`' docstring: visible data with a
    provenance label beats an invisible, silently-dropped file. A 0-name chain with NO 1-name
    ancestor below `root` anywhere above it never updates `best` at all, so `entry_folder` is
    returned unchanged and `require_questionnaire=True` drops it, exactly as before."""
    best = os.path.normpath(entry_folder)
    current = best
    root_norm = os.path.normpath(root)
    while True:
        parent = os.path.normpath(os.path.dirname(current))
        if not _is_strictly_under(parent, root_norm):
            break
        n = len(subtree_names.get(parent, set()))
        if n > 1:
            break
        if n == 1:
            best = parent
        current = parent
    return best


def _group_questionnaire_path(group_folder: str, quest_paths: list[str]) -> str:
    """The questionnaire in `group_folder`'s own subtree at the shallowest depth (ties broken
    alphabetically) -- `""` if none. This is the group's "official" questionnaire even when the
    group is ambiguous (`FolderScan.n_wells > 1`); `n_wells` is what tells the reviewer that, not
    this pick, which always resolves to exactly one path or none."""
    candidates = [p for p in quest_paths if _is_under(os.path.dirname(p), group_folder)]
    if not candidates:
        return ""

    def _depth(p: str) -> int:
        return os.path.normpath(os.path.dirname(p)).count(os.sep)

    return min(candidates, key=lambda p: (_depth(p), p))


# --------------------------------------------------------------------------------------------------
# scan_folders
# --------------------------------------------------------------------------------------------------
def scan_folders(
    root: str,
    require_questionnaire: bool = True,
    limit: int | None = None,
    progress=None,
) -> list[FolderScan]:
    """One `FolderScan` per **well root** found under `root`, in `rel` order -- see the
    well-root-grouping section above this function for the full rationale and the anti-merge
    guard, and `_well_root`'s own docstring for the exact walk-up rule (a best-tracking climb: an
    entry's group starts at `entry.folder`; a 1-name ancestor becomes the new best-so-far and the
    climb keeps going from there; a 0-name ancestor is climbed straight through without updating
    the best-so-far; a >1-name ancestor, or `root` itself, stops the climb for good). `FileFeatures.
    folder` still records each file's own immediate directory (unchanged), which can now differ
    from its `FolderScan.folder` (the well root) -- possibly by several levels, now that a 0-name
    ancestor no longer halts the climb.

    `require_questionnaire=True` keeps a group iff its own subtree holds >= 1 questionnaire file
    (equivalently, `FolderScan.n_wells >= 1`) -- not whether any single entry in it has its own
    `questionnaire_path` resolved, since a well root's files can come from several original
    directories, only some of which have one.

    Accepted trade-off (documented, not a bug): a sibling well folder with NO questionnaire of its
    own merges into a 1-name parent above it, exactly like a 0-name data-only directory does --
    e.g. `CustomerA\\WellA\\` (has its own questionnaire) next to `CustomerA\\WellB\\` (a real well,
    but no questionnaire ever reached it) both climb to `CustomerA` once `WellA` is the only name
    contributing to that subtree, merging WellB's files into WellA's review page. This is the same
    mechanism as the intentional 0-name-directory fix above, and it is kept for the same reason:
    the alternative -- stopping the climb at any ancestor with a 0-name child -- is exactly the
    stop-at-0 behavior that caused the original data-loss bug this feature exists to fix, and
    WellB's files would otherwise never surface at all under `require_questionnaire=True`. Visible
    data with a provenance label beats invisible data dropped by the scan: `review_app.py`'s
    per-file subfolder label (`file_subfolder_label`) is the guard that lets a human reviewer see
    WellB's files sitting under a `WellB\\` label on WellA's page and route them correctly, rather
    than the scan silently deciding they belong to WellA.

    A group whose subtree holds MORE than one distinct well name (`n_wells > 1`) is not split
    further here -- FEATURE 1, point 4's "multi-well single folder" case, and the ambiguous
    loose-file-at-a-shared-level case -- it is kept as one group and flagged for a human via
    `n_wells` (`review_app.py` renders the warning) rather than the scan guessing which well each
    file belongs to.

    Dedup is per-group (well root), not global and not per-original-directory: a real corpus case
    is the same raw file copied into two *separate* wells (two distinct tests), and a global dedup
    would wrongly drop the second test's copy as a "duplicate" -- so within one well-root group,
    byte-identical files collapse to one representative (path that sorts first) plus
    `verdict="duplicate"` siblings pointing at it via `dup_of`, REGARDLESS of which of the group's
    several original directories each copy came from (an `OLD Files\\` copy identical to the
    well-root copy now collapses within the one well page, which is the point of this feature);
    across groups, a byte-identical file stays a full, independently-verdicted candidate in every
    group it appears in, and instead gets the other copies' paths recorded in `same_bytes_as`.

    Extraction is still only paid for once per distinct signature: a `dict[sig, FileFeatures]`
    cache means a file whose bytes were already measured (whether the earlier copy was in this
    group or another one) is stamped onto a fresh record (its own path/folder/size_bytes) rather
    than reloaded.

    `progress`, if given, is called `progress(groups_done, total_groups, current_rel)` once per
    group. `limit`, if given, truncates the `rel`-sorted group list to its first N entries
    *before any file is read* -- signatures and features are computed only for the surviving
    groups, so `limit=3` touches exactly 3 groups' files and nothing else. Reproducibility is
    preserved because the sort still happens before the truncation. Consequence: `same_bytes_as`
    (below) can only see the groups inside the limit, so it is complete only for an unlimited
    scan -- a limited scan's `same_bytes_as` is fine for a smoke test but must not be read as a
    real result.
    """
    entries = store.scan_root(root)

    # FIX 5 (perf): one shared walk feeds both quest_paths and dirs, instead of
    # _all_questionnaire_paths/_subtree_well_names each walking the whole tree on their own.
    dirs, quest_paths = _walk_dirs_and_questionnaires(root)
    subtree_names = _subtree_well_names(root, quest_paths, dirs=dirs)

    well_root_cache: dict[str, str] = {}

    def _group_for(folder: str) -> str:
        if folder not in well_root_cache:
            well_root_cache[folder] = _well_root(folder, root, subtree_names)
        return well_root_cache[folder]

    by_group: dict[str, list] = {}
    for e in entries:
        by_group.setdefault(_group_for(e.folder), []).append(e)

    if require_questionnaire:
        by_group = {
            g: es for g, es in by_group.items() if len(subtree_names.get(g, set())) >= 1
        }

    group_rels = {
        g: os.path.relpath(g, root).replace(os.sep, "/") for g in by_group
    }
    ordered_groups = sorted(by_group, key=lambda g: group_rels[g])
    if limit is not None:
        ordered_groups = ordered_groups[:limit]
    by_group = {g: by_group[g] for g in ordered_groups}
    total = len(ordered_groups)

    group_files: dict[str, list[str]] = {}
    for g, es in by_group.items():
        paths: list[str] = []
        for e in es:
            if e.csv_path:
                paths.append(e.csv_path)
            if e.dbs_path:
                paths.append(e.dbs_path)
        group_files[g] = paths

    # Signature every file up front (cheap; see `signature`'s docstring). An unreadable file gets
    # a private, non-colliding pseudo-signature so it never joins a duplicate group -- `extract`
    # will hit the same error on its own and report `load_error`.
    sig_by_path: dict[str, str] = {}
    for paths in group_files.values():
        for p in paths:
            try:
                sig_by_path[p] = signature(p)
            except OSError:
                sig_by_path[p] = f"error:{p}"

    # Cross-group: every path sharing a signature, for `same_bytes_as` (informational only).
    paths_by_sig: dict[str, list[str]] = {}
    for p, sig in sig_by_path.items():
        paths_by_sig.setdefault(sig, []).append(p)

    path_to_group: dict[str, str] = {
        p: g for g, paths in group_files.items() for p in paths
    }

    # In-group duplicates only: within one well-root group's own files (which may span several
    # original directories -- see this function's docstring), group by signature; any group of
    # more than one collapses to the path that sorts first, the rest become `duplicate`.
    dup_of: dict[str, str] = {}
    for g, paths in group_files.items():
        by_sig: dict[str, list[str]] = {}
        for p in paths:
            by_sig.setdefault(sig_by_path[p], []).append(p)
        for sig_group in by_sig.values():
            if len(sig_group) > 1:
                rep = min(sig_group)
                for p in sig_group:
                    if p != rep:
                        dup_of[p] = rep

    feat_cache: dict[str, FileFeatures] = {}

    def _same_bytes_as(path: str, sig: str) -> list[str]:
        # Cross-group boundary is each file's own well-root GROUP, not its immediate directory --
        # now that a group can span several original directories, two copies in different
        # subfolders of the SAME group are an in-group dedup concern (dup_of, above), not this
        # informational cross-group annotation.
        own_group = path_to_group.get(path)
        return sorted(
            p for p in paths_by_sig.get(sig, [])
            if p != path and path_to_group.get(p) != own_group
        )

    def _feat_for(path: str) -> FileFeatures:
        sig = sig_by_path[path]
        try:
            size = os.path.getsize(path)
        except OSError as e:
            # The signature pass and this extraction pass are minutes apart on a full corpus
            # scan; a file that disappears in between (moved/deleted mid-walk) must not abort the
            # whole run -- same "never raises" contract as `extract` itself.
            feat = FileFeatures(
                path=path, folder=os.path.dirname(path), size_bytes=0, sig=sig,
                load_error=f"{type(e).__name__}: {e}", verdict="load_error",
            )
            feat.same_bytes_as = _same_bytes_as(path, sig)
            return feat

        if path in dup_of:
            feat = FileFeatures(
                path=path, folder=os.path.dirname(path), size_bytes=size,
                sig=sig, dup_of=dup_of[path], verdict="duplicate",
            )
        else:
            cached = feat_cache.get(sig)
            if cached is None:
                cached = extract(path)
                feat_cache[sig] = cached
            feat = replace(
                cached, path=path, folder=os.path.dirname(path),
                size_bytes=size, sig=sig, dup_of=None, same_bytes_as=[],
            )
        feat.same_bytes_as = _same_bytes_as(path, sig)
        return feat

    qcache: dict[str, tuple[str, str] | None] = {}

    scans: list[FolderScan] = []
    for i, g in enumerate(ordered_groups):
        rel = group_rels[g]
        n_wells = len(subtree_names.get(g, set()))

        well_name = ""
        formation = ""
        questionnaire_path = _group_questionnaire_path(g, quest_paths)
        if questionnaire_path:
            if questionnaire_path not in qcache:
                # FIX 5 (perf): the naming pass (`_questionnaire_well_names`, inside
                # `_subtree_well_names` above) already parsed every questionnaire in `quest_paths`
                # once, stashing each result in `_LAST_QUESTIONNAIRE_PARSE_CACHE` -- reuse that
                # instead of calling `parse_questionnaire` on the same file a second time. Falls
                # back to a fresh parse when the path isn't in there (e.g.
                # `_questionnaire_well_names` was monkeypatched, so the cache was never
                # populated) -- same "never raises" contract as before either way.
                if questionnaire_path in _LAST_QUESTIONNAIRE_PARSE_CACHE:
                    qres = _LAST_QUESTIONNAIRE_PARSE_CACHE[questionnaire_path]
                    try:
                        qcache[questionnaire_path] = (
                            (qres.well_name or "", qres.formation or "") if qres is not None else None
                        )
                    except Exception:
                        # Same "a bad questionnaire must never abort the scan" contract as the
                        # fresh-parse branch below.
                        qcache[questionnaire_path] = None
                else:
                    try:
                        qres = questionnaire.parse_questionnaire(questionnaire_path)
                        qcache[questionnaire_path] = (qres.well_name or "", qres.formation or "")
                    except Exception:
                        # A bad questionnaire must never abort the scan (same contract as
                        # store.scan_root/load_picks_for).
                        qcache[questionnaire_path] = None
            cached_q = qcache[questionnaire_path]
            if cached_q is not None:
                well_name, formation = cached_q

        files = [_feat_for(p) for p in group_files[g]]

        likely = [f for f in files if f.verdict == "likely_dfit"]
        suggested: list[str] = []
        if likely:
            best = max(likely, key=lambda f: f.post_shutin_hr or 0.0)
            suggested = [best.path]

        scans.append(FolderScan(
            folder=g, rel=rel, well_name=well_name, formation=formation,
            questionnaire_path=questionnaire_path, n_wells=n_wells, files=files,
            suggested=suggested,
        ))

        if progress is not None:
            progress(i + 1, total, rel)

    return scans


# --------------------------------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------------------------------
_FEATURES_FILENAME = "features.json"


def save_scan(root: str, scans: list[FolderScan]) -> str:
    """Write `scans` to `<root>/_triage/features.json` atomically (temp file + retried
    `os.replace`, same contract as `store.save_picks_for`, retry via `atomic.replace_with_retry`
    -- this is the single output of a scan that can take tens of minutes, so an unretried
    transient Windows PermissionError (real-time AV scanning the just-written temp file) must not
    discard it). Returns the path written."""
    out_dir = triage_dir(root)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _FEATURES_FILENAME)
    data = [asdict(s) for s in scans]

    fd, tmp_path = tempfile.mkstemp(dir=out_dir, prefix=".features_", suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        replace_with_retry(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def load_scan(root: str) -> list[FolderScan]:
    """Read back `save_scan`'s output. Raises `FileNotFoundError` if it doesn't exist. Filters
    both dataclasses to known field names on the way in, so a features.json written by a newer
    version of this module (extra keys) never raises here, matching `model._decode`'s contract
    for old/foreign JSON."""
    path = os.path.join(triage_dir(root), _FEATURES_FILENAME)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    file_fields = {f.name for f in fields(FileFeatures)}
    scan_fields = {f.name for f in fields(FolderScan)}

    scans: list[FolderScan] = []
    for item in data:
        files_data = item.get("files", [])
        files = [
            FileFeatures(**{k: v for k, v in fd.items() if k in file_fields})
            for fd in files_data
        ]
        filtered = {k: v for k, v in item.items() if k in scan_fields and k != "files"}
        filtered["files"] = files
        scans.append(FolderScan(**filtered))
    return scans
