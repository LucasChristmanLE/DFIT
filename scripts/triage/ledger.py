"""The triage review decisions ledger: one `FolderDecision` per folder, keyed by `rel`
(`FolderScan.rel`), atomically persisted to `<root>/_triage/decisions.json`.

Written after every folder decision during `review`, so quitting mid-way and relaunching
resumes at the first undecided folder (`Ledger.first_undecided`). Pure Python -- no Tkinter, no
matplotlib -- so it is unit-testable headless like `dfit_tool/store.py`'s picks persistence,
whose atomic-write contract (temp file + `os.replace`) this mirrors.

FIX 2 (decision fingerprint guard): a decision is keyed on `rel`, but `rel` is a GROUP identity,
not a fixed set of files -- `scripts/triage/features.py`'s well-root grouping can, across a
re-scan whose grouping semantics changed, resolve the same `rel` to a bigger (or smaller, or just
different) set of files than the one a human actually reviewed when the decision was made. Left
unguarded, a surviving `rel` would silently apply a stale decision to files nobody ever looked at:
kept files stay kept, but every newly-merged file gets quarantined (or, worse, filed as a "keep"
alongside files it was never grouped with) without a human ever seeing it. `FolderDecision.
files_sig` (see `group_files_sig`) is the guard: a fingerprint of the exact file set a decision
was made against, recorded alongside it, so a later reader can tell whether `rel`'s CURRENT file
set is still the one that decision covers -- see `Ledger.status_if_current`, used by `review_app.
py` (a mismatched decision resurfaces for review) and `apply.plan_moves`/`plan_warnings` (a
mismatched decision is skipped with a warning, never silently applied). A decision recorded
before this fingerprint existed has `files_sig == ""`, which can never equal a real fingerprint
(see `group_files_sig`'s docstring), so it is treated exactly like a mismatch -- stale-undecided,
never silently trusted.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import tempfile
import time  # noqa: F401 -- unused directly, kept so tests can monkeypatch it; see note below
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields

from .atomic import REPLACE_ATTEMPTS, REPLACE_BACKOFF_S, replace_with_retry

_LEDGER_FILENAME = "decisions.json"

# The atomic-replace-with-retry logic itself now lives in `atomic.py`, shared with
# `features.save_scan`. These private names are kept as aliases (rather than deleted) purely so
# existing callers/tests that reach into this module by its old private names keep working. The
# `import time` above is likewise kept only so `tests/test_triage_ledger.py`'s
# `monkeypatch.setattr("triage.ledger.time.sleep", ...)` still resolves -- `time` (here) and
# `atomic.time` are the same stdlib module object, so patching either patches the retry loop that
# `atomic.replace_with_retry` actually runs.
_replace_with_retry = replace_with_retry
_REPLACE_ATTEMPTS = REPLACE_ATTEMPTS
_REPLACE_BACKOFF_S = REPLACE_BACKOFF_S


@dataclass
class FolderDecision:
    rel: str
    keeps: list[str] = field(default_factory=list)   # absolute paths of files to keep
    status: str = ""                                 # "decided" | "none" | "unsure" | "" (undecided)
    ts: str = ""                                      # ISO 8601, set on write
    # FIX 2: fingerprint of the file set this decision was made against (see `group_files_sig`).
    # "" for a decision recorded before this field existed (legacy) -- never a real fingerprint's
    # value, so a legacy decision always reads as a mismatch, never as accidentally "current".
    files_sig: str = ""


def _ledger_path(root: str) -> str:
    return os.path.join(root, "_triage", _LEDGER_FILENAME)


def group_files_sig(sigs: Iterable[str]) -> str:
    """The fingerprint recorded alongside a decision (FIX 2): a stable hash of `sigs` -- each
    file's own content signature, `FileFeatures.sig` -- order-independent (sorted before hashing),
    so the same set of files always fingerprints the same regardless of iteration order. A group
    that gains, loses, or swaps even one file's signature produces a different fingerprint, which
    is exactly the property `review_app.py`/`apply.plan_moves` need to detect that a decision was
    made against a DIFFERENT set of files than the one on screen (or being planned) now.

    Never returns `""`, even for an empty `sigs` (SHA-256 of anything, including the empty
    string, is a 64-character hex digest, never `""`), so `FolderDecision.files_sig == ""`
    unambiguously means "no fingerprint was ever recorded" (a legacy decision), not "fingerprint
    of an empty file set" -- `status_if_current` relies on that distinction to treat a legacy
    decision as stale rather than accidentally matching a genuinely empty-file group."""
    return hashlib.sha256("\n".join(sorted(sigs)).encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, path: str, decisions: dict[str, FolderDecision] | None = None) -> None:
        self.path = path
        self.decisions: dict[str, FolderDecision] = decisions if decisions is not None else {}

    # ---- persistence ----
    @classmethod
    def load(cls, root: str) -> "Ledger":
        """The ledger at `<root>/_triage/decisions.json`, or an empty one if it doesn't exist
        yet. Unknown keys in a saved decision are filtered out (matches `model._decode`'s
        old/foreign-JSON-never-raises contract), so a ledger written by a newer version of this
        module never breaks an older reader."""
        path = _ledger_path(root)
        if not os.path.exists(path):
            return cls(path=path)
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        known = {f.name for f in fields(FolderDecision)}
        decisions = {
            rel: FolderDecision(**{k: v for k, v in d.items() if k in known})
            for rel, d in data.items()
        }
        return cls(path=path, decisions=decisions)

    def save(self) -> None:
        """Atomic write: a temp file in the same directory, then `os.replace` onto the final
        path -- same contract as `store.save_picks_for`, so a crash mid-write never leaves a
        half-written ledger behind."""
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        data = {rel: asdict(d) for rel, d in self.decisions.items()}
        fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=".decisions_", suffix=".tmp")
        os.close(fd)
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
            _replace_with_retry(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    # ---- decisions ----
    def set(self, rel: str, keeps: list[str], status: str, files_sig: str = "") -> None:
        """Record the decision for `rel`, stamp it with the current time, and save immediately
        (review writes after every folder, not just on quit). `files_sig` (FIX 2), if given, is
        the fingerprint (`group_files_sig`) of the exact file set this decision was made against
        -- callers that care about the staleness guard (`review_app.py`) always pass it; the
        default `""` exists only for callers that don't (older direct calls, and this module's own
        tests that aren't exercising the fingerprint guard), and reads back exactly like a legacy
        pre-fingerprint decision -- stale, per `status_if_current`."""
        ts = datetime.datetime.now().isoformat(timespec="seconds")
        self.decisions[rel] = FolderDecision(
            rel=rel, keeps=list(keeps), status=status, ts=ts, files_sig=files_sig,
        )
        self.save()

    def get(self, rel: str) -> FolderDecision:
        """The decision for `rel`, or a blank undecided one if `rel` has never been decided. This
        is the RAW recorded decision, unaware of whether it is still current for today's file set
        -- see `status_if_current` for the fingerprint-aware version FIX 2 needs."""
        return self.decisions.get(rel, FolderDecision(rel=rel))

    def status_if_current(self, rel: str, files_sig: str) -> str:
        """`rel`'s decision status, but only when its recorded `files_sig` matches `files_sig` --
        `""` (undecided) otherwise, whether because there is no decision at all, the decision
        predates this fingerprint (legacy, `files_sig == ""`, which can never equal a real
        fingerprint), or the group's file set has genuinely changed since the decision was made
        (FIX 2: a re-scan whose grouping semantics changed). The stale decision itself is left
        completely untouched in `self.decisions` -- only the status reported here degrades to
        undecided; only a fresh `set()` call ever overwrites it."""
        d = self.decisions.get(rel)
        if d is None or not d.files_sig or d.files_sig != files_sig:
            return ""
        return d.status

    def first_undecided(
        self, order: list[str], files_sig_by_rel: dict[str, str] | None = None
    ) -> int:
        """Index into `order` of the first folder with no recorded (and, when `files_sig_by_rel`
        is given, still-current -- FIX 2) status, or `len(order)` if every folder in `order` has
        one. `files_sig_by_rel`, if given, maps each `rel` to its group's CURRENT fingerprint
        (`group_files_sig`); a decision whose own `files_sig` no longer matches that current
        fingerprint is treated as undecided, same as if it had never been made. Omitting
        `files_sig_by_rel` (the default) falls back to the plain pre-FIX-2 status check -- for
        callers with no scans on hand to fingerprint (this module's own status-only tests)."""
        for i, rel in enumerate(order):
            if files_sig_by_rel is not None:
                status = self.status_if_current(rel, files_sig_by_rel.get(rel, ""))
            else:
                d = self.decisions.get(rel)
                status = d.status if d is not None else ""
            if not status:
                return i
        return len(order)

    def progress(
        self, order: list[str], files_sig_by_rel: dict[str, str] | None = None
    ) -> tuple[int, int]:
        """`(decided_count, total)` over `order` -- a folder counts as decided once it has any
        non-empty (and, when `files_sig_by_rel` is given, still-current -- FIX 2) status
        (including "unsure": it's been seen, even if flagged to revisit). See `first_undecided`
        for `files_sig_by_rel`'s contract; omitting it falls back to the plain pre-FIX-2 count."""
        decided = 0
        for rel in order:
            if files_sig_by_rel is not None:
                status = self.status_if_current(rel, files_sig_by_rel.get(rel, ""))
            else:
                d = self.decisions.get(rel)
                status = d.status if d is not None else ""
            if status:
                decided += 1
        return decided, len(order)
