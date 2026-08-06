"""Unit tests for scripts/triage/ledger.py: round-trip, resume, progress, and the atomic-write
contract. Headless -- no data files, no Tkinter -- and never touches C:\\DFIT Data.

`scripts/` is not a package (mirrors tests/test_well_locations.py's fixup), so the repo root and
the scripts directory are added to sys.path here.
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

from triage.ledger import (  # noqa: E402
    FolderDecision, Ledger, group_files_sig, _replace_with_retry, _REPLACE_ATTEMPTS,
)


# --------------------------------------------------------------------------------------------------
# load() on an absent ledger
# --------------------------------------------------------------------------------------------------
def test_load_absent_ledger_is_empty(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    assert ledger.decisions == {}
    assert ledger.path == os.path.join(str(tmp_path), "_triage", "decisions.json")


# --------------------------------------------------------------------------------------------------
# set / get / round trip
# --------------------------------------------------------------------------------------------------
def test_get_unseen_folder_is_blank_undecided(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    d = ledger.get("CustomerA/Well1")
    assert d == FolderDecision(rel="CustomerA/Well1")
    assert d.status == ""
    assert d.keeps == []


def test_set_then_get_records_decision(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    ledger.set("CustomerA/Well1", keeps=["/data/a.csv", "/data/b.dbs"], status="decided")

    d = ledger.get("CustomerA/Well1")
    assert d.status == "decided"
    assert d.keeps == ["/data/a.csv", "/data/b.dbs"]
    assert d.ts  # stamped on write


def test_set_then_reload_from_disk_shows_the_decision(tmp_path):
    root = str(tmp_path)
    ledger = Ledger.load(root)
    ledger.set("CustomerA/Well1", keeps=["/data/a.csv"], status="decided")

    reloaded = Ledger.load(root)
    d = reloaded.get("CustomerA/Well1")
    assert d.status == "decided"
    assert d.keeps == ["/data/a.csv"]
    assert d.ts == ledger.get("CustomerA/Well1").ts


def test_load_ignores_unknown_key(tmp_path):
    """A decisions.json written by a newer version of this module (extra keys) must never break
    an older reader -- same "old/foreign JSON never raises" contract as model._decode."""
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "_triage"), exist_ok=True)
    path = os.path.join(root, "_triage", "decisions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "CustomerA/Well1": {
                "rel": "CustomerA/Well1", "keeps": ["/data/a.csv"], "status": "decided",
                "ts": "2026-01-01T00:00:00", "future_field": "surprise",
            }
        }, fh)

    ledger = Ledger.load(root)
    d = ledger.get("CustomerA/Well1")
    assert d.status == "decided"
    assert d.keeps == ["/data/a.csv"]
    assert not hasattr(d, "future_field")


# --------------------------------------------------------------------------------------------------
# first_undecided / progress
# --------------------------------------------------------------------------------------------------
def test_first_undecided_partial_ledger():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2", "B/1", "B/2"]
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided")
    ledger.decisions["A/2"] = FolderDecision(rel="A/2", status="none")

    assert ledger.first_undecided(order) == 2  # "B/1" is the first with no recorded status


def test_first_undecided_all_decided_returns_len():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    for rel in order:
        ledger.decisions[rel] = FolderDecision(rel=rel, status="decided")

    assert ledger.first_undecided(order) == len(order)


def test_first_undecided_empty_ledger_returns_zero():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    assert ledger.first_undecided(order) == 0


def test_first_undecided_unsure_counts_as_seen():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="unsure")
    assert ledger.first_undecided(order) == 1


def test_progress_counts_correctly():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2", "B/1"]
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided")
    ledger.decisions["A/2"] = FolderDecision(rel="A/2", status="unsure")

    assert ledger.progress(order) == (2, 3)


def test_progress_empty_ledger():
    ledger = Ledger(path="unused")
    assert ledger.progress(["A/1", "A/2"]) == (0, 2)


def test_progress_all_decided():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    for rel in order:
        ledger.decisions[rel] = FolderDecision(rel=rel, status="decided")
    assert ledger.progress(order) == (2, 2)


# --------------------------------------------------------------------------------------------------
# FIX 2: decision fingerprint guard -- group_files_sig, files_sig on set/get, status_if_current
# --------------------------------------------------------------------------------------------------
def test_group_files_sig_order_independent():
    assert group_files_sig(["a", "b", "c"]) == group_files_sig(["c", "a", "b"])


def test_group_files_sig_changes_when_file_set_changes():
    """A group that gains a file's signature fingerprints differently -- the whole point of the
    guard: this is exactly what tells `status_if_current`/`apply.plan_moves` that a decision was
    made against a smaller set of files than the one on screen/being planned now."""
    before = group_files_sig(["a", "b"])
    after = group_files_sig(["a", "b", "c"])
    assert before != after


def test_group_files_sig_never_empty_string_even_for_empty_input():
    """`files_sig == ""` must unambiguously mean "never recorded" (legacy) -- never "fingerprint
    of a genuinely empty file set" -- so a real fingerprint (of any input, including none) must
    never itself be `""`."""
    assert group_files_sig([]) != ""


def test_set_records_files_sig_get_returns_it_raw(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    sig = group_files_sig(["a", "b"])
    ledger.set("A/1", keeps=["/data/a.csv"], status="decided", files_sig=sig)

    d = ledger.get("A/1")
    assert d.files_sig == sig


def test_set_without_files_sig_defaults_to_empty_string(tmp_path):
    """The default (no `files_sig` passed) reads back exactly like a legacy decision -- `""`, not
    some other placeholder -- so it is treated as stale by `status_if_current` just like a
    decision written before this field existed."""
    ledger = Ledger.load(str(tmp_path))
    ledger.set("A/1", keeps=[], status="decided")

    assert ledger.get("A/1").files_sig == ""


def test_status_if_current_matching_fingerprint_returns_status(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    sig = group_files_sig(["a", "b"])
    ledger.set("A/1", keeps=["/data/a.csv"], status="decided", files_sig=sig)

    assert ledger.status_if_current("A/1", sig) == "decided"


def test_status_if_current_mismatched_fingerprint_returns_undecided(tmp_path):
    """FIX 2: the group's file set changed since the decision was made -- the decision itself
    must NOT be silently trusted; it reads back as undecided."""
    ledger = Ledger.load(str(tmp_path))
    old_sig = group_files_sig(["a", "b"])
    new_sig = group_files_sig(["a", "b", "c"])
    ledger.set("A/1", keeps=["/data/a.csv"], status="decided", files_sig=old_sig)

    assert ledger.status_if_current("A/1", new_sig) == ""
    # The stale decision is preserved untouched, not deleted -- only the reported status degrades.
    assert ledger.get("A/1").status == "decided"
    assert ledger.get("A/1").files_sig == old_sig


def test_status_if_current_legacy_no_files_sig_returns_undecided(tmp_path):
    """A decision recorded before this fingerprint existed (`files_sig == ""`) must be treated
    the same as a genuine mismatch -- stale, never accidentally "current" just because the
    caller's current fingerprint check happens to short-circuit some other way."""
    ledger = Ledger.load(str(tmp_path))
    ledger.set("A/1", keeps=["/data/a.csv"], status="decided")  # no files_sig -> legacy

    assert ledger.status_if_current("A/1", group_files_sig(["a"])) == ""


def test_status_if_current_no_decision_returns_undecided(tmp_path):
    ledger = Ledger.load(str(tmp_path))
    assert ledger.status_if_current("A/1", group_files_sig(["a"])) == ""


# --------------------------------------------------------------------------------------------------
# FIX 2: first_undecided / progress with files_sig_by_rel (fingerprint-aware)
# --------------------------------------------------------------------------------------------------
def test_first_undecided_stale_decision_resurfaces_as_undecided():
    """A decided-but-stale folder must resurface as the first undecided one when a
    `files_sig_by_rel` mapping is supplied -- FIX 2's "resurfaces for review" behavior."""
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    old_sig = group_files_sig(["a", "b"])
    new_sig = group_files_sig(["a", "b", "c"])  # A/1's group grew since the decision
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided", files_sig=old_sig)

    assert ledger.first_undecided(order, files_sig_by_rel={"A/1": new_sig, "A/2": "irrelevant"}) == 0


def test_first_undecided_current_decision_is_not_first_undecided():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    sig = group_files_sig(["a", "b"])
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided", files_sig=sig)

    assert ledger.first_undecided(order, files_sig_by_rel={"A/1": sig, "A/2": "x"}) == 1


def test_first_undecided_without_files_sig_by_rel_falls_back_to_plain_status():
    """Omitting `files_sig_by_rel` entirely preserves the original pre-FIX-2 behavior exactly --
    a plain status check, no fingerprint awareness at all."""
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided")  # no files_sig

    assert ledger.first_undecided(order) == 1  # A/1 counts as decided -- no fingerprint checked


def test_progress_stale_decision_does_not_count_as_decided():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    old_sig = group_files_sig(["a"])
    new_sig = group_files_sig(["a", "b"])
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided", files_sig=old_sig)
    ledger.decisions["A/2"] = FolderDecision(rel="A/2", status="decided", files_sig=new_sig)

    decided, total = ledger.progress(order, files_sig_by_rel={"A/1": new_sig, "A/2": new_sig})
    assert (decided, total) == (1, 2)  # only A/2's fingerprint still matches


def test_progress_without_files_sig_by_rel_falls_back_to_plain_status():
    ledger = Ledger(path="unused")
    order = ["A/1", "A/2"]
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided")

    assert ledger.progress(order) == (1, 2)


# --------------------------------------------------------------------------------------------------
# FIX 2: legacy decisions.json (no files_sig key at all) loads with files_sig defaulting to ""
# --------------------------------------------------------------------------------------------------
def test_load_legacy_decision_with_no_files_sig_key_defaults_to_empty_string(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, "_triage"), exist_ok=True)
    path = os.path.join(root, "_triage", "decisions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "A/1": {"rel": "A/1", "keeps": ["/data/a.csv"], "status": "decided",
                     "ts": "2026-01-01T00:00:00"},  # no "files_sig" key at all
        }, fh)

    ledger = Ledger.load(root)
    d = ledger.get("A/1")
    assert d.status == "decided"
    assert d.files_sig == ""


# --------------------------------------------------------------------------------------------------
# atomic write
# --------------------------------------------------------------------------------------------------
def test_save_is_atomic_no_tmp_leftover_and_valid_json(tmp_path):
    root = str(tmp_path)
    ledger = Ledger.load(root)
    ledger.set("CustomerA/Well1", keeps=["/data/a.csv"], status="decided")

    triage_dir = os.path.join(root, "_triage")
    names = os.listdir(triage_dir)
    assert names == ["decisions.json"]  # no .tmp files left behind

    with open(os.path.join(triage_dir, "decisions.json"), encoding="utf-8") as fh:
        data = json.load(fh)  # raises if not valid JSON
    assert data["CustomerA/Well1"]["status"] == "decided"


def test_save_multiple_times_stays_atomic(tmp_path):
    root = str(tmp_path)
    ledger = Ledger.load(root)
    ledger.set("A/1", keeps=[], status="decided")
    ledger.set("A/2", keeps=[], status="none")
    ledger.set("A/1", keeps=["/data/x.csv"], status="unsure")

    triage_dir = os.path.join(root, "_triage")
    assert os.listdir(triage_dir) == ["decisions.json"]

    reloaded = Ledger.load(root)
    assert reloaded.get("A/1").status == "unsure"
    assert reloaded.get("A/1").keeps == ["/data/x.csv"]
    assert reloaded.get("A/2").status == "none"


# --------------------------------------------------------------------------------------------------
# _replace_with_retry -- transient Windows PermissionError on os.replace
# --------------------------------------------------------------------------------------------------
def test_replace_with_retry_succeeds_after_transient_failures(tmp_path, monkeypatch):
    monkeypatch.setattr("triage.ledger.time.sleep", lambda _s: None)
    src = os.path.join(str(tmp_path), "src.tmp")
    dst = os.path.join(str(tmp_path), "dst.txt")
    with open(src, "w", encoding="utf-8") as fh:
        fh.write("payload")

    real_replace = os.replace
    calls = {"count": 0}

    def flaky_replace(s, d):
        calls["count"] += 1
        if calls["count"] < 3:
            raise PermissionError("[WinError 5] Access is denied")
        real_replace(s, d)

    monkeypatch.setattr("triage.ledger.os.replace", flaky_replace)
    _replace_with_retry(src, dst)

    assert calls["count"] == 3
    assert os.path.exists(dst)
    with open(dst, encoding="utf-8") as fh:
        assert fh.read() == "payload"


def test_replace_with_retry_reraises_after_exhausting_attempts(tmp_path, monkeypatch):
    monkeypatch.setattr("triage.ledger.time.sleep", lambda _s: None)
    calls = {"count": 0}

    def always_fails(s, d):
        calls["count"] += 1
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr("triage.ledger.os.replace", always_fails)

    with pytest.raises(PermissionError):
        _replace_with_retry("src", "dst")

    assert calls["count"] == _REPLACE_ATTEMPTS


def test_replace_with_retry_non_permission_error_is_not_retried(monkeypatch):
    monkeypatch.setattr("triage.ledger.time.sleep", lambda _s: None)
    calls = {"count": 0}

    def raises_os_error(s, d):
        calls["count"] += 1
        raise OSError("some other failure")

    monkeypatch.setattr("triage.ledger.os.replace", raises_os_error)

    with pytest.raises(OSError):
        _replace_with_retry("src", "dst")

    assert calls["count"] == 1


def test_save_cleans_up_tmp_file_when_replace_ultimately_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("triage.ledger.time.sleep", lambda _s: None)
    root = str(tmp_path)
    ledger = Ledger.load(root)
    ledger.decisions["A/1"] = FolderDecision(rel="A/1", status="decided")

    def always_fails(s, d):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr("triage.ledger.os.replace", always_fails)

    with pytest.raises(PermissionError):
        ledger.save()

    triage_dir = os.path.join(root, "_triage")
    leftover_tmp = [n for n in os.listdir(triage_dir) if n.endswith(".tmp")]
    assert leftover_tmp == []
