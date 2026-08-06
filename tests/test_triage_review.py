"""Headless coverage for scripts/triage/review_app.py's ReviewApp: panel-index <-> file-index
mapping, keep toggling, _seed_keeps precedence, and the navigation/decision rules from the module
docstring (FIX 4: an empty Enter records nothing and does not advance; FIX 5: Left then Right
preserves an in-progress toggle; the index clamp so advancing past the last folder never grows
`index` unboundedly).

`ReviewApp` needs a real `tk.Tk()` to build its widgets, which this headless (no display) suite
can't construct -- same duck-typed stand-in approach as `tests/test_folder_mode.py`: bind the
real `ReviewApp` methods onto a `types.SimpleNamespace` exposing only what each method touches,
never `_redraw` itself (which builds a `FigureCanvasTkAgg`), so it is stubbed out here.

`scripts/` is not a package (mirrors `well_locations.py`/`test_well_locations.py`'s own fixup),
so the repo root and the scripts directory are added to `sys.path` here, matching
`tests/test_triage_apply.py`.
"""

from __future__ import annotations

import os
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from triage import figure, review_app  # noqa: E402
from triage.features import FileFeatures, FolderScan  # noqa: E402
from triage.ledger import Ledger, group_files_sig  # noqa: E402


class _FakeLabel:
    def __init__(self):
        self.text = None

    def config(self, **kw):
        if "text" in kw:
            self.text = kw["text"]


def _make_stub(scans, tmp_path, index=None):
    """A ReviewApp stand-in with the real navigation/decision/toggle methods bound, `_redraw`
    stubbed to a no-op (it is the one method that touches tkinter/matplotlib canvases), and
    `_goto` invoked exactly like `__init__` would (landing on the first undecided folder) unless
    `index` is given."""
    stub = types.SimpleNamespace()
    stub.scans = scans
    stub.order = [s.rel for s in scans]
    stub.ledger = Ledger.load(str(tmp_path))
    # FIX 2: same fingerprint map the real ReviewApp.__init__ computes, so _seed_keeps/
    # first_undecided/progress exercise their fingerprint-aware branches here too.
    stub._files_sig_by_rel = {s.rel: group_files_sig(f.sig for f in s.files) for s in scans}
    stub.page = 0
    stub.keeps = set()
    stub._inprogress = {}
    stub.status_lbl = _FakeLabel()

    stub._current_scan = types.MethodType(review_app.ReviewApp._current_scan, stub)
    stub._seed_keeps = types.MethodType(review_app.ReviewApp._seed_keeps, stub)
    stub._redraw = lambda: None
    stub._goto = types.MethodType(review_app.ReviewApp._goto, stub)
    stub._file_index = types.MethodType(review_app.ReviewApp._file_index, stub)
    stub._toggle_keep = types.MethodType(review_app.ReviewApp._toggle_keep, stub)
    stub._commit_and_advance = types.MethodType(review_app.ReviewApp._commit_and_advance, stub)
    stub._mark_none_and_advance = types.MethodType(review_app.ReviewApp._mark_none_and_advance, stub)
    stub._mark_unsure_and_advance = types.MethodType(
        review_app.ReviewApp._mark_unsure_and_advance, stub)
    stub._go_back = types.MethodType(review_app.ReviewApp._go_back, stub)

    stub._goto(
        index if index is not None
        else stub.ledger.first_undecided(stub.order, stub._files_sig_by_rel)
    )
    return stub


def _files(folder, n, prefix="f"):
    return [
        FileFeatures(path=f"{folder}/{prefix}{i}.dbs", folder=folder, size_bytes=1, sig=str(i))
        for i in range(n)
    ]


# --------------------------------------------------------------------------------------------------
# panel index -> file index across pages
# --------------------------------------------------------------------------------------------------
def test_toggle_keep_maps_panel_index_to_file_index_on_second_page(tmp_path):
    files = _files("/f", 16)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)
    stub.page = 1  # second page, per_page (PER_PAGE) == 8

    stub._toggle_keep(1)
    assert files[8].path in stub.keeps

    stub._toggle_keep(8)
    assert files[15].path in stub.keeps


def test_toggle_keep_past_end_of_file_list_is_ignored(tmp_path):
    files = _files("/f", 10)  # only indices 0-9 exist
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)
    stub.page = 1  # page*PER_PAGE + (key_n - 1) = 8 + (key_n - 1)

    before = set(stub.keeps)
    stub._toggle_keep(3)  # index 10 -- one past the last valid index (9)
    assert stub.keeps == before  # ignored: no crash, no change


# --------------------------------------------------------------------------------------------------
# keep-toggle on/off round trip
# --------------------------------------------------------------------------------------------------
def test_toggle_keep_on_then_off_round_trips(tmp_path):
    files = _files("/f", 3)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)

    stub._toggle_keep(1)
    assert files[0].path in stub.keeps

    stub._toggle_keep(1)
    assert files[0].path not in stub.keeps


# --------------------------------------------------------------------------------------------------
# _seed_keeps precedence: ledger decision > cached in-progress > scan.suggested
# --------------------------------------------------------------------------------------------------
def test_seed_keeps_precedence(tmp_path):
    files = _files("/f", 3)
    scan = FolderScan(folder="/f", rel="C/f", files=files, suggested=[files[0].path])
    stub = _make_stub([scan], tmp_path, index=0)

    # Nothing cached, no ledger decision -> scan.suggested wins.
    stub._seed_keeps(scan)
    assert stub.keeps == {files[0].path}

    # An in-progress cache for this index, still no ledger decision -> cache wins over suggested.
    stub._inprogress[stub.index] = {files[1].path}
    stub._seed_keeps(scan)
    assert stub.keeps == {files[1].path}

    # A ledger decision -> wins over both the cache and the suggestion (FIX 2: it must be
    # CURRENT, i.e. recorded with this group's fingerprint, or it would resurface as undecided).
    stub.ledger.set(scan.rel, [files[2].path], "decided",
                     files_sig=stub._files_sig_by_rel[scan.rel])
    stub._seed_keeps(scan)
    assert stub.keeps == {files[2].path}


# --------------------------------------------------------------------------------------------------
# FIX 2: a stale (fingerprint-mismatched) or legacy (no files_sig) ledger decision must not win
# over the in-progress cache/suggestion -- it resurfaces exactly like an undecided folder.
# --------------------------------------------------------------------------------------------------
def test_seed_keeps_stale_decision_falls_through_to_suggested(tmp_path):
    files = _files("/f", 3)
    scan = FolderScan(folder="/f", rel="C/f", files=files, suggested=[files[0].path])
    stub = _make_stub([scan], tmp_path, index=0)

    # Recorded against a DIFFERENT (smaller) file set than this group's current one.
    stub.ledger.set(scan.rel, [files[2].path], "decided", files_sig="stale-fingerprint")
    stub._seed_keeps(scan)
    assert stub.keeps == {files[0].path}  # falls through to scan.suggested, not the stale decision


def test_seed_keeps_legacy_decision_no_files_sig_falls_through_to_suggested(tmp_path):
    files = _files("/f", 3)
    scan = FolderScan(folder="/f", rel="C/f", files=files, suggested=[files[0].path])
    stub = _make_stub([scan], tmp_path, index=0)

    stub.ledger.set(scan.rel, [files[2].path], "decided")  # no files_sig -- legacy
    stub._seed_keeps(scan)
    assert stub.keeps == {files[0].path}


# --------------------------------------------------------------------------------------------------
# FIX 4: Enter/Right with an empty keep set records nothing and does not advance.
# --------------------------------------------------------------------------------------------------
def test_commit_and_advance_empty_keeps_is_noop(tmp_path):
    files = _files("/f", 2)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)
    stub.keeps = set()

    stub._commit_and_advance()

    assert stub.index == 0  # did not advance
    assert stub.ledger.get(scan.rel).status == ""  # nothing recorded
    assert "0" in stub.status_lbl.text  # points the analyst at 0/u


def test_commit_and_advance_nonempty_keeps_commits_and_advances(tmp_path):
    files = _files("/f", 2)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)
    stub.keeps = {files[0].path}

    stub._commit_and_advance()

    assert stub.index == 1
    decision = stub.ledger.get(scan.rel)
    assert decision.status == "decided"
    assert decision.keeps == [files[0].path]


# --------------------------------------------------------------------------------------------------
# FIX 5: toggle, Left, Right preserves the toggle instead of re-seeding from scan.suggested.
# --------------------------------------------------------------------------------------------------
def test_go_back_then_forward_preserves_inprogress_toggle(tmp_path):
    file_a = _files("/A", 1, prefix="x")[0]
    scan_a = FolderScan(folder="/A", rel="A", files=[file_a])

    files_b = _files("/B", 2)
    scan_b = FolderScan(folder="/B", rel="B", files=files_b, suggested=[files_b[1].path])

    stub = _make_stub([scan_a, scan_b], tmp_path, index=0)
    # A already has a prior decision (simulates a previous review pass), so committing it again
    # while walking back through it is a legitimate no-op re-commit, not a fresh empty commit.
    # Recorded with A's current fingerprint (FIX 2) so it reads back as CURRENT, not stale.
    stub.ledger.set(scan_a.rel, [file_a.path], "decided",
                     files_sig=stub._files_sig_by_rel[scan_a.rel])

    stub._goto(1)  # land on B; seeded from scan.suggested
    assert stub.keeps == {files_b[1].path}

    stub._toggle_keep(1)  # toggle files_b[0] on -> {files_b[0], files_b[1]}
    assert stub.keeps == {files_b[0].path, files_b[1].path}

    stub._go_back()  # -> A, caching B's in-progress {files_b[0], files_b[1]}
    assert stub.index == 0
    assert stub.keeps == {file_a.path}  # A's own ledger decision, unaffected

    stub._commit_and_advance()  # re-commit A (unchanged), advance back to B
    assert stub.index == 1
    # B must come back with the toggle intact, not re-seeded down to just scan.suggested.
    assert stub.keeps == {files_b[0].path, files_b[1].path}


# --------------------------------------------------------------------------------------------------
# advancing past the last folder does not crash and does not leave `index` unbounded.
# --------------------------------------------------------------------------------------------------
def test_advancing_past_last_folder_clamps_index(tmp_path):
    files = _files("/f", 1)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)
    stub.keeps = {files[0].path}

    for _ in range(5):
        stub._commit_and_advance()

    assert stub.index == len(stub.scans)  # clamped at the "queue complete" sentinel, not beyond
    assert stub._current_scan() is None  # past-the-end contract still holds


def test_advancing_past_last_folder_via_mark_none_also_clamps_index(tmp_path):
    """`0`/`u` unconditionally call `_goto(self.index + 1)` even once already past the end --
    the clamp lives in `_goto` itself, so every advance path is covered, not just Enter."""
    files = _files("/f", 1)
    scan = FolderScan(folder="/f", rel="C/f", files=files)
    stub = _make_stub([scan], tmp_path, index=0)

    for _ in range(5):
        stub._mark_none_and_advance()

    assert stub.index == len(stub.scans)


# --------------------------------------------------------------------------------------------------
# FEATURE 1: multi-well-folder warning text (review_app._multi_well_warning_text)
# --------------------------------------------------------------------------------------------------
def test_multi_well_warning_text_empty_when_unambiguous():
    assert review_app._multi_well_warning_text(1) == ""
    assert review_app._multi_well_warning_text(0) == ""


def test_multi_well_warning_text_flags_ambiguous_group():
    text = review_app._multi_well_warning_text(3)
    assert "MULTI-WELL FOLDER" in text
    assert "3" in text
    assert "split manually" in text


# --------------------------------------------------------------------------------------------------
# FEATURE 2: per-file subfolder provenance (review_app.file_subfolder_label,
# review_app._annotate_file_subfolders)
# --------------------------------------------------------------------------------------------------
def test_file_subfolder_label_root_file_is_empty():
    """A file directly in the well-root group folder gets no label at all -- never the literal
    "." `os.path.relpath` would otherwise return."""
    assert review_app.file_subfolder_label(os.path.join("root", "well"), os.path.join("root", "well")) == ""


def test_file_subfolder_label_one_level_down():
    group = os.path.join("root", "well")
    file_folder = os.path.join("root", "well", "OLD Files")
    assert review_app.file_subfolder_label(group, file_folder) == "OLD Files"


def test_file_subfolder_label_two_levels_down():
    group = os.path.join("root", "well")
    file_folder = os.path.join("root", "well", "Raw Data", "CSVs")
    assert review_app.file_subfolder_label(group, file_folder) == os.path.join("Raw Data", "CSVs")


def test_annotate_file_subfolders_labels_nested_file_not_root_file(tmp_path):
    """Review-app-level check that the label text reaches the rendered widget metadata: build a
    real `Figure` via `figure.render_grid` (headless-safe, Agg backend) for a well-root group
    mixing a root file and a nested one, then confirm `_annotate_file_subfolders` adds the
    subfolder text to the nested file's panel and nothing to the root file's panel."""
    group_folder = str(tmp_path / "well")
    old_files_folder = str(tmp_path / "well" / "OLD Files")
    root_file = FileFeatures(
        path=os.path.join(group_folder, "root.csv"), folder=group_folder,
        size_bytes=1, sig="1",
    )
    nested_file = FileFeatures(
        path=os.path.join(old_files_folder, "old.csv"), folder=old_files_folder,
        size_bytes=1, sig="2",
    )
    scan = FolderScan(folder=group_folder, rel="Well1", files=[root_file, nested_file])

    fig = figure.render_grid(scan, str(tmp_path), page=0, per_page=8)
    review_app._annotate_file_subfolders(fig, scan, page=0, per_page=8)

    visible = [ax for ax in fig.axes if ax.axison]
    root_ax, nested_ax = visible[0], visible[1]

    root_texts = [t.get_text() for t in root_ax.texts]
    nested_texts = [t.get_text() for t in nested_ax.texts]

    assert not any("OLD Files" in t for t in root_texts)
    assert any("OLD Files" in t for t in nested_texts)


def test_annotate_file_subfolders_respects_pagination(tmp_path):
    """The label must be attached to the panel matching this PAGE's slice of `scan.files`, not
    always index 0 -- exercises `_annotate_file_subfolders` on page 1 of a 9-file scan."""
    group_folder = str(tmp_path / "well")
    nested_folder = str(tmp_path / "well" / "OLD Files")
    files = [
        FileFeatures(path=os.path.join(group_folder, f"f{i}.csv"), folder=group_folder,
                     size_bytes=1, sig=str(i))
        for i in range(8)
    ]
    files.append(FileFeatures(
        path=os.path.join(nested_folder, "old.csv"), folder=nested_folder,
        size_bytes=1, sig="8",
    ))
    scan = FolderScan(folder=group_folder, rel="Well1", files=files)

    fig = figure.render_grid(scan, str(tmp_path), page=1, per_page=8)
    review_app._annotate_file_subfolders(fig, scan, page=1, per_page=8)

    visible = [ax for ax in fig.axes if ax.axison]
    assert len(visible) == 1  # 9 files, 8 per page -> 1 on page 2
    texts = [t.get_text() for t in visible[0].texts]
    assert any("OLD Files" in t for t in texts)
