"""Phase 2 of DFIT triage: the Tkinter review window. The only file in `scripts/triage/` that
imports tkinter -- `apply.py`, `features.py`, `figure.py`, `ledger.py`, and `basins.py` all stay
headless, matching how `dfit_tool/picks.py`/`plots.py` stay Tk-free per `CLAUDE.md`.

One folder per screen: a header with progress, the folder's path, its questionnaire well
name/formation, and how many folders are decided so far; a matplotlib grid of that folder's
files (`triage.figure.render_grid`), paginated at up to `PER_PAGE` panels; a status line with the
current keep set and the page indicator, and a static key legend below it. Every decision writes
through `Ledger.set`, which persists atomically per the ledger's own contract, so quitting
mid-review (`q`) loses nothing beyond the current, uncommitted screen.

Two navigation/decision rules worth stating explicitly, since they are easy to get wrong:

- `Enter`/`Right` with an empty keep set is a no-op, not a "none" verdict -- `"none"` ("no DFIT
  here") is `0`'s job alone, so a stray `Enter` can never quarantine a folder's real DFIT file by
  accident.
- `Left` (back, no decision recorded) caches the screen's in-progress keep-set toggles by folder
  index in `self._inprogress`, and `_seed_keeps` restores that cache in preference to
  `scan.suggested` (but an existing, still-CURRENT ledger decision always wins over both -- see
  the next point) -- so glancing back to compare folders never silently discards an un-committed
  toggle.
- FIX 2: an existing ledger decision only wins in `_seed_keeps` (and only counts as "decided" in
  `first_undecided`/`progress`) when its recorded `files_sig` matches the group's CURRENT file
  fingerprint (`self._files_sig_by_rel`, via `Ledger.status_if_current`). A decision made against
  a different file set -- most concretely, a re-scan whose grouping semantics changed and grew or
  shrank the group -- resurfaces as if it had never been decided; the stale decision itself is
  left untouched in the ledger, not deleted, until a fresh commit overwrites it.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from triage import features, figure
from triage.ledger import Ledger, group_files_sig

PER_PAGE = 8

_KEY_LEGEND = (
    "Keys:  1-8 toggle keeper  |  Enter/Right = decide  |  0 = no DFIT here  |  u = unsure  |  "
    "Left = back (non-destructive)  |  [ / ] = prev/next page  |  q = quit"
)


# --------------------------------------------------------------------------------------------------
# per-file subfolder provenance (well-root grouping now mixes several subfolders on one page)
# --------------------------------------------------------------------------------------------------
def file_subfolder_label(group_folder: str, file_folder: str) -> str:
    """`file_folder`'s path relative to `group_folder` -- `""` when the file is directly in the
    well-root group folder itself (never the literal `"."` `os.path.relpath` would otherwise
    return), else the subfolder path (e.g. `"OLD Files"`, `"Raw Data\\CSVs"`). Well-root grouping
    (`triage.features.scan_folders`) can now put files from several original directories onto one
    review page, and data found under an "Old"/"OLD Files" directory is usually not the best
    pick -- this is the provenance a reviewer needs to see, computed purely from
    `FileFeatures.folder` vs. the group's own `FolderScan.folder`, both of which already exist."""
    rel = os.path.relpath(file_folder, group_folder)
    return "" if rel == os.curdir else rel


def _multi_well_warning_text(n_wells: int) -> str:
    """The header warning shown near the questionnaire label (FEATURE 1, point 9) for a well-root
    group whose subtree holds more than one distinct well's questionnaire -- `""` (no warning)
    whenever the group is unambiguous (`n_wells <= 1`). Factored out as a pure function so it is
    testable without a real `tk.Tk()` the way `_redraw` (which sets a `ttk.Label`'s text from
    this) is not."""
    if n_wells <= 1:
        return ""
    return f"MULTI-WELL FOLDER: {n_wells} questionnaires -- split manually"


def _annotate_file_subfolders(fig, scan, page: int, per_page: int = PER_PAGE) -> None:
    """Add a dim, always-visible subfolder-provenance label to each populated panel of `fig`
    (built by `figure.render_grid` for this same `scan`/`page`/`per_page`) -- never baked into
    the cached per-file PNGs (`figure.py` is untouched; no `PNG_RENDER_VERSION` bump needed).
    Panel N's axes is `fig.axes[N]` in creation order (`render_grid` adds one subplot per grid
    cell, visible ones first); a panel whose file is directly in the well root gets no label at
    all (an empty `file_subfolder_label` is a no-op here, matching "nothing for files directly in
    the well root")."""
    start = page * per_page
    files = scan.files[start:start + per_page]
    for i, feat in enumerate(files):
        if i >= len(fig.axes):
            break
        label = file_subfolder_label(scan.folder, feat.folder)
        if not label:
            continue
        fig.axes[i].text(
            0.98, 0.03, label, transform=fig.axes[i].transAxes, ha="right", va="bottom",
            fontsize=7, style="italic", color="dimgray",
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )


class ReviewApp:
    """Drives one review session over `data_root`. Construct with the `tk.Tk()` (or any
    toplevel-like widget) that will host it; `main()` below is the normal entry point."""

    def __init__(self, root_win: tk.Tk, data_root: str):
        self.data_root = data_root
        self.scans = features.load_scan(data_root)
        self.ledger = Ledger.load(data_root)
        self.order = [s.rel for s in self.scans]
        # FIX 2: each group's CURRENT file fingerprint, so a ledger decision recorded against a
        # different (usually smaller, pre-grouping-change) file set can be told apart from one
        # that is still current -- see `Ledger.status_if_current`. Computed once here rather than
        # per-navigation, since it only changes across a fresh `scan`/`review` process, not within
        # one review session.
        self._files_sig_by_rel = {
            s.rel: group_files_sig(f.sig for f in s.files) for s in self.scans
        }

        self.root_win = root_win
        self.page = 0
        self.keeps: set[str] = set()
        # In-progress (never-committed) keep-set toggles, cached by folder index on the way out
        # via `Left` so `_seed_keeps` can restore them instead of losing the toggle to a fresh
        # `scan.suggested` reseed (FIX 5) -- see the module docstring's navigation rules.
        self._inprogress: dict[int, set[str]] = {}
        # FigureCanvasTkAgg keeps no strong reference of its own to the figure or the canvas --
        # both must live on self for the same reason CLAUDE.md gives for holding slider refs.
        self.fig = None
        self.canvas: FigureCanvasTkAgg | None = None

        root_win.title("DFIT triage review")

        header = ttk.Frame(root_win, padding=6)
        header.pack(side="top", fill="x")
        self.progress_lbl = ttk.Label(header, text="")
        self.progress_lbl.pack(anchor="w")
        self.rel_lbl = ttk.Label(header, text="", font=("", 10, "bold"))
        self.rel_lbl.pack(anchor="w")
        self.quest_lbl = ttk.Label(header, text="", foreground="gray")
        self.quest_lbl.pack(anchor="w")
        # FEATURE 1: a well-root group whose subtree holds more than one distinct well's
        # questionnaire is never split automatically -- this makes that ambiguity impossible to
        # miss rather than silently picking one well's name for the whole page.
        self.warning_lbl = ttk.Label(header, text="", foreground="red", font=("", 9, "bold"))
        self.warning_lbl.pack(anchor="w")

        self.canvas_frame = ttk.Frame(root_win)
        self.canvas_frame.pack(side="top", fill="both", expand=True)

        footer = ttk.Frame(root_win, padding=6)
        footer.pack(side="bottom", fill="x")
        self.status_lbl = ttk.Label(footer, text="")
        self.status_lbl.pack(anchor="w")
        self.legend_lbl = ttk.Label(footer, text=_KEY_LEGEND, foreground="gray")
        self.legend_lbl.pack(anchor="w")

        root_win.bind("<Key>", self._on_key)

        self._goto(self.ledger.first_undecided(self.order, self._files_sig_by_rel))

    # ----------------------------------------------------------------------------------------
    # navigation / seeding
    # ----------------------------------------------------------------------------------------
    def _current_scan(self):
        if 0 <= self.index < len(self.scans):
            return self.scans[self.index]
        return None

    def _seed_keeps(self, scan) -> None:
        """Precedence (FIX 5): an existing ledger decision's keeps win (even an empty set, e.g.
        a "none" folder) -- but only when that decision is still CURRENT (FIX 2: its `files_sig`
        matches this group's present file set, via `_files_sig_by_rel`). A stale or legacy
        decision is treated exactly as if it had never been made, falling through to the same
        precedence an undecided folder gets. Otherwise an in-progress selection cached by
        `_go_back` for this folder index wins; otherwise seed from the scan's own suggestion."""
        current_sig = self._files_sig_by_rel.get(scan.rel, "")
        if self.ledger.status_if_current(scan.rel, current_sig):
            self.keeps = set(self.ledger.get(scan.rel).keeps)
        elif self.index in self._inprogress:
            self.keeps = set(self._inprogress[self.index])
        else:
            self.keeps = set(scan.suggested)

    def _goto(self, idx: int) -> None:
        # Clamp to [0, len(scans)] -- len(scans) is the "queue complete" sentinel _current_scan
        # already treats as past-the-end. Without this, repeated stray advances past the last
        # folder (e.g. several Enters in a row) would grow `index` unboundedly, each needing its
        # own Left to walk back.
        self.index = max(0, min(idx, len(self.scans)))
        self.page = 0
        scan = self._current_scan()
        if scan is not None:
            self._seed_keeps(scan)
        else:
            self.keeps = set()
        self._redraw()

    # ----------------------------------------------------------------------------------------
    # rendering
    # ----------------------------------------------------------------------------------------
    def _redraw(self) -> None:
        scan = self._current_scan()
        if scan is None:
            self._show_complete()
            return

        total = len(self.scans)
        decided, total_folders = self.ledger.progress(self.order, self._files_sig_by_rel)
        self.progress_lbl.config(
            text=f"folder {self.index + 1} / {total}    decided {decided}/{total_folders}")
        self.rel_lbl.config(text=scan.rel)
        if scan.well_name or scan.formation:
            self.quest_lbl.config(
                text=f"Well: {scan.well_name or '(unknown)'}   "
                     f"Formation: {scan.formation or '(unknown)'}")
        else:
            self.quest_lbl.config(text="no questionnaire")
        self.warning_lbl.config(text=_multi_well_warning_text(scan.n_wells))

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.fig = figure.render_grid(scan, self.data_root, page=self.page, per_page=PER_PAGE,
                                       keeps=self.keeps)
        _annotate_file_subfolders(self.fig, scan, self.page, PER_PAGE)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        pages = figure.page_count(scan, per_page=PER_PAGE)
        page_note = f"    page {self.page + 1}/{pages}" if pages > 1 else ""
        keeps_note = ", ".join(os.path.basename(k) for k in sorted(self.keeps)) or "(none)"
        self.status_lbl.config(text=f"keeps: {keeps_note}{page_note}")

    def _show_complete(self) -> None:
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            self.fig = None
        total = len(self.scans)
        decided, total_folders = self.ledger.progress(self.order, self._files_sig_by_rel)
        self.progress_lbl.config(text=f"folder {total} / {total}    decided {decided}/{total_folders}")
        self.rel_lbl.config(text="")
        self.quest_lbl.config(text="")
        self.warning_lbl.config(text="")
        self.status_lbl.config(text="queue complete")

    # ----------------------------------------------------------------------------------------
    # panel index <-> file index
    # ----------------------------------------------------------------------------------------
    def _file_index(self, key_n: int) -> int:
        """`key_n` is the pressed digit (1-8); the panel it selects on the current page."""
        return self.page * PER_PAGE + (key_n - 1)

    def _toggle_keep(self, key_n: int) -> None:
        scan = self._current_scan()
        if scan is None:
            return
        idx = self._file_index(key_n)
        if idx < 0 or idx >= len(scan.files):
            return  # out of range for this folder/page: ignore
        path = scan.files[idx].path
        if path in self.keeps:
            self.keeps.discard(path)
        else:
            self.keeps.add(path)
        self._redraw()

    # ----------------------------------------------------------------------------------------
    # paging
    # ----------------------------------------------------------------------------------------
    def _next_page(self) -> None:
        scan = self._current_scan()
        if scan is None:
            return
        if self.page + 1 < figure.page_count(scan, per_page=PER_PAGE):
            self.page += 1
            self._redraw()

    def _prev_page(self) -> None:
        if self.page > 0:
            self.page -= 1
            self._redraw()

    # ----------------------------------------------------------------------------------------
    # decisions
    # ----------------------------------------------------------------------------------------
    def _commit_and_advance(self) -> None:
        """Enter / Right: commit the current keep set as "decided", then move on.

        An empty keep set is NOT committed here (FIX 4): that used to silently record "none"
        ("no DFIT here"), indistinguishable from a deliberate `0` press, so a stray Enter could
        quarantine a folder's real DFIT file. Instead this is a no-op -- the ledger is untouched
        and the screen does not advance -- with a status message pointing at `0`/`u`."""
        scan = self._current_scan()
        if scan is None:
            return  # already past the end of the queue: nothing to commit or advance to
        if not self.keeps:
            self.status_lbl.config(
                text='No files selected -- press "0" for "no DFIT here", or "u" for "unsure".')
            return
        self.ledger.set(scan.rel, sorted(self.keeps), "decided",
                        files_sig=self._files_sig_by_rel.get(scan.rel, ""))
        self._goto(self.index + 1)

    def _mark_none_and_advance(self) -> None:
        """`0`: no DFIT in this folder."""
        scan = self._current_scan()
        if scan is not None:
            self.keeps = set()
            self.ledger.set(scan.rel, [], "none",
                             files_sig=self._files_sig_by_rel.get(scan.rel, ""))
        self._goto(self.index + 1)

    def _mark_unsure_and_advance(self) -> None:
        """`u`: revisit later, keeping whatever is currently selected."""
        scan = self._current_scan()
        if scan is not None:
            self.ledger.set(scan.rel, sorted(self.keeps), "unsure",
                             files_sig=self._files_sig_by_rel.get(scan.rel, ""))
        self._goto(self.index + 1)

    def _go_back(self) -> None:
        """`Left`: previous folder, without recording any decision for the one being left --
        but the in-progress keep-set toggles for the folder being left ARE cached, keyed by its
        index, so a later `Right`/`Enter` back onto it restores them via `_seed_keeps` instead of
        re-seeding from `scan.suggested` and losing the toggle (FIX 5)."""
        if self.index > 0:
            self._inprogress[self.index] = set(self.keeps)
            self._goto(self.index - 1)

    def _quit(self) -> None:
        self.ledger.save()
        self.root_win.destroy()

    # ----------------------------------------------------------------------------------------
    # key dispatch
    # ----------------------------------------------------------------------------------------
    def _on_key(self, event: tk.Event) -> None:
        ch = event.char
        keysym = event.keysym

        if ch.isdigit():
            if ch == "0":
                self._mark_none_and_advance()
            else:
                n = int(ch)
                if 1 <= n <= 8:
                    self._toggle_keep(n)
            return
        if ch == "u":
            self._mark_unsure_and_advance()
            return
        if ch == "q":
            self._quit()
            return
        if keysym in ("Return", "Right"):
            self._commit_and_advance()
            return
        if keysym == "Left":
            self._go_back()
            return
        if keysym == "bracketright":
            self._next_page()
            return
        if keysym == "bracketleft":
            self._prev_page()
            return


def main(root: str) -> None:
    win = tk.Tk()
    ReviewApp(win, root)
    win.mainloop()
