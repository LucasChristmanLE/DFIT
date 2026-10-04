"""Tkinter/ttk shell: file open, channel mapping, the eight-step canvas, live value panel.

Hosts the matplotlib canvas and wires the per-step pickers from picks.py to a recompute+redraw
loop. Holds no interpretation logic itself -- every number comes from model.compute_all.

There is no matplotlib toolbar: its sticky zoom/pan mode silently swallowed pick clicks/drags,
which is exactly the interaction this app depends on. View state (pan/zoom) is instead a
first-class per-step concept -- ``_views`` below, resolved by ``plots.apply_step_view`` -- restored on every
revisit to a step rather than only optionally preserved across one recompute.
"""

from __future__ import annotations

import math
import os
import pathlib
import shutil
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import APP_NAME
from . import (colors as C, guide_content, interpret, io_load, picks, plots, sliders, store,
               summary)
from .model import (NO_CONTACT_SCENARIOS, STEPS, PickState, TangentPick, blocking_issues,
                    closure_uninterpretable, compute_all, first_not_visited_step,
                    infer_step_status, last_step, loglog_window_suppressed, next_step,
                    pp_from_peak, prev_step, resolve_step, skipped_steps, step_gate_error)
from .plots import D2_AXIS_GID, ViewDefaults, ViewState
from .questionnaire import find_questionnaire, parse_questionnaire

_SLIDER_GID = "slider"
# Guess used only when no renderer is available yet to measure the real twin/d2 axis overhang
# (the initial _build_sliders() rect, before _layout_sliders() corrects it; or a resize firing
# on a canvas that can't produce a renderer) -- a reasonable stand-in for a typical tick-label
# column, not a measurement.
_FALLBACK_OVERHANG_PX = 60.0
# Neutral fallback for a slider with no ViewDefaults.y_color/y2_color (a horizontal slider, or
# a step with no single dominant color for that axis, e.g. log-log's two same-axis series).
_SLIDER_NEUTRAL_COLOR = C.UI_MUTED
# _layout_sliders measure->apply passes to let the d2P/dG2 axis's overhang settle (its spine
# sits at a FRACTION of the primary Axes' own width, ("axes", 1.12) -- so its absolute pixel
# overhang moves every time subplots_adjust changes that width, and a single measure-then-apply
# pass under-reserves after a big width change). Module-level (not a class attribute) so a
# duck-typed test stub -- a plain SimpleNamespace with no class hierarchy of its own -- can bind
# ``_layout_sliders`` without also needing to redeclare these.
_LAYOUT_PASSES = 3
# A pass-to-pass overhang change below this (px) counts as settled -- stop iterating early.
_LAYOUT_SETTLE_PX = 1.0
# dpi the pixel-ish layout constants below (col_px/pad_px/TRACK_PX, the x slider's track height/
# label gap/text pad) are tuned for. Every use scales by fig.dpi / _LAYOUT_DPI_REF, since a raw
# pixel count doesn't track a points-based quantity -- text, drawn at a fixed point size -- as
# dpi rises (e.g. Windows 150%/200% display scaling under TkAgg): an unscaled 60px column and
# an unscaled 14px track both still made visual sense at 100dpi but not at 150+.
_LAYOUT_DPI_REF = 100.0
# Guess used only when no renderer is available yet to measure the real x-axis tick-label/
# xlabel overhang below the primary Axes (_build_sliders' initial rect, or a resize firing on a
# canvas that can't produce a renderer) -- mirrors _FALLBACK_OVERHANG_PX for the bottom margin.
_FALLBACK_BOTTOM_OVERHANG_PX = 40.0
# Nominal (100dpi) height of the horizontal (x) slider's own track.
_X_TRACK_HEIGHT_PX = 18.0
# Nominal (100dpi) small gap between the x slider's track top and the x-axis tick labels/xlabel
# sitting just above it.
_BOTTOM_LABEL_GAP_PX = 6.0
# Nominal (100dpi) small pad added below the x slider's split text's estimated height
# (sliders.TEXT_HEIGHT_PT; never measured), so the text doesn't sit flush against the figure's bottom edge.
_BOTTOM_TEXT_PAD_PX = 3.0

CLOSURE_SCENARIOS = ["", "C-A clear", "C-B adequate", "C-C no-contact", "C-D rapid",
                     "C-X uninterpretable"]
POSTCLOSURE_SCENARIOS = ["", "PC-A linear", "PC-B false-radial",
                         "PC-C false radial to genuine linear",
                         "PC-D genuine linear to genuine radial",
                         "PC-E no trend", "PC-F no peak", "PC-X uninterpretable"]

# Advisory hints for the postclosure scenarios that don't fully dictate the pore-pressure axis
# (picks.suggest_pp_axis). Full explanatory text + figures live in the interpretation guide
# window (guide_content.py / _open_guide below); this dict is still consulted by _on_scenario.
_PC_HINTS = {
    "PC-D": "either axis valid -- choose t^(-1/2) or t^(-1) manually",
    "PC-E": "no clear slope -- pore pressure from a -1/2 line off the t*dP/dt peak; low confidence",
    "PC-F": "derivative still rising -- no log-log window; pore pressure step is skipped, "
            "Finish is available on this (log-log) step",
    "PC-X": "postclosure uninterpretable -- pore pressure and stiffness steps are skipped, "
            "Finish is available on this (log-log) step",
}

# Tabs for the single "Interpretation guide..." window (_open_guide), in display order.
GUIDE_TABS = [("closure", guide_content.CLOSURE_GUIDE), ("postclosure", guide_content.POSTCLOSURE_GUIDE)]
_GUIDE_ASSETS = pathlib.Path(__file__).parent / "assets" / "guide"
_LOGO_PATH = pathlib.Path(__file__).parent / "assets" / "liberty_logo.png"

# The 20 result-panel rows, in display order -- module level (not just a literal inside
# _build_body) so FIELD_STEP below and tests can both refer to the same list.
PANEL_FIELDS = [
    "te (min)", "Vinj (bbl)", "qmax (bpm)", "apparent ISIP",
    "eff ISIP (compliance)", "NWB complexity",
    "contact P", "Shmin compliance",
    "Shmin tangent", "Shmin variable", "Shmin Liberty",
    "tc compliance (min)", "tc tangent (min)", "tc variable (min)",
    "net (compliance)", "net (tangent)", "net (variable)",
    "delta closure", "pore pressure",
    "Shmin stiffness",
]

# Which step "owns" each panel field -- _update_panel shows "-" for a field whose step is still
# not_visited, even if compute_all already produced a value for it (e.g. a value carried over
# from a loaded JSON pick file the user hasn't actually visited yet this session). The three
# variable-method rows are guarded in compute_all on both the contact and closure picks, so they
# stay None (and display "-") until the gfunction step has actually been visited -- same
# precedent as "delta closure" owning only "tangent".
FIELD_STEP = {
    "te (min)": "injection",
    "Vinj (bbl)": "injection",
    "qmax (bpm)": "injection",
    "apparent ISIP": "isip",
    "eff ISIP (compliance)": "gfunction",
    # Needs the isip pick (apparent ISIP) and the gfunction pick (the reference eff ISIP);
    # gfunction is the later of the two, same precedent as "net (compliance)".
    "NWB complexity": "gfunction",
    "contact P": "gfunction",
    "Shmin compliance": "gfunction",
    "tc compliance (min)": "gfunction",
    "net (compliance)": "gfunction",
    "Shmin tangent": "tangent",
    "tc tangent (min)": "tangent",
    "net (tangent)": "tangent",
    "delta closure": "tangent",
    "Shmin variable": "tangent",
    "Shmin Liberty": "gfunction",
    "tc variable (min)": "tangent",
    "net (variable)": "tangent",
    "pore pressure": "porepressure",
    # Comparison-only fourth Shmin estimate (URTeC-2019-123 A.8/A.9 relative stiffness).
    "Shmin stiffness": "stiffness",
}


def _resolve_load_source(entry: store.TestEntry, saved: Optional[PickState]) -> str:
    """Which of ``entry.available_sources`` ``_load_test`` should open, absent an explicit
    ``source=`` override: the saved picks' ``active_source`` when there is a saved PickState
    naming a source that's actually available for this entry, else the first available source
    (``TestEntry.available_sources`` orders CSV, then DBS, then XLSX, among whichever exist)."""
    if saved is not None:
        for candidate in entry.available_sources:
            if candidate.lower() == saved.active_source:
                return candidate
    return entry.available_sources[0]


def _plan_source_load(entry: store.TestEntry, saved: Optional[PickState]) -> tuple[str, Optional[str]]:
    """`(source, missing)` for `_load_test`: `source` is what to open (as `_resolve_load_source`),
    `missing` is the saved picks' `active_source` when it names a source this entry no longer has
    (the picks are index-based and must not be applied to another file), else None."""
    source = _resolve_load_source(entry, saved)
    if saved is not None and source.lower() != saved.active_source:
        return source, saved.active_source
    return source, None


def _next_new_index(statuses: list[str], current_index: int) -> Optional[int]:
    """The index of the next ``"new"``-status entry in ``statuses``, scanning circularly
    starting just after ``current_index`` -- the pure selection logic behind Finish's and
    Skip test's auto-advance. Deliberately never revisits ``current_index`` itself even if its
    own status is ``"new"`` (its work was just saved this call), so "the only new entry is the
    current one" correctly reports no candidate. Returns None if ``statuses`` is empty or no
    other entry is ``"new"``."""
    n = len(statuses)
    for offset in range(1, n):
        i = (current_index + offset) % n
        if statuses[i] == "new":
            return i
    return None


def _isip_pick_in_minutes(pick: Optional[TangentPick],
                          t_shutin_s: float) -> Optional[TangentPick]:
    """Convert the stored apparent-ISIP tangent -- ``anchor_x`` in seconds-since-file-start
    (``td.t_s`` scale), ``slope`` in psi/s -- into the minutes-from-shut-in coordinates
    ``plots.render_isip`` actually plots, for the ``AnchorLineController`` wired to that Axes.
    Inverse: ``_isip_minutes_to_seconds``."""
    if pick is None:
        return None
    return TangentPick(anchor_x=(pick.anchor_x - t_shutin_s) / 60.0, anchor_y=pick.anchor_y,
                       slope=pick.slope * 60.0)


def _isip_minutes_to_seconds(anchor_x_min: float, slope_per_min: float,
                             t_shutin_s: float) -> tuple[float, float]:
    """Inverse of ``_isip_pick_in_minutes`` for the ``(anchor_x, slope)`` an
    ``AnchorLineController`` wired to the ISIP Axes reports on release -- back to the
    seconds-since-file-start / psi-per-second convention ``picks.commit_isip_tangent`` and
    ``state.isip_tangent`` use. ``anchor_y`` needs no conversion (BHP psi on both axes)."""
    return anchor_x_min * 60.0 + t_shutin_s, slope_per_min / 60.0


# Issue levels, most severe first: (DerivedResults attribute, section title, count noun,
# Overview header color, line prefix in the expanded side-panel label).
# Marks a questionnaire warning line; _update_issues_panel draws it bold in the warning color.
_QUEST_WARNING_PREFIX = "Warning: "

ISSUE_LEVELS = [
    ("blockers", "Blocking", "blocking issue", C.UI_ERROR, "Blocking: "),
    ("warnings", "Warnings", "warning", C.UI_WARNING, ""),
    ("notes", "Notes", "note", C.UI_MUTED, "Note: "),
]


def issue_sections(res) -> list[tuple[str, list[str]]]:
    """``(title, messages)`` for each non-empty issue level on ``res``, most severe first."""
    return [(title, list(getattr(res, attr))) for attr, title, *_ in ISSUE_LEVELS
            if getattr(res, attr)]


def format_warnings_text(sections: list[tuple[str, list[str]]], expanded: bool) -> str:
    """Render the side panel's issue label text from ``issue_sections``. Collapsed (the default
    after every recompute -- see ``DfitApp._update_panel``) shows only per-level counts, so a long
    list can never crowd the scenario controls above it out of view. Expanded lists every
    message, notes and blockers marked by a prefix."""
    if not sections:
        return ""
    by_title = {title: (noun, prefix) for _, title, noun, _, prefix in ISSUE_LEVELS}
    counts = []
    lines = []
    for title, msgs in sections:
        noun, prefix = by_title[title]
        counts.append(f"{len(msgs)} {noun}{'' if len(msgs) == 1 else 's'}")
        lines.extend(prefix + m for m in msgs)
    summary = ", ".join(counts)
    if not expanded:
        return f"{summary} (click to expand)"
    return f"{summary} (click to collapse)\n" + "\n".join(lines)


class DfitApp:
    def __init__(self, root: tk.Tk, path: str | None = None):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1400x850")
        self.root.state("zoomed")

        self.td: io_load.TestData | None = None
        self.state = PickState()
        self.res = None
        self.step = "overview"
        self._controllers: list = []
        self._views: dict[str, Optional[ViewState]] = {}
        self._x_slider: Optional[sliders.PanRangeSlider] = None
        self._y_slider: Optional[sliders.PanRangeSlider] = None
        self._y2_slider: Optional[sliders.PanRangeSlider] = None
        self._guide_win: Optional[tk.Toplevel] = None
        self._results_win: Optional[tk.Toplevel] = None  # Expanded results window
        self._guide_tab_index: dict[str, int] = {}

        # Folder mode: self.current_entry is the single mode flag -- None means single-file
        # mode (today's behavior, sidebar never packed). Task C adds dfit_log.csv writing on
        # top of log_df; this task only keeps it loaded.
        self.current_entry: store.TestEntry | None = None
        self.folder_root: str | None = None
        self.queue_entries: list[store.TestEntry] = []
        self.log_df = None

        # Closing the window must not lose the current folder-mode test's unsaved picks.
        self.root.protocol("WM_DELETE_WINDOW", self._on_root_close)

        self._build_top()
        self._build_body()
        self._build_stepbar()

        if path:
            if os.path.isdir(path):
                self._open_folder_path(path)
            else:
                self._load(path)

    # ---- layout ---------------------------------------------------------------------------------
    def _build_top(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open File…", command=self._open).pack(side="left")
        ttk.Button(top, text="Open Folder…", command=self._open_folder).pack(side="left")
        self.file_lbl = ttk.Label(top, text="(no file)")
        self.file_lbl.pack(side="left", padx=8)

        # Folder mode only: which of a test's CSV/DBS files is loaded. Disabled/cleared in
        # single-file mode and whenever a test has only one source -- _update_folder_controls
        # is the one sync point for this widget's state/values.
        ttk.Label(top, text="Source:").pack(side="left", padx=(8, 2))
        self.var_source = tk.StringVar()
        self.cmb_source = ttk.Combobox(top, textvariable=self.var_source, width=6,
                                       state="disabled")
        self.cmb_source.pack(side="left")
        self.cmb_source.bind("<<ComboboxSelected>>", lambda e: self._on_source_change())

        # Held on self: Tk GCs an unreferenced PhotoImage and the label goes blank.
        try:
            self._logo_img = tk.PhotoImage(file=str(_LOGO_PATH))
        except Exception:
            self._logo_img = None
        if self._logo_img is not None:
            # A packed spacer fixes the bar's size and keeps other widgets clear of the logo;
            # the logo is placed over it so it can sit lower, into the frame's bottom padding,
            # without the bar growing.
            w, h = self._logo_img.width(), self._logo_img.height()
            ttk.Frame(top, width=w + 22, height=h + 10).pack(side="right")
            # tk.Label, not ttk: the theme's label border adds 4 px that pushes the image off
            # the bar. place() coordinates start inside the frame's 6 px padding.
            bg = ttk.Style().lookup("TFrame", "background")
            tk.Label(top, image=self._logo_img, bd=0, padx=0, pady=0, highlightthickness=0,
                     bg=bg).place(relx=1.0, x=-14, y=14, anchor="ne")

        cfg = ttk.Frame(self.root, padding=(6, 0))
        cfg.pack(side="top", fill="x")
        self.var_pressure = tk.StringVar()
        self.var_isbhp = tk.BooleanVar(value=False)
        self.var_rate = tk.StringVar()
        self.var_volume = tk.StringVar()
        self.var_density = tk.StringVar()
        self.var_tvd = tk.StringVar()
        self.var_alpha = tk.StringVar(value="1.0")
        self.var_step = tk.StringVar(value="30")
        # Per-channel unit overrides -- "auto" (the default) means "detect it" (header suffix ->
        # magnitude heuristic -> field-unit inheritance; see io_load.detect_channel_unit). Changing
        # one away from "auto" resets picks (_on_unit_change), same reasoning as the Source combo.
        self.var_pressure_unit = tk.StringVar(value="auto")
        self.var_rate_unit = tk.StringVar(value="auto")
        self.var_volume_unit = tk.StringVar(value="auto")

        def combo(parent, label, var, width=24):
            ttk.Label(parent, text=label).pack(side="left", padx=(8, 2))
            c = ttk.Combobox(parent, textvariable=var, width=width, state="readonly")
            c.pack(side="left")
            return c

        def unit_combo(parent, var, values, kind):
            c = ttk.Combobox(parent, textvariable=var, values=values, width=8, state="readonly")
            c.pack(side="left", padx=(2, 0))
            c.bind("<<ComboboxSelected>>", lambda e: self._on_unit_change(kind))
            lbl = ttk.Label(parent, text="", foreground=C.UI_MUTED)
            lbl.pack(side="left", padx=(2, 0))
            return c, lbl

        self.cmb_pressure = combo(cfg, "Pressure:", self.var_pressure)
        self.cmb_pressure_unit, self.lbl_pressure_unit = unit_combo(
            cfg, self.var_pressure_unit, ["auto", "psi", "kpa", "mpa", "bar"], "pressure")
        ttk.Checkbutton(cfg, text="is BHP", variable=self.var_isbhp,
                        command=self._apply_config).pack(side="left", padx=4)
        self.cmb_rate = combo(cfg, "Rate:", self.var_rate, 20)
        self.cmb_rate_unit, self.lbl_rate_unit = unit_combo(
            cfg, self.var_rate_unit, ["auto", "bpm", "m3/min"], "rate")
        self.cmb_volume = combo(cfg, "Volume:", self.var_volume, 18)
        self.cmb_volume_unit, self.lbl_volume_unit = unit_combo(
            cfg, self.var_volume_unit, ["auto", "bbl", "m3"], "volume")

        # Pure metadata -- prefilled from the questionnaire like density/TVD, but nothing
        # computes on them and nothing gates on them. Free text, so plain Entry widgets.
        meta = ttk.Frame(self.root, padding=(6, 0))
        meta.pack(side="top", fill="x")
        self.var_well = tk.StringVar()
        self.var_formation = tk.StringVar()
        ttk.Label(meta, text="Well Name:").pack(side="left", padx=(8, 2))
        ttk.Entry(meta, textvariable=self.var_well, width=36).pack(side="left")
        ttk.Label(meta, text="Formation:").pack(side="left", padx=(8, 2))
        ttk.Entry(meta, textvariable=self.var_formation, width=26).pack(side="left")

        cfg2 = ttk.Frame(self.root, padding=(6, 2))
        cfg2.pack(side="top", fill="x")
        for label, var, w in [("Density (ppg):", self.var_density, 7),
                              ("TVD (ft):", self.var_tvd, 8),
                              ("alpha:", self.var_alpha, 5),
                              ("resample step (psi):", self.var_step, 6)]:
            ttk.Label(cfg2, text=label).pack(side="left", padx=(8, 2))
            ttk.Entry(cfg2, textvariable=var, width=w).pack(side="left")
        ttk.Button(cfg2, text="Apply", command=self._apply_config).pack(side="left", padx=10)
        ttk.Button(cfg2, text="Save picks…", command=self._save_picks).pack(side="right", padx=4)
        ttk.Button(cfg2, text="Load picks…", command=self._load_picks).pack(side="right")

        # Provenance for the density/TVD prefill above -- set by _load_questionnaire when a
        # questionnaire xlsx is auto-detected next to the data file; empty when none was found.
        # Density/TVD stay ordinary editable entries either way, this is just so the user can see
        # (and judge) the source. Shown on Overview only, below the issues list
        # (_update_issues_panel).
        self._quest_lines: list[str] = []

    def _build_body(self):
        # sashrelief="raised" makes the drag affordance visible; a flat sash is
        # indistinguishable from the surrounding ttk frames.
        self.body = tk.PanedWindow(self.root, orient="horizontal", sashwidth=5, bd=0,
                                   sashrelief="raised")
        self.body.pack(side="top", fill="both", expand=True)
        body = self.body
        self._queue_width = 240  # remembered pane width; see _show_queue/_hide_queue

        # left: folder-mode test queue -- built here but not added to the paned window yet.
        # _show_queue()/_hide_queue() (folder open / _load's exit-folder-mode path) own its
        # pane membership; single-file mode must stay pixel-identical to today, so this frame
        # starts absent from the paned window (no left pane at all, not just zero-width).
        self.queue_frame = ttk.Frame(body)

        self.progress_lbl = ttk.Label(self.queue_frame, text="0/0", padding=(4, 4))
        self.progress_lbl.pack(side="top", fill="x")

        tree_frame = ttk.Frame(self.queue_frame)
        tree_frame.pack(side="top", fill="both", expand=True)
        self.queue_tree = ttk.Treeview(tree_frame, columns=("status",), show="tree headings")
        self.queue_tree.heading("#0", text="Test")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("#0", width=140, stretch=True)
        self.queue_tree.column("status", width=90, anchor="w", stretch=False)
        queue_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=queue_vsb.set)
        self.queue_tree.pack(side="left", fill="both", expand=True)
        queue_vsb.pack(side="right", fill="y")
        self.queue_tree.bind("<<TreeviewSelect>>", self._on_queue_select)
        # Status is always readable as text (the "status" column); these tags are a secondary
        # color cue only, never the sole signal.
        self.queue_tree.tag_configure("done", foreground=C.UI_DONE)
        self.queue_tree.tag_configure("skipped", foreground=C.UI_MUTED)
        self.queue_tree.tag_configure("in_progress", foreground=C.UI_WARNING)
        self.queue_tree.tag_configure("new", foreground=C.UI_TEXT)

        # center: canvas
        center = ttk.Frame(body)
        self.body.add(center, stretch="always", minsize=400)
        self._center_frame = center  # so _show_queue can add the sidebar before= it
        self.fig = Figure(figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        # Re-run the pixel-based slider layout on every canvas resize -- tick-label pixel width
        # doesn't scale with a fixed figure-fraction margin, so a window resize (this app has no
        # fixed size) can reopen the exact overlap _layout_sliders() fixes on refresh(). This
        # only repositions existing Axes (set_position never destroys a slider) and redraws --
        # it must never call refresh() (that would fig.clf() the sliders mid-resize, same
        # invariant as a slider's own on_changed callback).
        self.canvas.mpl_connect("resize_event", self._on_canvas_resize)

        # right: pick panel
        panel = ttk.Frame(body, padding=8)
        self.body.add(panel, width=320, minsize=220, stretch="never")

        # Bottom-packed first so these keep their full height when a scenario frame overfills
        # the panel -- the squeeze then falls on the top-packed Results rows (all of them are in the
        # Expanded results window), never on the warnings, Notes, or step controls. On Overview
        # the issues list is packed ahead of Notes instead (_update_panel_visibility), so there
        # Notes is squeezed first. First-packed side="bottom" is bottommost: hint, warn, notes.
        self.hint_lbl = ttk.Label(panel, text="", wraplength=300, foreground=C.UI_MUTED)
        self.hint_lbl.pack(side="bottom", anchor="w", pady=(6, 0))
        self.warn_lbl = ttk.Label(panel, text="", foreground=C.UI_ERROR, wraplength=300,
                                  justify="left")
        self.warn_lbl.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))
        self.frm_notes = ttk.Frame(panel)
        self.frm_notes.pack(side="bottom", fill="x")
        self.sep_before_notes = ttk.Separator(self.frm_notes)
        self.sep_before_notes.pack(fill="x", pady=6)
        ttk.Label(self.frm_notes, text="Notes").pack(anchor="w")
        self.txt_notes = tk.Text(self.frm_notes, height=5, width=36)
        self.txt_notes.pack(fill="x")
        self._issue_sections: list[tuple[str, list[str]]] = []
        self._warnings_expanded: bool = False
        self.warn_lbl.bind("<Button-1>", self._toggle_warnings)
        self._panel_wrap = 300

        # Panel is now a resizable pane (sash-draggable), so wraplength must track its actual
        # width instead of a value pinned to the old fixed width=320.
        def _on_panel_configure(event):
            wrap = max(event.width - 20, 100)
            self._panel_wrap = wrap
            self.hint_lbl.configure(wraplength=wrap)
            self.warn_lbl.configure(wraplength=wrap)
            for child in self.frm_issues.winfo_children():
                if isinstance(child, ttk.Label):
                    child.configure(wraplength=wrap)
                else:  # questionnaire warning row: [bold prefix][message]
                    prefix, msg = child.winfo_children()
                    msg.configure(wraplength=max(wrap - prefix.winfo_reqwidth() - 4, 50))
        panel.bind("<Configure>", _on_panel_configure)

        # Results rows and the Overview issues list share one slot above sep_after_results:
        # Overview has no results yet, so it shows every issue there instead
        # (_update_panel_visibility swaps the two frames).
        self.frm_results = ttk.Frame(panel)
        self.frm_results.pack(fill="x")
        hdr = ttk.Frame(self.frm_results)
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Results", font=("", 10, "bold")).pack(side="left")
        ttk.Button(hdr, text="Expanded results...",
                   command=self._open_results_window).pack(side="right")
        self.value_lbls: dict[str, ttk.Label] = {}
        self.name_lbls: dict[str, ttk.Label] = {}
        for key in PANEL_FIELDS:
            # Value packed first: pane minsize only binds sash drags, so a too-narrow
            # window can still squeeze this pane -- the later-packed name label loses
            # pixels then, keeping the number readable.
            row = ttk.Frame(self.frm_results); row.pack(fill="x")
            v = ttk.Label(row, text="-", width=14, anchor="e"); v.pack(side="right")
            n = ttk.Label(row, text=key); n.pack(side="left")
            self.value_lbls[key] = v
            self.name_lbls[key] = n
        self.frm_issues = ttk.Frame(panel)  # filled by _update_issues_panel

        self.sep_after_results = ttk.Separator(panel)
        self.sep_after_results.pack(fill="x", pady=6)

        # Closure-scenario and postclosure/pp-axis widgets are step-aware: only relevant once the
        # user has reached the step that produces the pick they annotate. Each cluster lives in
        # its own frame so _update_panel_visibility can pack/pack_forget it as a unit without
        # disturbing anything else in the panel. Neither frame is packed here -- refresh() ->
        # _update_panel_visibility() does that, bottom-packed after frm_notes so each sits just
        # above Notes.
        self.frm_cscen = ttk.Frame(panel)
        ttk.Label(self.frm_cscen, text="Closure scenario").pack(anchor="w")
        self.var_cscen = tk.StringVar(value="")
        self.cmb_cscen = ttk.Combobox(self.frm_cscen, textvariable=self.var_cscen,
                                      values=CLOSURE_SCENARIOS, state="readonly")
        self.cmb_cscen.pack(fill="x")
        self.cmb_cscen.bind("<<ComboboxSelected>>", lambda e: self._on_scenario())
        self.var_showd2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm_cscen, text="show d²P/dG²", variable=self.var_showd2,
                        command=self._on_showd2).pack(anchor="w", pady=(4, 0))
        ttk.Button(self.frm_cscen, text="Interpretation guide...",
                   command=lambda: self._open_guide("closure")).pack(anchor="w", pady=(6, 0))

        self.frm_pcscen = ttk.Frame(panel)
        ttk.Label(self.frm_pcscen, text="Postclosure scenario").pack(anchor="w")
        self.var_pcscen = tk.StringVar(value="")
        self.cmb_pcscen = ttk.Combobox(self.frm_pcscen, textvariable=self.var_pcscen,
                                       values=POSTCLOSURE_SCENARIOS, state="readonly")
        self.cmb_pcscen.pack(fill="x")
        self.cmb_pcscen.bind("<<ComboboxSelected>>", lambda e: self._on_pcscen_selected())

        ttk.Label(self.frm_pcscen, text="Pore-pressure axis").pack(anchor="w", pady=(6, 0))
        self.var_ppaxis = tk.StringVar(value="tm12")
        self.rb_ppaxis = []
        ppaxis_row = ttk.Frame(self.frm_pcscen)  # side by side, to keep the frame short
        ppaxis_row.pack(anchor="w")
        for txt, val in [("t^(-1/2)", "tm12"), ("t^(-1)", "tm1")]:
            rb = ttk.Radiobutton(ppaxis_row, text=txt, variable=self.var_ppaxis, value=val,
                                 command=self._on_scenario)
            rb.pack(side="left", padx=(0, 12))
            self.rb_ppaxis.append(rb)

        ttk.Button(self.frm_pcscen, text="Interpretation guide...",
                   command=lambda: self._open_guide("postclosure")).pack(anchor="w", pady=(6, 0))

        # Stiffness step's "no slope change apparent" negative finding -- same step-gated frame
        # pattern as frm_cscen/frm_pcscen above, shown only on "stiffness".
        self.frm_stiffness = ttk.Frame(panel)
        self.var_stiffness_no_upturn = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm_stiffness, text="No slope change apparent",
                        variable=self.var_stiffness_no_upturn,
                        command=self._on_stiffness_no_upturn).pack(anchor="w")

        # Tangent step's "uninterpretable" negative finding -- same pattern, shown only on
        # "tangent".
        self.frm_tangent = ttk.Frame(panel)
        self.var_tangent_uninterpretable = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm_tangent, text="Tangent closure uninterpretable",
                        variable=self.var_tangent_uninterpretable,
                        command=self._on_tangent_uninterpretable).pack(anchor="w")

        # ISIP step's "use shut-in pressure" option (no water hammer) -- same pattern, shown only
        # on "isip".
        self.frm_isip = ttk.Frame(panel)
        self.var_isip_at_shutin = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm_isip, text="Use shut-in pressure (no tangent)",
                        variable=self.var_isip_at_shutin,
                        command=self._on_isip_at_shutin).pack(anchor="w")

        # Injection step's zoom toggle: show the whole record to find a missed injection, then
        # zoom back to the (moved) window.
        self.frm_injection = ttk.Frame(panel)
        self.btn_injection_full = ttk.Button(self.frm_injection, text="Show all data",
                                             command=self._on_injection_full_toggle)
        self.btn_injection_full.pack(anchor="w")

        # Overview's manual-mask tool: one button that drops every analyst mask/keep band.
        self.frm_masks = ttk.Frame(panel)
        self.btn_clear_masks = ttk.Button(self.frm_masks, text="Clear manual masks (0)",
                                          state="disabled", command=self._on_clear_masks)
        self.btn_clear_masks.pack(anchor="w")

    def _show_queue(self):
        """Add the folder-mode sidebar as the leftmost pane -- `before=` re-slots it ahead
        of the center pane regardless of add order. No-op when the pane is already present
        (folder-to-folder open): re-adding would reconfigure width= and snap a sash-dragged
        sidebar back to the remembered value."""
        if str(self.queue_frame) in (str(p) for p in self.body.panes()):
            return
        self.body.add(self.queue_frame, before=self._center_frame,
                      width=self._queue_width, minsize=140, stretch="never")

    def _hide_queue(self):
        # No-op when the queue pane isn't currently added (_load calls this even in
        # single-file mode). PanedWindow.panes() returns Tcl_Obj path names, so compare
        # by str() on both sides rather than relying on cross-type equality.
        if str(self.queue_frame) not in (str(p) for p in self.body.panes()):
            return
        # Remember the (possibly sash-dragged) width for re-entry this session. An unmapped
        # widget reports winfo_width()==1, so only trust it above that.
        w = self.queue_frame.winfo_width()
        if w > 1:
            self._queue_width = w
        self.body.forget(self.queue_frame)

    def _build_stepbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="bottom", fill="x")

        # Right side: Reset view, then Skip test. Packed right-to-left, so pack the
        # rightmost-visually one (Skip test) first, giving "[Reset view] [Skip test]"
        # reading left to right.
        # Folder mode only: park the whole test as "skipped" regardless of how far its steps
        # got, an override no per-step Skip > can express. Disabled/labeled in single-file mode
        # and toggled by _update_skip_test_btn, the one sync point for this button's state/text.
        self.btn_skip_test = ttk.Button(bar, text="Skip test", command=self._skip_test,
                                        state="disabled")
        self.btn_skip_test.pack(side="right", padx=4)
        ttk.Button(bar, text="Reset view", command=self._reset_view).pack(side="right", padx=4)

        # Left side: < Back, the seven breadcrumbs, Next >, Skip >.
        ttk.Button(bar, text="< Back", command=self._back).pack(side="left", padx=2)
        self.step_buttons: dict[str, ttk.Button] = {}
        for key, label in STEPS:
            btn = ttk.Button(bar, text=label, command=lambda k=key: self._goto(k))
            btn.pack(side="left", padx=2)
            self.step_buttons[key] = btn
        self.next_btn = ttk.Button(bar, text="Next >", command=self._advance)
        self.next_btn.pack(side="left", padx=2)
        ttk.Button(bar, text="Skip >", command=self._skip).pack(side="left", padx=2)
        self.gate_lbl = ttk.Label(bar, text="", foreground=C.UI_ERROR)
        self.gate_lbl.pack(side="left", padx=8)

    # ---- data / config --------------------------------------------------------------------------
    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("DFIT data", "*.csv *.dbs *.xlsx"),
                ("CSV", "*.csv"),
                ("Fracpro DBS", "*.dbs"),
                ("Excel", "*.xlsx"),
                ("All", "*.*"),
            ]
        )
        if path:
            self._load(path)

    def _load_common(self, path: str, well_hint: Optional[str] = None) -> bool:
        """Load `path` into a fresh PickState and land on "overview" -- shared by single-file
        _load and folder-mode _load_test. Returns False (leaving the previous self.td/state
        untouched) if the load failed, True on success.

        `well_hint` picks which sheet of a multi-well questionnaire workbook to read (see
        questionnaire._select_sheet) -- folder mode passes the entry's test_id; single-file mode
        passes nothing and _load_questionnaire falls back to the data file's stem."""
        try:
            self.td = io_load.load(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return False
        self.file_lbl.config(text=os.path.basename(path))
        cols = self.td.columns
        for cmb in (self.cmb_pressure, self.cmb_rate, self.cmb_volume):
            cmb["values"] = [""] + cols
        g = io_load.suggest_channels(cols, column=self.td.column)
        self.var_pressure.set(g["pressure"] or "")
        self.var_rate.set(g["rate"] or "")
        self.var_volume.set(g["volume"] or "")
        self.var_isbhp.set(bool(g["pressure_is_bhp"]))
        self.var_pressure_unit.set("auto")
        self.var_rate_unit.set("auto")
        self.var_volume_unit.set("auto")
        self.state = PickState()
        self._views = {k: None for k, _ in STEPS}
        self._injection_full = False
        self.var_cscen.set("")
        self.var_pcscen.set("")
        self.var_ppaxis.set("tm12")
        self.var_showd2.set(False)
        self.var_stiffness_no_upturn.set(False)
        self.var_tangent_uninterpretable.set(False)
        self.var_isip_at_shutin.set(False)
        self.txt_notes.delete("1.0", "end")
        # Density/TVD are per-well; clear the stale previous well's values before (maybe)
        # prefilling from a questionnaire, so a well with no questionnaire doesn't inherit them.
        self.var_density.set("")
        self.var_tvd.set("")
        self.var_well.set("")
        self.var_formation.set("")
        self._load_questionnaire(path, well_hint)
        self._sync_state_from_widgets()
        self._goto("overview")
        return True

    def _on_root_close(self):
        """WM_DELETE_WINDOW on the root: in folder mode save the current test's picks first; a
        failed save asks whether to close anyway. Single-file mode just closes (picks there save
        through a file dialog)."""
        if self.current_entry is not None:
            try:
                self._save_current_queue_picks()
            except Exception as e:
                if not messagebox.askyesno("Close", f"Save failed: {e}. Close anyway?"):
                    return
        self.root.destroy()

    def _load(self, path: str):
        """Single-file open: exit folder mode (saving any outgoing queue test's picks first),
        then load `path` via _load_common. The manual Save picks…/Load picks… buttons and every
        existing caller (__init__, _open) keep working unchanged through this wrapper."""
        self._save_current_queue_picks()
        self.current_entry = None
        self.folder_root = None
        self.queue_entries = []
        self._hide_queue()
        self.queue_tree.delete(*self.queue_tree.get_children())
        self.root.title(APP_NAME)
        self._update_folder_controls()
        self._load_common(path)

    # ---- folder mode ----------------------------------------------------------------------------
    def _open_folder(self):
        path = filedialog.askdirectory()
        if path:
            self._open_folder_path(path)

    def _make_scan_progress(self):
        """A small modal Toplevel with an indeterminate progress bar, shown over `self.root`
        while `_open_folder_path` scans and loads -- main thread only, no background work. Grabs
        input so the user can't click into the half-scanned queue. Returns `(win, set_text)`;
        callers must destroy `win` themselves (in a `finally`, since the scan/load below can
        raise or return early)."""
        win = tk.Toplevel(self.root)
        win.title("Opening folder")
        win.transient(self.root)
        win.resizable(False, False)
        # The scan/load below is a blocking main-thread call with no way to cancel mid-flight --
        # don't let the WM 'X' button destroy the modal (and drop its grab) out from under it.
        win.protocol("WM_DELETE_WINDOW", lambda: None)
        lbl = ttk.Label(win, text="Scanning folder…", width=48)
        lbl.pack(padx=16, pady=(16, 8))
        bar = ttk.Progressbar(win, mode="indeterminate", length=280)
        bar.pack(padx=16, pady=(0, 16))
        win.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - win.winfo_width()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - win.winfo_height()) // 2
        win.geometry(f"+{max(x, 0)}+{max(y, 0)}")
        win.grab_set()
        bar.start()

        def set_text(text: str):
            lbl.config(text=text)
            self.root.update()

        return win, set_text

    def _open_folder_path(self, path: str):
        progress_win, set_progress_text = self._make_scan_progress()
        entries: list[store.TestEntry] = []
        try:
            def _on_scan_progress(dirs_scanned: int, tests_found: int):
                set_progress_text(f"Scanning folder…  {tests_found} test(s) found")

            entries, log_df = store.list_tests(path, progress=_on_scan_progress)
            if entries:
                self._save_current_queue_picks()  # never lose the outgoing test's work
                self.folder_root = path
                self.queue_entries = entries
                self.log_df = log_df
                self._populate_queue()
                self._show_queue()
                # Clear current_entry before the auto-open below -- if _load_test's
                # _load_common fails (corrupt/unreadable first file), it returns early
                # without ever assigning current_entry, and this folder's queue must not be
                # left paired with either no test (fine) or, worse, a stale TestEntry from
                # whatever folder/test was open before this call (current_entry is None is
                # the mode invariant "no test loaded", not "no possibly-wrong test loaded").
                self.current_entry = None
                target = next((e for e in entries if e.status == "new"), entries[0])
                set_progress_text(f"Loading {target.display_label}…")
                self._load_test(target)
                self._update_folder_controls()  # covers _load_test's early return too
        finally:
            try:
                progress_win.grab_release()
                progress_win.destroy()
            except tk.TclError:
                pass  # already destroyed (e.g. window closed out from under the scan)

        if not entries:
            messagebox.showinfo("Open Folder", "No DFIT tests found in this folder.")
            return

        # Surface scan warnings as a short summary rather than dialog-spamming per test.
        warn_count = sum(len(e.scan_warnings) for e in entries)
        entry_ids = {e.test_id for e in entries}
        orphan_ids = [tid for tid in log_df["test_id"].tolist() if tid not in entry_ids]
        parts = []
        if log_df.attrs.get("read_error"):
            parts.append(f"{store.LOG_FILENAME} is unreadable and will not be written")
        if warn_count:
            parts.append(f"{warn_count} scan warning(s) -- see individual test folders")
        if orphan_ids:
            parts.append(f"{len(orphan_ids)} orphaned log row(s) with no matching test")
        if parts:
            self.warn_lbl.config(text="\n".join(parts))

    def _populate_queue(self):
        self.queue_tree.delete(*self.queue_tree.get_children())
        for entry in self.queue_entries:
            self.queue_tree.insert("", "end", iid=entry.test_id, text=entry.display_label,
                                   values=(entry.status,), tags=(entry.status,))
        self._update_progress_label()

    def _refresh_queue_row(self, entry: store.TestEntry):
        if self.queue_tree.exists(entry.test_id):
            self.queue_tree.item(entry.test_id, values=(entry.status,), tags=(entry.status,))
        self._update_progress_label()

    def _update_progress_label(self):
        total = len(self.queue_entries)
        n = sum(1 for e in self.queue_entries if e.status in ("done", "skipped"))
        self.progress_lbl.config(text=f"{n}/{total}")

    def _save_current_queue_picks(self):
        """The only persistence on queue navigation -- no dfit_log.csv writes here (Task C).
        No-op in single-file mode (current_entry is None) or before any file is loaded."""
        if self.current_entry is None or self.td is None:
            return
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        # Capture any unapplied entry-widget edits (density, TVD, well name, formation,
        # channel mapping, alpha, resample step) so they aren't silently dropped on save.
        self._sync_state_from_widgets()
        store.save_picks_for(self.current_entry, self.state)
        self.current_entry.status = store.status_for(self.state)
        self._refresh_queue_row(self.current_entry)

    def _load_test(self, entry: store.TestEntry, source: Optional[str] = None,
                   force_reset: bool = False):
        """Load `entry` into the workspace: resolve which data file to open, load it fresh via
        _load_common, then (unless force_reset) resume any saved picks. force_reset=True is for
        Task C's source switching -- it skips the saved-picks resume, keeping the fresh state
        _load_common already made."""
        source_was_none = source is None
        probed_picks = None
        missing_source = None
        if source_was_none:
            try:
                probed_picks = store.load_picks_for(entry)
            except store.PicksReadError as e:
                messagebox.showerror("Picks unreadable", f"{e}\n\nThe test was not opened.")
                return
            source, missing_source = _plan_source_load(entry, probed_picks)
            if missing_source is not None:
                # Index-based picks from a source that is gone must not be applied to another
                # file. Keep a copy of the picks file and start fresh (no resume). Without the
                # copy, the next save would overwrite the only record, so don't open the test.
                try:
                    shutil.copy2(entry.picks_path, f"{entry.picks_path}.{missing_source}.bak")
                except OSError as e:
                    messagebox.showerror(
                        "Saved picks",
                        f"Picks were made on {missing_source.upper()}, which is missing, and "
                        f"the picks file could not be backed up ({e}). The test was not opened.")
                    return
                probed_picks = None
        elif not force_reset:
            # Explicit source: still read the picks before _load_common replaces the workspace,
            # so a read error leaves the current test (and current_entry) untouched.
            try:
                probed_picks = store.load_picks_for(entry)
            except store.PicksReadError as e:
                messagebox.showerror("Picks unreadable", f"{e}\n\nThe test was not opened.")
                return
        path = entry.data_path(source)
        if not self._load_common(path, well_hint=entry.test_id):
            return
        if source_was_none and missing_source is not None:
            messagebox.showwarning(
                "Saved picks",
                f"Picks were made on {missing_source.upper()}, which is missing; "
                "starting fresh (backup kept).")
        saved = None
        if not force_reset:
            saved = probed_picks  # read above, before _load_common
            if saved is not None:
                self._apply_loaded_state(saved)
        self.state.active_source = source.lower()
        self.current_entry = entry
        self.root.title(f"{APP_NAME} — {entry.display_label}")
        entry.status = store.status_for(self.state if saved else None)
        self._refresh_queue_row(entry)
        self._update_folder_controls()

    def _update_folder_controls(self):
        """One sync point for the Source combobox and the Skip-test button -- called from
        _load_test (after current_entry is set), the single-file _load wrapper (after clearing
        folder state), and _open_folder_path (which also covers _load_test's early return on a
        failed load, since that return happens before this call runs inside _load_test itself).

        Single-file mode (current_entry is None): cmb_source cleared/disabled, Skip-test
        disabled via _update_skip_test_btn. Folder mode: cmb_source lists the entry's available
        sources (readonly only if there's more than one -- a single-source test has nothing to
        switch to), and Skip-test is enabled with its label synced to explicit_status."""
        if self.current_entry is None:
            self.var_source.set("")
            self.cmb_source["values"] = []
            self.cmb_source.config(state="disabled")
            self._update_skip_test_btn()
            return
        entry = self.current_entry
        self.cmb_source["values"] = entry.available_sources
        self.var_source.set(self.state.active_source.upper())
        self.cmb_source.config(
            state="readonly" if len(entry.available_sources) > 1 else "disabled")
        self._update_skip_test_btn()

    def _on_source_change(self):
        """The Source combobox: switching between CSV/DBS/XLSX resets all picks for this test (a
        fresh _load_test, not a resume), so confirm first -- reverting the combobox on decline."""
        new = self.var_source.get()
        current = self.state.active_source.upper()
        if new == current:
            return
        if not messagebox.askyesno(
                "Switch data source",
                "Switching the data source resets all picks for this test. Continue?"):
            self.var_source.set(current)
            return
        self._load_test(self.current_entry, source=new, force_reset=True)

    def _on_unit_change(self, kind: str):
        """One of the three per-channel unit dropdowns (pressure/rate/volume): mirrors
        _on_source_change -- an override rescales the same column under existing picks
        (`isip_tangent` stores an absolute psi anchor/slope, and `te_s` feeds `g_time`, shifting
        the meaning of every stored G pick), so switching it resets picks rather than silently
        reinterpreting them under a new scale. Confirm first; decline reverts the combobox to
        its prior value. The existing plain channel-remap combos keep their current no-confirm
        behavior -- a deliberate scope boundary (see ../CLAUDE.md)."""
        # kind is "pressure"/"rate"/"volume" -- both the widget var and the PickState attribute
        # it mirrors follow the same "<kind>_unit" naming, so this is looked up rather than
        # spelled out three times (and, unlike a dict literal of all three, never touches the
        # other two kinds' widgets).
        var = getattr(self, f"var_{kind}_unit")
        attr = f"{kind}_unit"
        current = getattr(self.state, attr)
        if self.td is None:
            # No file loaded: there's nothing to reset picks on and nothing for the dropdown to
            # mean yet -- revert it rather than popping a confirm dialog with no test in play.
            var.set(current)
            return
        new = var.get()
        if new == current:
            return
        if not messagebox.askyesno(
                "Change unit",
                "Changing this channel's unit resets all picks for this test. Continue?"):
            var.set(current)
            return
        setattr(self.state, attr, new)
        self._reset_picks_keep_mapping()
        self.var_cscen.set("")
        self.var_pcscen.set("")
        self.var_ppaxis.set("tm12")
        self.var_showd2.set(False)
        self.var_stiffness_no_upturn.set(False)
        self.var_tangent_uninterpretable.set(False)
        self.var_isip_at_shutin.set(False)
        self._views = {k: None for k, _ in STEPS}
        self._injection_full = False
        self._goto("overview")

    def _reset_picks_keep_mapping(self):
        """A fresh PickState that keeps only what a unit-override change must not disturb:
        channel mapping, unit overrides, density/TVD, well/formation, alpha, resample step,
        notes, and the active data source -- every numeric pick, step_status, and the tail trim
        reset to their defaults. Used by _on_unit_change (see its docstring for why those picks
        can't just be carried forward)."""
        old = self.state
        self.state = PickState(
            pressure_col=old.pressure_col,
            rate_col=old.rate_col,
            volume_col=old.volume_col,
            pressure_is_bhp=old.pressure_is_bhp,
            pressure_unit=old.pressure_unit,
            rate_unit=old.rate_unit,
            volume_unit=old.volume_unit,
            density_ppg=old.density_ppg,
            tvd_ft=old.tvd_ft,
            well_name=old.well_name,
            formation=old.formation,
            alpha=old.alpha,
            resample_step=old.resample_step,
            notes=old.notes,
            active_source=old.active_source,
        )

    def _write_log_row(self, entry: store.TestEntry):
        """Build and upsert one dfit_log.csv row for `entry` from the current state/res, then
        persist the whole log -- shared by _finish's and _skip_test's folder branches, the only
        two places dfit_log.csv is written. Callers wrap this in try/except (OneDrive file
        locks happen); the picks JSON save must already have succeeded independently before
        this runs."""
        row = store.build_log_row(entry, entry.data_path(self.state.active_source.upper()),
                                  self.folder_root, self.state, self.td, self.res)
        try:
            # Re-read from disk so an unreadable log is never replaced by an empty frame.
            log_df = store.load_log(self.folder_root)
        except store.LogReadError as e:
            messagebox.showerror(
                "Log not written",
                f"{e.path} could not be read ({e.cause}). "
                "The log was NOT written; the picks were saved.")
            return
        self.log_df = store.upsert_log_row(log_df, row)
        store.save_log(self.folder_root, self.log_df)

    def _advance_queue(self):
        """Advance to the next "new"-status queue entry (scanning circularly from just after
        the current one), or report the queue is exhausted. Shared tail of Finish and Skip
        test's folder branches -- both have just finalized the current entry's status before
        calling this, so `current_entry` is still the just-finished test when this runs."""
        entry = self.current_entry
        statuses = [e.status for e in self.queue_entries]
        current_index = self.queue_entries.index(entry)
        next_index = _next_new_index(statuses, current_index)
        if next_index is None:
            messagebox.showinfo("Queue", "No new tests remain.")
            return
        self._load_test(self.queue_entries[next_index])

    def _update_skip_test_btn(self):
        """One sync point for the Skip-test button's enabled state and toggle label -- called
        from _update_folder_controls (mode changes / test load) and _update_stepbar (every
        refresh, so the label tracks a state swapped in by _apply_loaded_state, e.g. loading a
        picks JSON that already has explicit_status set)."""
        if self.current_entry is None:
            self.btn_skip_test.config(state="disabled", text="Skip test")
            return
        self.btn_skip_test.config(
            state="normal",
            text="Unskip test" if self.state.explicit_status == "skipped" else "Skip test")

    def _skip_test(self):
        """Bound to the Skip-test button -- only reachable in folder mode (the button is
        disabled otherwise), guarded anyway. A toggle: flags the whole test as "skipped"
        regardless of how far its steps got (an override no per-step Skip > can express), or
        clears that flag if it's already set (the button reads "Unskip test" then). Flagging
        saves + logs + advances, same as Finish minus the PNG export -- a skipped test produces
        no plots. Unflagging saves + logs and stays put, with no confirm dialog either way:
        nothing is discarded (picks are untouched) and the action is reversible by clicking
        again."""
        if self.current_entry is None or self.td is None:
            return
        entry = self.current_entry
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        # Capture any unapplied entry-widget edits, then refresh so self.res (feeding the log
        # row below) is recomputed from the synced state rather than a stale prior compute.
        self._sync_state_from_widgets()
        self.refresh()
        flagging = self.state.explicit_status != "skipped"
        self.state.explicit_status = "skipped" if flagging else None
        try:
            store.save_picks_for(entry, self.state)
        except Exception as e:
            messagebox.showerror("Save failed", f"Picks were not saved: {e}")
            return
        entry.status = store.status_for(self.state)
        try:
            self._write_log_row(entry)
        except Exception as e:
            messagebox.showerror("Log write failed", str(e))
        self._refresh_queue_row(entry)
        self._update_skip_test_btn()
        if flagging:
            self._advance_queue()

    def _on_queue_select(self, event=None):
        sel = self.queue_tree.selection()
        if not sel:
            return
        test_id = sel[0]
        if self.current_entry is not None and test_id == self.current_entry.test_id:
            return
        entry = next((e for e in self.queue_entries if e.test_id == test_id), None)
        if entry is None:
            return
        try:
            self._save_current_queue_picks()
        except Exception as e:
            messagebox.showerror("Save failed", f"Picks were not saved: {e}")
            # Stay on the current test; put the sidebar selection back.
            if self.current_entry is not None:
                self.queue_tree.selection_set(self.current_entry.test_id)
            return
        self._load_test(entry)

    def _load_questionnaire(self, csv_path: str, well_hint: Optional[str] = None):
        """Auto-detect and parse a DFIT Questionnaire xlsx next to `csv_path`; prefill
        density/TVD/well name/formation.

        `well_hint` disambiguates a multi-sheet workbook (see questionnaire._select_sheet) --
        falls back to the data file's own stem when not given (single-file mode; folder mode
        passes the entry's test_id via _load_common).

        Best-effort only: a missing or malformed questionnaire must never block the CSV load
        already underway, so any failure here is swallowed and just leaves the provenance label
        empty. Density/TVD entries are prefilled even when the parse is uncertain (e.g. a coerced
        SG->ppg reading) -- the provenance label shows the raw source text so it can be checked.
        Well name/formation are plain free text, so there's no analogous "source" text to show.
        """
        self._quest_lines = []
        try:
            xlsx_path, find_warnings = find_questionnaire(csv_path)
            if xlsx_path is None:
                return
            hint = well_hint if well_hint is not None else pathlib.Path(csv_path).stem
            result = parse_questionnaire(xlsx_path, well_hint=hint)
        except Exception:
            return

        parts = []
        if result.density_ppg is not None:
            self.var_density.set(str(result.density_ppg))
            parts.append(f'Density: {result.density_ppg} ppg ["{result.density_source}"]')
        if result.tvd_ft is not None:
            self.var_tvd.set(str(result.tvd_ft))
            parts.append(f'TVD: {result.tvd_ft} ft ["{result.tvd_source}"]')
        if result.well_name is not None:
            self.var_well.set(result.well_name)
            parts.append(f'Well: {result.well_name}')
        if result.formation is not None:
            self.var_formation.set(result.formation)
            parts.append(f'Formation: {result.formation}')
        parts += [f"{_QUEST_WARNING_PREFIX}{w}" for w in find_warnings + result.warnings]
        if parts:
            self._quest_lines = [f"File: {os.path.basename(xlsx_path)}"] + parts

    def _sync_state_from_widgets(self):
        self.state.pressure_col = self.var_pressure.get()
        self.state.rate_col = self.var_rate.get() or None
        self.state.volume_col = self.var_volume.get() or None
        self.state.pressure_is_bhp = self.var_isbhp.get()
        self.state.pressure_unit = self.var_pressure_unit.get()
        self.state.rate_unit = self.var_rate_unit.get()
        self.state.volume_unit = self.var_volume_unit.get()
        # This can fire mid-edit (e.g. on autosave), so a non-empty but
        # unparseable entry ("8." mid-keystroke) must not null out a
        # previously-good, already-logged value -- only an explicitly
        # emptied box clears it.
        self.state.density_ppg = _num(self.var_density.get(), self.state.density_ppg)
        self.state.tvd_ft = _num(self.var_tvd.get(), self.state.tvd_ft)
        self.state.well_name = self.var_well.get().strip()
        self.state.formation = self.var_formation.get().strip()
        self.state.alpha = _to_float(self.var_alpha.get()) or 1.0
        self.state.resample_step = _to_float(self.var_step.get()) or 30.0
        # Belt-and-suspenders: refresh_unit_detection also runs at the top of every compute_all,
        # but doing it here too means td.unit_factors reflects the just-synced overrides before
        # anything (e.g. a step seeder) reads a channel off self.td directly.
        if self.td is not None:
            io_load.refresh_unit_detection(
                self.td, self.state.pressure_col, self.state.rate_col, self.state.volume_col,
                self.state.pressure_unit, self.state.rate_unit, self.state.volume_unit)

    def _apply_config(self):
        if self.td is None:
            return
        self._sync_state_from_widgets()
        self.refresh()

    def _on_pcscen_selected(self):
        """Any pick in the postclosure combobox is the analyst's, including re-picking an
        auto-set PC-A, so a later window drag no longer re-evaluates it. Kept off _on_scenario,
        which the closure combobox and the axis radios also call."""
        self.state.postclosure_auto = False
        self._on_scenario()

    def _on_scenario(self):
        cscen = self.var_cscen.get()
        cscen_changed = cscen != self.state.closure_scenario
        self.state.closure_scenario = cscen
        pcscen = self.var_pcscen.get()
        pcscen_changed = pcscen != self.state.postclosure_scenario
        self.state.postclosure_scenario = pcscen
        self.state.pp_axis = self.var_ppaxis.get()
        hint = None
        if cscen_changed and self.td is not None:
            # Selecting a closure scenario is an explicit request to re-derive the contact
            # pick from that scenario's rule (it may overwrite a previous pick).
            hint = picks.apply_closure_scenario(self.state, compute_all(self.state, self.td))
        if pcscen_changed:
            # Selecting a postclosure scenario drives the pore-pressure axis (see
            # picks.suggest_pp_axis); PC-D/PC-F/PC-X leave the axis to the analyst.
            axis = picks.suggest_pp_axis(pcscen)
            if axis is not None:
                self.state.pp_axis = axis
            self.var_ppaxis.set(self.state.pp_axis)
            hint = _PC_HINTS.get(pcscen[:4]) or hint
            if pp_from_peak(self.state):
                hint = self._ensure_pce_peak() or hint
        self._update_ppaxis_enabled()
        if pcscen_changed and self.step in skipped_steps(self.state):
            # PC-F/PC-X just got selected while sitting on the now-skipped pore-pressure step -- the
            # scenario combobox is visible on both loglog and porepressure, so this can happen
            # without ever leaving porepressure. _goto redirects (resolve_step) and calls
            # refresh() itself; calling refresh() again here would just redo the same work.
            self._goto(self.step)
        else:
            self.refresh()
        if hint:
            # After refresh()/_goto(): _attach_controllers just set the step's default hint
            # text, and the scenario feedback must win.
            self.hint_lbl.config(text=hint)

    def _on_showd2(self):
        self.state.show_d2pdg2 = self.var_showd2.get()
        self.refresh()

    def _on_stiffness_no_upturn(self):
        """Toggle the stiffness step's negative finding. On UNCHECK, if no pick exists yet
        (the step's first-visit seeder already fired once and won't fire again), seed one now
        so the analyst isn't left with no line and no way to get one."""
        self.state.stiffness_no_upturn = self.var_stiffness_no_upturn.get()
        if not self.state.stiffness_no_upturn and self.state.stiffness_pick_P is None:
            picks.seed_stiffness(self.state, self.res)
        self.refresh()

    def _on_tangent_uninterpretable(self):
        """Toggle the tangent step's negative finding. On UNCHECK, seed a pick if none exists
        (same reason as _on_stiffness_no_upturn)."""
        self.state.tangent_uninterpretable = self.var_tangent_uninterpretable.get()
        if not self.state.tangent_uninterpretable and self.state.closure_G is None:
            picks.seed_tangent(self.state, self.res)
        self.refresh()

    def _on_isip_at_shutin(self):
        """Toggle taking the apparent ISIP at the shut-in sample instead of the tangent. The
        tangent pick stays in state (hidden), so unchecking restores it."""
        self.state.isip_at_shutin = self.var_isip_at_shutin.get()
        self.refresh()

    # ---- expanded results window ------------------------------------------------------------------
    def _open_results_window(self):
        """Open the single non-modal Expanded results window (or refocus it)."""
        if self._results_win is not None and self._results_win.winfo_exists():
            self._results_win.deiconify()
            self._results_win.lift()
            self._results_win.focus_set()
        else:
            self._build_results_window()
        self._update_results_window()

    def _build_results_window(self):
        win = tk.Toplevel(self.root)
        win.title("Expanded results")
        win.geometry("1300x800")

        def _on_close():
            self._results_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        panes = ttk.PanedWindow(win, orient="horizontal")
        panes.pack(fill="both", expand=True)

        left = ttk.Frame(panes)
        canvas = tk.Canvas(left, highlightthickness=0, width=560)
        vsb = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(inner_id, width=e.width))
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))
        panes.add(left, weight=0)  # tables keep their width; extra window width goes to charts

        right = ttk.Frame(panes)
        fig = Figure(figsize=(6.5, 6.5))
        chart = FigureCanvasTkAgg(fig, master=right)
        chart.get_tk_widget().pack(fill="both", expand=True)
        panes.add(right, weight=1)

        self._results_tables = inner
        self._results_fig = fig
        self._results_canvas = chart
        self._results_win = win

    def _update_results_window(self):
        """Rebuild the tables and redraw the charts from the current state/results. The figure
        is the window's own, so the main canvas's slider invariants are untouched."""
        win = self._results_win
        if win is None or not win.winfo_exists():
            self._results_win = None
            return
        res = self.res
        for child in self._results_tables.winfo_children():
            child.destroy()
        if res is None:
            ttk.Label(self._results_tables, text="No results yet").pack(anchor="w", padx=6, pady=6)
            self._results_fig.clear()
            self._results_canvas.draw_idle()
            return
        for sec in summary.summary_sections(self.state, res):
            ttk.Label(self._results_tables, text=sec.title,
                      font=("", 10, "bold")).pack(anchor="w", padx=6, pady=(8, 2))
            tree = ttk.Treeview(self._results_tables, columns=list(range(len(sec.columns))),
                                show="headings", height=len(sec.rows), selectmode="none")
            # The label column keeps a fixed width; only the numeric columns stretch, so a narrow
            # pane squeezes the numbers rather than the labels (Tk shrinks stretchable columns).
            for i, name in enumerate(sec.columns):
                tree.heading(i, text=name)
                if i == 0:
                    tree.column(i, width=150, minwidth=150, anchor="w", stretch=False)
                else:
                    tree.column(i, width=58, minwidth=45, anchor="e", stretch=True)
            for row in sec.rows:
                tree.insert("", "end", values=row)
            tree.pack(fill="x", padx=6)
        plots.render_summary(self._results_fig, self.state, res)
        self._results_canvas.draw_idle()

    # ---- interpretation guide window --------------------------------------------------------------
    def _open_guide(self, key: str):
        """Open the single interpretation-guide window (or refocus it) on the tab for `key`
        ("closure" or "postclosure"). Reused across both side-panel buttons so there is never
        more than one guide window."""
        if self._guide_win is not None and self._guide_win.winfo_exists():
            self._guide_win.deiconify()
            self._guide_win.lift()
            self._guide_win.focus_set()
        else:
            self._build_guide_window()
        self._guide_notebook.select(self._guide_tab_index[key])

    def _build_guide_window(self):
        win = tk.Toplevel(self.root)
        win.title(f"{APP_NAME} interpretation guide")
        win.state("zoomed")
        win._guide_images = []  # keep PhotoImage refs alive; Tk GCs unreferenced images.

        def _on_close():
            self._guide_win = None
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", _on_close)

        notebook = ttk.Notebook(win)
        notebook.pack(fill="both", expand=True)
        self._guide_notebook = notebook
        self._guide_tab_index = {}

        for tab_i, (key, guide) in enumerate(GUIDE_TABS):
            container = ttk.Frame(notebook)
            notebook.add(container, text=guide.title)
            self._guide_tab_index[key] = tab_i

            canvas = tk.Canvas(container, highlightthickness=0)
            vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=vsb.set)
            canvas.pack(side="left", fill="both", expand=True)
            vsb.pack(side="right", fill="y")

            inner = ttk.Frame(canvas)
            canvas.create_window((0, 0), window=inner, anchor="nw")

            def _on_inner_configure(event, canvas=canvas):
                canvas.configure(scrollregion=canvas.bbox("all"))
            inner.bind("<Configure>", _on_inner_configure)

            def _on_mousewheel(event, canvas=canvas):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            canvas.bind("<MouseWheel>", _on_mousewheel)

            self._render_guide(inner, guide, win)

            # Tk delivers <MouseWheel> only to the widget under the pointer and does not bubble
            # to parents, so the bare canvas binding never fires while the pointer is over the
            # labels/images that cover most of the tab. Bind every rendered child too.
            def _bind_wheel(widget):
                widget.bind("<MouseWheel>", _on_mousewheel)
                for child in widget.winfo_children():
                    _bind_wheel(child)
            _bind_wheel(inner)

        self._guide_win = win

    def _render_guide(self, inner: ttk.Frame, guide: guide_content.Guide, win: tk.Toplevel):
        """Render one Guide, top-down, into `inner` (a scroll-region frame in `win`)."""
        ttk.Label(inner, text=guide.title, font=("", 12, "bold"), wraplength=900,
                  justify="left").pack(anchor="w", padx=10, pady=(10, 4))
        ttk.Label(inner, text=guide.intro, wraplength=900, justify="left").pack(
            anchor="w", padx=10, pady=(0, 10))

        for section in guide.sections:
            ttk.Separator(inner).pack(fill="x", padx=10, pady=6)
            ttk.Label(inner, text=section.title, font=("", 10, "bold"), wraplength=900,
                      justify="left").pack(anchor="w", padx=10, pady=(4, 2))
            ttk.Label(inner, text=section.body, wraplength=900, justify="left").pack(
                anchor="w", padx=10, pady=(0, 6))
            for fig in section.figures:
                img = self._load_guide_image(fig.image)
                if img is not None:
                    win._guide_images.append(img)
                    ttk.Label(inner, image=img).pack(anchor="w", padx=10, pady=(0, 2))
                else:
                    ttk.Label(inner, text=f"[figure unavailable: {fig.image}]",
                              foreground=C.UI_ERROR).pack(anchor="w", padx=10, pady=(0, 2))
                ttk.Label(inner, text=fig.caption, wraplength=900, justify="left",
                          font=("", 8, "italic"), foreground=C.UI_MUTED).pack(
                    anchor="w", padx=10, pady=(0, 10))

        ttk.Label(inner, text=guide.source, wraplength=900, justify="left",
                  foreground=C.UI_MUTED).pack(anchor="w", padx=10, pady=(6, 10))

    def _load_guide_image(self, name: str):
        try:
            return tk.PhotoImage(file=str(_GUIDE_ASSETS / name))
        except Exception:
            return None

    def _reconcile_pp_axis(self):
        """Force pp_axis to the value a postclosure scenario dictates (if any), so a locked
        axis can't disagree with its scenario. pp_axis feeds compute_all, so refresh() calls
        this before recomputing; PC-D/PC-F/PC-X/unset return None and leave a manual choice intact."""
        axis = picks.suggest_pp_axis(self.state.postclosure_scenario)
        if axis is not None and axis != self.state.pp_axis:
            self.state.pp_axis = axis
            self.var_ppaxis.set(self.state.pp_axis)

    def _update_ppaxis_enabled(self):
        """Lock the pore-pressure axis radios whenever the postclosure scenario dictates the
        axis (picks.suggest_pp_axis), so the scenario and the manual radios can't disagree."""
        dictated = picks.suggest_pp_axis(self.state.postclosure_scenario) is not None
        for rb in self.rb_ppaxis:
            rb.state(["disabled"] if dictated else ["!disabled"])

    # ---- steps / render -------------------------------------------------------------------------
    def _goto(self, step: str):
        """Navigate to ``step``. Breadcrumb buttons for a ``not_visited`` step are disabled by
        _update_stepbar, so reaching one here means either it was already reached, or this is
        the programmatic first jump onto a step (initial load, or Next/Skip/Back stepping one
        further than the user has been). First-visit seeding lives here, not in Next/Skip/Back,
        so the seed always runs regardless of which control got the user there.

        A destination the workflow leaves out (model.skipped_steps; PC-F/PC-X drop porepressure and
        stiffness) redirects through model.resolve_step -- this one place covers the log-log
        Skip button, resume-on-load (first_not_visited_step), and any other programmatic jump."""
        if self.td is None:
            return
        step = resolve_step(self.state, step)
        # A blocking issue (blocking_issues) makes every later step meaningless: land on Overview
        # instead, without seeding or marking the requested step.
        blocked = step != "overview" and bool(blocking_issues(self.state))
        if blocked:
            step = "overview"
        seed_hint = None
        if self.state.step_status.get(step, "not_visited") == "not_visited":
            seed_hint = self._seed_step(step)
            self.state.step_status[step] = "visited"
        if pp_from_peak(self.state):
            # Any step, so a resume onto stiffness or Overview (then Skip/Finish) still logs Pp.
            seed_hint = self._ensure_pce_peak() or seed_hint
        self.step = step
        self.refresh()
        if seed_hint:
            # After refresh(): _attach_controllers resets the step's default hint.
            self.hint_lbl.config(text=seed_hint)
        if blocked:
            self.gate_lbl.config(text=step_gate_error(self.state, "overview"))

    def _ensure_pce_peak(self) -> Optional[str]:
        """Seed the PC-E peak pick when PC-E is set and the pick is missing (a save made before
        the pick existed) or unusable (past the last sample after a trim, or no positive
        t*dP/dt there, which also leaves no marker to drag). Runs on selecting PC-E and on every
        _goto. Returns a hint when an unusable pick was replaced, else None."""
        if self.td is None or not pp_from_peak(self.state):
            return None
        res = compute_all(self.state, self.td)
        old = self.state.pce_peak_t
        if old is not None and res.pce_peak_t is not None:
            return None
        self.state.pce_peak_t = None
        picks.seed_pce_peak(self.state, res)
        if self.state.pce_peak_t is None:
            self.state.pce_peak_t = old  # nothing better to offer; keep the analyst's pick
            return None
        if old is not None:
            return "PC-E peak re-seeded: the old pick was unusable. Check it on Log-log."
        return None

    def _seed_step(self, key: str) -> Optional[str]:
        """Pre-populate reasonable default picks for ``key`` on its first visit, via
        ``picks.SEEDERS``. "overview" and "injection" need ``self.td`` too (seed_overview just
        delegates to seed_injection); "isip" needs both; the rest take only (state, res).

        "overview" additionally seeds the tail-trim line (picks.seed_tail_trim, not in SEEDERS)
        once the window exists: the ``res`` computed below predates seed_overview on a fresh file
        (no start_idx/shutin_idx yet, so te_s/resampled_full/guard_dt are all None), so it must be
        recomputed after seed_overview runs before the trim seeder has anything to see.

        Returns a one-shot hint when the seed auto-set a scenario ("gfunction" auto C-A), for
        _goto to show after its refresh; else None."""
        if self.td is None:
            return None
        res = compute_all(self.state, self.td)
        seeder = picks.SEEDERS[key]
        if key in ("overview", "injection"):
            seeder(self.state, self.td)
            if key == "overview":
                res = compute_all(self.state, self.td)
                picks.seed_tail_trim(self.state, self.td, res)
        elif key == "isip":
            seeder(self.state, self.td, res)
        else:
            hint = seeder(self.state, res)
            if key == "gfunction":
                # seed_gfunction may auto-assign C-A, which the combobox doesn't see on its own.
                self.var_cscen.set(self.state.closure_scenario)
                return hint
            if key == "loglog":
                # seed_loglog may auto-assign PC-A, which the combobox doesn't see on its own.
                self.var_pcscen.set(self.state.postclosure_scenario)
        return None

    def _next(self):
        """Mark the current step done and advance. next_step() clamps at the last step, so at
        "stiffness" this simply re-marks it done and re-refreshes -- a no-op in terms of
        navigation."""
        if self.td is None:
            return
        self.state.step_status[self.step] = "done"
        self._goto(next_step(self.step))

    def _skip(self):
        """Mark the current step skipped and advance, same clamping behavior as _next(). Not
        gated by the scenario picks, but Overview's blocking issues gate it like Next."""
        if self.td is None:
            return
        if self.step == "overview":
            msg = self._overview_gate()
            if msg:
                self.gate_lbl.config(text=msg)
                return
        self.state.step_status[self.step] = "skipped"
        self._goto(next_step(self.step))

    def _advance(self):
        """Bound to the Next/Finish stepbar button. On the effective last step (model.last_step,
        normally "stiffness" but "loglog" under PC-F/PC-X) the button reads
        "Finish" and exports (_finish). Otherwise it advances (_next) -- but only once the
        current step's required scenario pick is present; step_gate_error gates the forward
        jump and the inline gate_lbl says what is missing. Back/Skip/breadcrumb navigation are
        NOT gated."""
        if self.td is None:
            return
        if self.step == last_step(self.state):
            self._finish()
            return
        msg = (self._overview_gate() if self.step == "overview"
               else step_gate_error(self.state, self.step))
        if msg:
            self.gate_lbl.config(text=msg)
            return
        self._next()

    def _overview_gate(self) -> Optional[str]:
        """Blocking-issue message for leaving Overview, or None. Next/Skip act as Apply here:
        typed-but-unapplied Density/TVD or channel edits are synced (and recomputed) first, so
        filled-in boxes count. Gates on ``res.blockers`` rather than ``blocking_issues`` alone,
        so a BHP conversion that raises also stops navigation before the next step is seeded."""
        before = self.state.channel_config()
        self._sync_state_from_widgets()
        if self.state.channel_config() != before:
            self.refresh()
        if not self.res.blockers:
            return None
        return (step_gate_error(self.state, "overview")
                or f"{self.res.blockers[0]}; fix it before continuing.")

    def _back(self):
        """Go to the previous step. No status change -- prev_step() clamps at the first step."""
        if self.td is None:
            return
        self._goto(prev_step(self.step))

    def refresh(self):
        if self.td is None:
            return
        self.gate_lbl.config(text="")
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        # Reconcile a locked axis with its scenario before recomputing -- pp_axis feeds
        # compute_all, so an older save (e.g. PC-B + tm12) must be corrected here or the first
        # render (e.g. resuming directly onto porepressure) would show a stale pore pressure.
        self._reconcile_pp_axis()
        self.res = compute_all(self.state, self.td)
        # A blocker can appear while on a later step (density/TVD cleared and applied, or a BHP
        # conversion that raises): fall back to Overview, where the issue is listed.
        redirected = bool(self.res.blockers) and self.step != "overview"
        if redirected:
            self.step = "overview"

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        kwargs = {"full_record": self._injection_full} if self.step == "injection" else {}
        defaults = plots.RENDERERS[self.step](self.ax, self.td, self.state, self.res, **kwargs)

        sv = plots.apply_step_view(self.step, self.ax, defaults, self._views.get(self.step))
        self._views[self.step] = sv.view

        self._build_sliders(sv.full_x, sv.full_y, sv.full_y2, sv.view, sv.twin,
                            y_color=defaults.y_color, y2_color=defaults.y2_color)
        # tight_layout would fight the manually placed slider axes reserved on the right margin.
        # _layout_sliders() measures the twin/d2 axis's real tick-label overhang in pixels and
        # sets both the plot's right margin and the sliders' positions from that -- replacing the
        # old fixed-fraction right=0.70/0.84 split, which didn't scale with window width/DPI and
        # let the sliders sit on top of the twin's tick labels once the figure got narrow.
        self._layout_sliders()
        self._attach_controllers()
        self.canvas.draw_idle()
        self._update_stepbar()
        self._update_panel_visibility()
        self._update_panel()
        self._update_issues_panel()
        self._update_unit_labels()
        if getattr(self, "_results_win", None) is not None:
            self._update_results_window()
        if redirected:
            self.gate_lbl.config(text=step_gate_error(self.state, "overview")
                                 or f"{self.res.blockers[0]}; fix it before continuing.")

    def _update_stepbar(self):
        """Disable breadcrumb buttons for steps still ``not_visited`` (so a click is only ever
        honored for a reached step) and highlight the current step. Bold text rather than an
        Accent.TButton style -- that style name is theme-specific and not guaranteed to exist.

        Breadcrumbs for steps the workflow leaves out (model.skipped_steps) are force-disabled,
        even if visited earlier in the session (e.g. the analyst picked PC-F after already
        reaching pore pressure) -- _goto redirects them regardless, so none must look reachable."""
        style = ttk.Style()
        style.configure("StepCurrent.TButton", font=("TkDefaultFont", 9, "bold"))
        skipped = skipped_steps(self.state)
        for key, btn in self.step_buttons.items():
            status = self.state.step_status.get(key, "not_visited")
            reachable = status != "not_visited" and key not in skipped
            btn.state(["!disabled"] if reachable else ["disabled"])
            btn.configure(style="StepCurrent.TButton" if key == self.step else "TButton")
        # On the effective last step the Next button becomes Finish (bold, like the current-step
        # breadcrumb) -- _advance dispatches to _finish() instead of _next() in that case.
        if self.step == last_step(self.state):
            self.next_btn.configure(text="Finish", style="StepCurrent.TButton")
        else:
            self.next_btn.configure(text="Next >", style="TButton")
        self._update_skip_test_btn()

    def _update_panel_visibility(self):
        """Show the closure-scenario widgets only on "gfunction", the postclosure/pp-axis
        widgets only on "loglog"/"porepressure", the tangent "uninterpretable" checkbox only on
        "tangent", and the stiffness "no slope change apparent" checkbox only on "stiffness" --
        each packed side="bottom" after frm_notes, so it sits directly above Notes and a
        short panel clips the Results rows instead of the controls."""
        self.frm_cscen.pack_forget()
        self.frm_pcscen.pack_forget()
        self.frm_tangent.pack_forget()
        self.frm_isip.pack_forget()
        self.frm_stiffness.pack_forget()
        self.frm_masks.pack_forget()
        self.frm_injection.pack_forget()
        if self.step == "injection":
            self.btn_injection_full.config(
                text="Zoom to injection" if self._injection_full else "Show all data")
            self.frm_injection.pack(side="bottom", fill="x", after=self.frm_notes)
        if self.step == "overview":
            self.frm_masks.pack(side="bottom", fill="x", after=self.frm_notes)
            self._update_masks_button()
            self.frm_results.pack_forget()
            # Ahead of frm_notes in pack order (still drawn at the top): the issues list is the
            # only full list of blockers, so a short panel squeezes Notes instead of it.
            self.frm_issues.pack(fill="x", before=self.frm_notes)
        else:
            self.frm_issues.pack_forget()
            self.frm_results.pack(fill="x", before=self.sep_after_results)
        if self.step == "gfunction":
            self.frm_cscen.pack(side="bottom", fill="x", after=self.frm_notes)
        if self.step == "isip":
            self.var_isip_at_shutin.set(self.state.isip_at_shutin)
            self.frm_isip.pack(side="bottom", fill="x", after=self.frm_notes)
        if self.step == "tangent":
            self.var_tangent_uninterpretable.set(self.state.tangent_uninterpretable)
            self.frm_tangent.pack(side="bottom", fill="x", after=self.frm_notes)
        if self.step in ("loglog", "porepressure"):
            self.frm_pcscen.pack(side="bottom", fill="x", after=self.frm_notes)
            # refresh() already reconciled pp_axis with the scenario before recomputing; here
            # just lock/unlock the radios to match.
            self._update_ppaxis_enabled()
        if self.step == "stiffness":
            self.var_stiffness_no_upturn.set(self.state.stiffness_no_upturn)
            self.frm_stiffness.pack(side="bottom", fill="x", after=self.frm_notes)

    def _update_masks_button(self):
        """Show the manual mask+keep band count on the Clear button; disabled when there are none."""
        n = len(self.state.mask_intervals) + len(self.state.keep_intervals)
        self.btn_clear_masks.config(text=f"Clear manual masks ({n})",
                                    state="normal" if n else "disabled")

    def _on_injection_full_toggle(self):
        """Flip the Injection step between the whole record and the injection-window zoom. The
        stored view is dropped either way so the new default (full extent, or the window around
        the current picks) applies."""
        self._injection_full = not self._injection_full
        self._views["injection"] = None
        self.refresh()

    def _on_clear_masks(self):
        picks.clear_manual_masks(self.state)
        self.refresh()

    def _twin_axes(self):
        """The step's twin (secondary y) Axes if it has one, else None.

        Excludes the slider Axes _build_sliders adds to the right margin -- those are tagged
        with gid ``_SLIDER_GID`` precisely so this scan doesn't mistake one of them for the
        step's twin -- and excludes the gfunction step's optional d2P/dG2 axis
        (``D2_AXIS_GID``, decision D3), which gets no slider/persisted view of its own and must
        not be grabbed here in its place.
        """
        for a in self.fig.axes:
            if a is not self.ax and a.get_gid() not in (_SLIDER_GID, D2_AXIS_GID):
                return a
        return None

    def _d2_axes(self):
        """The gfunction step's optional d2P/dG2 twin Axes (``D2_AXIS_GID``), if present."""
        for a in self.fig.axes:
            if a.get_gid() == D2_AXIS_GID:
                return a
        return None

    def _build_sliders(self, full_x, full_y, full_y2, view, twin, y_color=None, y2_color=None):
        """Per-axis RangeSlider zoom controls: one under the plot for x, one on the right edge
        for y, and (only when a twin exists) one further right for y2.

        Called from refresh() after the Axes are (re)built and ``view`` applied, so slider
        ranges/initial values always reflect the just-applied ViewState. The on_changed
        callbacks only set limits on the target Axes, mutate ``view`` in place, and draw_idle --
        never refresh() (fig.clf() would destroy the slider mid-drag).

        The rects built here are only an initial guess (via ``sliders.right_margin_layout``/
        ``bottom_margin_layout`` with fixed fallback overhangs, since there is no renderer to
        measure the real ones from yet) -- ``refresh()`` immediately calls ``_layout_sliders()``
        after this, which corrects every Axes' position from the actually-measured overhangs
        (and, for the vertical sliders' column width, their own real text). Both steps share the
        same pure layout functions so the two positions agree in everything but those figures.
        """
        fig_w_px = self.fig.get_size_inches()[0] * self.fig.dpi
        fig_h_px = self.fig.get_size_inches()[1] * self.fig.dpi
        dpi_scale = self.fig.dpi / _LAYOUT_DPI_REF
        # full_y (and so a y slider) is always present once there is data; full_y2 (and a
        # second column) only when refresh() found a twin to read it from. Reserving a column
        # here even for a slider that _make_range_slider ends up refusing (a degenerate extent)
        # is harmless -- _layout_sliders() recomputes from the sliders actually built anyway.
        n_vertical = 2 if twin is not None else 1
        axes_right_frac, x_fracs = sliders.right_margin_layout(
            fig_w_px, n_vertical, _FALLBACK_OVERHANG_PX * dpi_scale,
            col_px=60.0 * dpi_scale, pad_px=12.0 * dpi_scale, track_px=sliders.TRACK_PX * dpi_scale)
        # Same safety clamp _layout_sliders applies -- without it a very narrow canvas gives the x
        # slider a negative width and fig.add_axes raises.
        axes_right_frac = min(max(axes_right_frac, 0.3), 0.95)
        track_w_frac = sliders.TRACK_PX * dpi_scale / fig_w_px

        default_track_y0_px, track_h_px, text_below_px, label_gap_px = (
            self._x_track_geometry_px(fig_h_px))
        _, track_y0_frac = sliders.bottom_margin_layout(
            fig_h_px, default_track_y0_px, track_h_px, text_below_px,
            _FALLBACK_BOTTOM_OVERHANG_PX * dpi_scale, label_gap_px)

        self._x_slider = self._make_range_slider(
            rect=[0.10, track_y0_frac, axes_right_frac - 0.10, track_h_px / fig_h_px],
            orientation="horizontal",
            full_range=full_x, cur_range=view.xlim, scale=self.ax.get_xscale(),
            apply=lambda lo, hi: self.ax.set_xlim(lo, hi),
            store=lambda lo, hi: setattr(view, "xlim", (lo, hi)),
        )
        self._y_slider = self._make_range_slider(
            rect=[x_fracs[0], 0.16, track_w_frac, 0.74], orientation="vertical",
            full_range=full_y, cur_range=view.ylim, scale=self.ax.get_yscale(),
            apply=lambda lo, hi: self.ax.set_ylim(lo, hi),
            store=lambda lo, hi: setattr(view, "ylim", (lo, hi)),
            color=y_color,
        )
        self._y2_slider = None
        if twin is not None:  # refresh() only sets full_y2 when there is a twin to read it from
            self._y2_slider = self._make_range_slider(
                rect=[x_fracs[1], 0.16, track_w_frac, 0.74], orientation="vertical",
                full_range=full_y2, cur_range=view.y2lim if view.y2lim is not None else full_y2,
                scale=twin.get_yscale(),
                apply=lambda lo, hi: twin.set_ylim(lo, hi),
                store=lambda lo, hi: setattr(view, "y2lim", (lo, hi)),
                color=y2_color,
            )

    def _make_range_slider(self, rect, orientation, full_range, cur_range, scale, apply, store,
                           color=None):
        """Build one PanRangeSlider, or return None for a degenerate/non-finite extent (a flat
        or single-sample axis has nothing to zoom).

        ``scale`` is the target Axes' actual xscale/yscale ("log" or "linear") -- for a log axis
        the slider itself operates in log10 space (clamped to a 1e-12 floor) with a valfmt that
        displays the linear value, and the callback exponentiates before applying limits.

        ``color`` (a vertical slider only, per ``ViewDefaults.y_color``/``y2_color``) paints the
        selected-range fill, the handles, and the split value texts to match the line this
        slider zooms -- ``None`` (a horizontal slider, or a step with no single dominant color
        for that axis, e.g. log-log) falls back to a neutral gray.
        """
        is_log = scale == "log"
        lo_full, hi_full = full_range
        lo_cur, hi_cur = cur_range
        if is_log:
            lo_full, hi_full = sliders.to_log_bounds(lo_full, hi_full)
            lo_cur, hi_cur = sliders.to_log_bounds(lo_cur, hi_cur)
        lo_full, hi_full = sorted((lo_full, hi_full))
        lo_cur, hi_cur = sorted((lo_cur, hi_cur))
        if not (math.isfinite(lo_full) and math.isfinite(hi_full) and lo_full < hi_full):
            return None
        # The stored view can drift outside the freshly autoscaled full extent (e.g. after the
        # data changes) -- clamp valinit into range rather than letting RangeSlider reject it.
        lo_cur = min(max(lo_cur, lo_full), hi_full)
        hi_cur = min(max(hi_cur, lo_full), hi_full)
        if lo_cur >= hi_cur:
            lo_cur, hi_cur = lo_full, hi_full

        ax = self.fig.add_axes(rect)
        ax.set_gid(_SLIDER_GID)
        valfmt = (lambda v: f"{10.0 ** v:.3g}") if is_log else None
        c = color if color is not None else _SLIDER_NEUTRAL_COLOR
        slider = sliders.PanRangeSlider(
            ax, "", lo_full, hi_full, valinit=(lo_cur, hi_cur), orientation=orientation,
            valfmt=valfmt, facecolor=c, handle_style={"facecolor": c})
        if slider._hi_text is not None:
            slider._hi_text.set_color(c)
            slider._lo_text.set_color(c)

        def _on_changed(val):
            lo, hi = val
            if is_log:
                lo, hi = sliders.from_log_bounds(lo, hi)
            apply(lo, hi)
            store(lo, hi)
            self.canvas.draw_idle()

        slider.on_changed(_on_changed)
        return slider

    def _on_canvas_resize(self, event):
        """Canvas resize_event hook (connected once in __init__): re-run the pixel-based slider
        layout so the sliders track the twin axis's tick labels as the window is resized, then
        redraw. Never calls refresh() (see the connection comment in __init__)."""
        self._layout_sliders()
        self.canvas.draw_idle()

    def _x_track_geometry_px(self, fig_h_px):
        """Pixel inputs for ``sliders.bottom_margin_layout`` that don't depend on a measured
        overhang -- the look-and-feel baseline track position, the track's own height, the
        split text's vertical clearance requirement, and the small label gap -- all dpi-scaled
        from this module's nominal (100dpi) constants. Points-denominated figures (the text
        offset/height) need no separate ``dpi_scale`` factor of their own: ``px = pt * dpi/72``
        is already dpi-correct, which is the whole point of using points there in the first
        place (see sliders.py's ``TEXT_OFFSET_PT``)."""
        dpi_scale = self.fig.dpi / _LAYOUT_DPI_REF
        default_track_y0_px = 0.04 * fig_h_px
        track_h_px = _X_TRACK_HEIGHT_PX * dpi_scale
        text_below_px = ((sliders.TEXT_OFFSET_PT + sliders.TEXT_HEIGHT_PT) * self.fig.dpi / 72.0
                         + _BOTTOM_TEXT_PAD_PX * dpi_scale)
        label_gap_px = _BOTTOM_LABEL_GAP_PX * dpi_scale
        return default_track_y0_px, track_h_px, text_below_px, label_gap_px

    def _layout_sliders(self):
        """Pixel-based right/bottom-margin layout for the plot and its zoom sliders.

        Measures how far the twin axis (and, on the gfunction step with d2P/dG2 on, the third
        axis too) overhangs past the primary Axes' right edge, and how far the x-axis's own tick
        labels/xlabel overhang past its bottom edge -- via ``get_tightbbox`` against a live
        renderer, so both reflect the actual tick-label text at the current font/DPI/window
        size, not a guess -- and feeds those into ``sliders.right_margin_layout``/
        ``bottom_margin_layout`` to get the plot's margins and each slider's track position.
        Each vertical slider's own column width comes from the wider of a dpi-scaled default and
        the actual measured width of that slider's own rendered text (``_measure_slider_text_col_px``),
        so a wide log-format value (e.g. "1.3e+03") gets a wide-enough column instead of
        colliding with its neighbor. Applies everything with ``subplots_adjust``/
        ``set_position``, which never destroys a slider (unlike ``fig.clf()`` -- so this is safe
        to call from a slider's own resize/redraw path without breaking the "sliders never call
        refresh()" invariant).

        Repeats measure-then-apply up to ``_LAYOUT_PASSES`` times, since applying a new right
        margin can itself move the d2P/dG2 axis's overhang (see that constant's comment) --
        stops as soon as both measured overhangs settle to within ``_LAYOUT_SETTLE_PX``. Every
        value used to position a slider after the loop (``axes_right_frac``/``x_fracs``/
        ``bottom_frac``/``track_y0_frac``) is exactly what the loop's LAST iteration computed
        and then passed to ``subplots_adjust`` in that same iteration -- never a value clamped
        or recomputed afterward without being re-applied, which is what let the sliders and the
        plot's own margin disagree whenever the loop exhausted without settling.

        Always runs -- including ``subplots_adjust`` -- even when no slider exists at all (e.g.
        no file loaded yet, or a step whose extents are all degenerate): the plot still gets a
        margin sized for zero reserved columns rather than being left at whatever margin the
        previous step happened to leave behind. Called from refresh() right after
        _build_sliders() (correcting that call's fallback-overhang guesses) and from the
        canvas's resize_event hook on every resize.
        """
        fig_w_px = self.fig.get_size_inches()[0] * self.fig.dpi
        fig_h_px = self.fig.get_size_inches()[1] * self.fig.dpi
        dpi_scale = self.fig.dpi / _LAYOUT_DPI_REF
        present = [s for s in (self._y_slider, self._y2_slider) if s is not None]
        default_track_y0_px, track_h_px, text_below_px, label_gap_px = (
            self._x_track_geometry_px(fig_h_px))
        col_px_floor = 60.0 * dpi_scale
        pad_px = 12.0 * dpi_scale
        track_px = sliders.TRACK_PX * dpi_scale

        right_overhang_px = self._measure_overhang_px()
        bottom_overhang_px = self._measure_bottom_overhang_px()
        for _ in range(_LAYOUT_PASSES):
            col_px = max(col_px_floor, self._measure_slider_text_col_px(present, dpi_scale))
            axes_right_frac, x_fracs = sliders.right_margin_layout(
                fig_w_px, len(present), right_overhang_px,
                col_px=col_px, pad_px=pad_px, track_px=track_px)
            # Safety floor/ceiling: an extreme overhang measurement (or a pathologically narrow
            # window) should never collapse the plot to zero/negative width or claim the whole
            # figure for it -- clamped here, in the same value that both subplots_adjust and the
            # slider positioning below use, so they can't disagree.
            axes_right_frac = min(max(axes_right_frac, 0.3), 0.95)
            bottom_frac, track_y0_frac = sliders.bottom_margin_layout(
                fig_h_px, default_track_y0_px, track_h_px, text_below_px,
                bottom_overhang_px, label_gap_px)

            self.fig.subplots_adjust(left=0.10, right=axes_right_frac, bottom=bottom_frac,
                                     top=0.90)

            new_right_overhang_px = self._measure_overhang_px()
            new_bottom_overhang_px = self._measure_bottom_overhang_px()
            settled = (abs(new_right_overhang_px - right_overhang_px) < _LAYOUT_SETTLE_PX
                      and abs(new_bottom_overhang_px - bottom_overhang_px) < _LAYOUT_SETTLE_PX)
            right_overhang_px, bottom_overhang_px = new_right_overhang_px, new_bottom_overhang_px
            if settled:
                break

        track_w_frac = track_px / fig_w_px
        for slider, x_frac in zip(present, x_fracs):
            # The vertical span tracks the plot's own (bottom_frac..0.90), not a fixed
            # 0.16..0.90, so it stays aligned with the plot even when bottom_frac grows past
            # 0.16 to clear a short figure's x-slider text (see bottom_margin_layout).
            slider.ax.set_position([x_frac, bottom_frac, track_w_frac, 0.90 - bottom_frac])
        if self._x_slider is not None:
            self._x_slider.ax.set_position(
                [0.10, track_y0_frac, axes_right_frac - 0.10, track_h_px / fig_h_px])

    def _get_renderer(self):
        """The canvas's current renderer, or ``None`` if one isn't available yet (e.g. a canvas
        not yet realized) -- shared by every ``_layout_sliders`` measurement helper so each one
        degrades to its own fallback constant the same way, rather than raising."""
        try:
            return self.fig.canvas.get_renderer()
        except Exception:
            return None

    def _measure_overhang_px(self):
        """Max, over the primary Axes, its twin, and the gfunction d2P/dG2 axis, of how far that
        Axes' own tightbbox extends past the primary Axes' right edge, in device pixels. Falls
        back to a fixed guess when no renderer is available yet (e.g. a canvas not yet realized),
        or when an Axes' own tightbbox comes back ``None`` (matplotlib can return that for some
        empty/degenerate Axes), rather than raising -- ``_layout_sliders`` still runs, just
        against a guess until the next resize/refresh gives it a real measurement."""
        renderer = self._get_renderer()
        if renderer is None:
            return _FALLBACK_OVERHANG_PX * self.fig.dpi / _LAYOUT_DPI_REF
        overhangs = []
        # The primary Axes keeps its full tightbbox (any title/legend/annotation could in
        # principle reach past its right edge). The twin and d2 axes carry nothing wider than
        # their y-axis (spine, ticks, tick labels, label), so measuring just that axis skips
        # re-walking their lines/patches/x axis -- same extent, a fraction of the cost.
        for ax, axis_only in ((self.ax, False), (self._twin_axes(), True),
                              (self._d2_axes(), True)):
            if ax is None:
                continue
            try:
                bbox = (ax.yaxis.get_tightbbox(renderer) if axis_only
                        else ax.get_tightbbox(renderer))
            except Exception:
                continue
            if bbox is None:
                continue
            overhangs.append(bbox.x1 - self.ax.bbox.x1)
        return (max(overhangs) if overhangs
                else _FALLBACK_OVERHANG_PX * self.fig.dpi / _LAYOUT_DPI_REF)

    def _measure_bottom_overhang_px(self):
        """How far the primary Axes' own x-axis tick labels + xlabel extend below its bottom
        edge, in device pixels -- the bottom-margin counterpart of ``_measure_overhang_px``
        (mirroring it, including the same no-renderer/``None``-tightbbox fallback). Measures
        the x axis alone: nothing else on a step's Axes is drawn below it."""
        renderer = self._get_renderer()
        if renderer is None:
            return _FALLBACK_BOTTOM_OVERHANG_PX * self.fig.dpi / _LAYOUT_DPI_REF
        try:
            bbox = self.ax.xaxis.get_tightbbox(renderer)
        except Exception:
            bbox = None
        if bbox is None:
            return _FALLBACK_BOTTOM_OVERHANG_PX * self.fig.dpi / _LAYOUT_DPI_REF
        return max(self.ax.bbox.y0 - bbox.y0, 0.0)

    def _measure_slider_text_col_px(self, present, dpi_scale):
        """Widest rendered split-text width among the given vertical sliders' hi/lo texts, plus
        a dpi-scaled side pad on each side -- sizing a slider's column from its own real content
        (e.g. a log-scaled slider's "1.3e+03") rather than a flat guess that can be too narrow
        for it and collide with its neighbor. Returns 0.0 (leaving the caller's dpi-scaled
        default floor untouched) when there's nothing to measure yet (no sliders, or no
        renderer available)."""
        renderer = self._get_renderer()
        if renderer is None:
            return 0.0
        widths = []
        for slider in present:
            for text in (slider._hi_text, slider._lo_text):
                try:
                    bbox = text.get_window_extent(renderer)
                except Exception:
                    continue
                if bbox is not None:
                    widths.append(bbox.width)
        if not widths:
            return 0.0
        return max(widths) + 2.0 * sliders.COLUMN_TEXT_PAD_PX * dpi_scale

    def _reset_view(self):
        self._views[self.step] = None
        self.refresh()

    def _attach_controllers(self):
        for c in self._controllers:
            c.disconnect()
        self._controllers = []

        step = self.step
        if step == "overview":
            res = self.res
            mask_hint = ""
            # One gate for every Overview gesture: a Shift/Ctrl press near the trim line goes to
            # the mask/keep span (connected first), a plain press to the trim drag.
            gate = picks._CaptureGate()
            if self.td is not None:

                def _commit_band(kind):
                    def on_span(lo_h, hi_h):
                        picks.commit_mask_interval(self.state, kind, lo_h * 3600.0, hi_h * 3600.0)
                        self.refresh()
                    return on_span

                def get_spans():
                    return [(kind, i, lo / 3600.0, hi / 3600.0)
                            for kind, ivs in (("mask", self.state.mask_intervals),
                                              ("keep", self.state.keep_intervals))
                            for i, (lo, hi) in enumerate(ivs)]

                def on_remove(kind, idx):
                    picks.remove_interval(self.state, kind, idx)
                    self.refresh()

                self._controllers.append(picks.ModifierSpanController(
                    self.canvas, self.ax, _commit_band("mask"), modifier="shift",
                    exclude=("ctrl",), gate=gate))
                self._controllers.append(picks.ModifierSpanController(
                    self.canvas, self.ax, _commit_band("keep"), modifier="ctrl",
                    exclude=("shift",), gate=gate))
                self._controllers.append(picks.IntervalRemoveController(
                    self.canvas, self.ax, get_spans, on_remove, gate=gate))
                mask_hint = (" Shift+drag to mask a glitch, Ctrl+drag to keep auto-masked data, "
                             "right-click a band to remove it.")
            if (res.resampled_full is not None and res.t_shutin_s is not None
                    and len(res.resampled_full.dt)):
                dt_full = res.resampled_full.dt
                t_shutin_s = res.t_shutin_s
                guard_dt = res.resampled_full.guard_dt
                post_mask = self.td.t_s >= res.t_shutin_s
                raw_dt_post = self.td.t_s[post_mask] - res.t_shutin_s

                def commit_trim(x_hours):
                    dt_target = x_hours * 3600.0 - t_shutin_s
                    if guard_dt is not None and dt_target > guard_dt:
                        # Deliberate override: snap against the RAW record, not the resampled
                        # grid -- the resampler may have kept nothing new past the guard at all
                        # (e.g. a permanently-elevated tail), but the analyst can still choose to
                        # include it. Never clear to None here even at the raw record's last
                        # sample: clearing would fall back through resolve_tail_cut_dt to
                        # guard_dt again, defeating the whole point of the override.
                        idx = picks._nearest(raw_dt_post, dt_target)
                        dt = float(raw_dt_post[idx])
                    else:
                        # dt_target is at/before the guard (or there's no guard at all): snap to
                        # the nearest RAW sample, not a 30-psi kept point. Kept points can be
                        # hours apart on a slow late falloff, which left a drag only two places
                        # to land (AEF 05-61-34-5649B). Candidates are restricted to strictly
                        # before guard_dt so a drag at/before the guard never crosses into
                        # override territory.
                        if guard_dt is None:
                            kept, candidates = dt_full, raw_dt_post
                        else:
                            kept = dt_full[dt_full < guard_dt]
                            candidates = raw_dt_post[raw_dt_post < guard_dt]
                        if len(kept) < 3 or len(candidates) == 0:
                            dt = None  # no room for a trim that keeps >= 3 kept points
                        else:
                            idx = picks._nearest(candidates, dt_target)
                            if idx >= len(candidates) - 1:
                                dt = None  # released at/past the last sample: clear
                            else:
                                # never trim below 3 kept points
                                dt = max(float(candidates[idx]), float(kept[2]))
                    picks.commit_tail_trim(self.state, dt, guard_dt)
                    self.refresh()

                ctrl = picks.DragLineController(self.canvas, self.ax,
                                                handlers={"tail_trim": commit_trim},
                                                gate=gate)
                self._controllers.append(ctrl)
                self._controllers.append(picks.HoverCursorController(self.canvas, [ctrl]))
                hint = ("Drag the blue dashed line to trim a bad tail; release it at the right "
                        "edge to clear the trim.")
                if guard_dt is not None:
                    # The tail guard has fired for this record -- releasing at the right edge
                    # doesn't just clear back to the (already-guard-clamped) default here, it
                    # also overrides the guard by extending the cutoff to the record's end
                    # (though the resampler may have nothing new to report there -- see the
                    # warning shown after the drag).
                    hint = ("Drag the blue dashed line to trim a bad tail; release it at the "
                            "right edge to override the tail guard and extend the cutoff to "
                            "the end of the record.")
                self.hint_lbl.config(text=hint + mask_hint)
            else:
                self.hint_lbl.config(
                    text="Entire dataset. Trim tool unavailable until a shut-in/falloff exists."
                    + mask_hint)
        elif step == "injection":
            def _commit(idx_attr):
                def on_release(x_hours):
                    idx = picks._nearest(self.td.t_s / 3600.0, x_hours)
                    setattr(self.state, idx_attr, idx)
                    self.state.qmax_bpm = None  # re-derive from the new window
                    if idx_attr == "shutin_idx":
                        # tail_trim_dt/guard_dt both live in dt-from-shut-in space, so moving
                        # shut-in can move the auto trim's cut in absolute time (F1) -- resync it
                        # before refreshing. start_idx deliberately does NOT trigger this:
                        # resample_pressure_increment takes only (dt, p) built from shut-in
                        # onward, so a start-only change can't move tail_trim_dt or guard_dt.
                        picks.resync_auto_tail_trim(self.state, self.td, compute_all(self.state,
                                                                                     self.td))
                    self.refresh()
                return on_release
            drag_ctrl = picks.DragLineController(
                self.canvas, self.ax,
                handlers={"start": _commit("start_idx"),
                          "shutin": _commit("shutin_idx")})
            self._controllers.append(drag_ctrl)
            self._controllers.append(picks.HoverCursorController(self.canvas, [drag_ctrl]))
            self.hint_lbl.config(
                text="Drag the injection-start and shut-in lines to adjust the window."
                + ("" if self._injection_full
                   else " Use Show all data to find a missed injection."))
        elif step == "isip":
            res = self.res
            step_ctrls = []
            if (res.bhp_all is not None and res.t_shutin_s is not None
                    and not self.state.isip_at_shutin):
                t_min = (self.td.t_s - res.t_shutin_s) / 60.0
                gate = picks._CaptureGate()

                def get_pick():
                    return _isip_pick_in_minutes(self.state.isip_tangent, res.t_shutin_s)

                def commit(kind, anchor_x, anchor_y, slope):
                    sec_x, sec_slope = _isip_minutes_to_seconds(anchor_x, slope, res.t_shutin_s)
                    picks.commit_isip_tangent(self.state, self.td, res, kind, sec_x, anchor_y,
                                              sec_slope)
                    self.refresh()

                def readout(kind, anchor_x, anchor_y, slope):
                    isip = anchor_y - slope * anchor_x  # value at x=0 -- shut-in on this axes
                    return f"ISIP ≈ {isip:.0f} psi"

                step_ctrls.append(picks.AnchorLineController(
                    self.canvas, self.ax,
                    gids={"segment": "isip_tangent_segment", "tick": "isip_tangent_tick",
                          "extension": "isip_tangent_extension", "point": "isip_value_dot"},
                    get_pick=get_pick, commit_fn=commit, curve=(t_min, res.bhp_all),
                    anchor_half=interpret.ISIP_ANCHOR_HALF, readout_fn=readout, gate=gate,
                    pin_x=0.0))
            self._controllers.extend(step_ctrls)
            if step_ctrls:
                self._controllers.append(picks.HoverCursorController(self.canvas, step_ctrls))
            if self.state.isip_at_shutin:
                self.hint_lbl.config(
                    text="Apparent ISIP is the pressure at the shut-in line. Move shut-in on the "
                         "Injection step to change it.")
            else:
                self.hint_lbl.config(
                    text="Drag the anchor along the curve, the body to pan, or an end to rotate "
                         "the ISIP tangent.")
        elif step == "gfunction":
            res = self.res
            ax2 = self._twin_axes()
            step_ctrls = []
            scenario = self.state.closure_scenario
            # One gate shared by the two point controllers below (e.g. the min-dP/dG triangle and
            # the contact marker, whose hit zones can sit close together on screen).
            gate = picks._CaptureGate()
            if res.diagnostics is not None and res.resampled is not None and ax2 is not None:
                G, p, dPdG = res.diagnostics.G, res.resampled.p, res.diagnostics.dPdG

                def commit_min_dpdg(x):
                    # The triangle is the analyst's control point (decision D4): committing its
                    # drag re-derives the contact from the new anchor under the active scenario
                    # before refreshing, same re-assert-hint-after-refresh pattern as
                    # _on_scenario (ui.py:438-464) -- refresh() resets hint_lbl to the step's
                    # default text, so a re-derive failure hint must be applied after it.
                    picks.commit_min_dpdg_point(self.state, x)
                    hint = picks.re_derive_contact_from_min(self.state, res)
                    self.refresh()
                    if hint:
                        self.hint_lbl.config(text=hint)

                def commit_point(x):
                    picks.commit_contact_point(self.state, x)
                    self.refresh()

                def on_min_dpdg_window(lo, hi):
                    # Shift+drag window correction: handle_min_dpdg_window is the complete commit
                    # for this gesture -- it moves the triangle within [lo, hi] AND sets the
                    # contact pick itself (its own window-scoped rule, not the g_min-masked
                    # re_derive_contact_from_min), so there is nothing left to re-derive here.
                    hint = picks.handle_min_dpdg_window(self.state, res, lo, hi)
                    self.refresh()
                    if hint:
                        self.hint_lbl.config(text=hint)

                # min-dP/dG-first ordering preserved (tests unpack step_ctrls by position) --
                # the triangle only applies to C-A (rel-min anchor) / C-B (inflection seed); the
                # contact marker applies to every scenario outside NO_CONTACT_SCENARIOS (C-C, C-D,
                # C-X), which have no contact rule at all (see the CLAUDE.md closure-scenario table).
                if scenario.startswith(("C-A", "C-B")):
                    # Registered before the point controllers below so a Shift-press claims the
                    # shared gate first -- an ordinary (unmodified) press never captures here, so
                    # the point controllers still win a plain drag.
                    step_ctrls.append(picks.ModifierSpanController(
                        self.canvas, ax2, on_min_dpdg_window, gate=gate))
                    step_ctrls.append(picks.DraggablePointController(
                        self.canvas, ax2, "min_dpdg_point", G, dPdG, commit_fn=commit_min_dpdg,
                        gate=gate))
                if not scenario.startswith(NO_CONTACT_SCENARIOS):
                    step_ctrls.append(picks.DraggablePointController(
                        self.canvas, self.ax, "contact_point", G, p, commit_fn=commit_point,
                        gate=gate))
            self._controllers.extend(step_ctrls)
            if step_ctrls:
                self._controllers.append(picks.HoverCursorController(self.canvas, step_ctrls))
            self.hint_lbl.config(text=picks.gfunction_hint_text(scenario))
        elif step == "tangent":
            res = self.res
            ax2 = self._twin_axes()
            step_ctrls = []
            if self.state.tangent_uninterpretable:
                # No line or marker to drag -- same treatment as stiffness_no_upturn below.
                self.hint_lbl.config(
                    text="Tangent marked uninterpretable -- uncheck to restore the pick.")
                return
            if res.diagnostics is not None and res.resampled is not None and ax2 is not None:
                dg = res.diagnostics
                gate = picks._CaptureGate()

                def get_closure_pick():
                    if self.state.closure_slope is None:
                        return None
                    return TangentPick(anchor_x=0.0, anchor_y=0.0, slope=self.state.closure_slope)

                def commit_line(kind, anchor_x, anchor_y, slope):
                    picks.commit_closure_line(self.state, res, kind, anchor_x, anchor_y, slope)
                    self.refresh()

                def commit_point(x):
                    picks.commit_closure_point(self.state, x)
                    self.refresh()

                # marker+vline first: the whole line body is a rotate hit zone for the
                # through-origin line, and the closure marker/vline sit on the primary axis where
                # they cross it on screen -- press/hover priority follows this order, so the
                # marker/vline must win before the through-origin line claims the shared gate
                step_ctrls.append(picks.DraggablePointController(
                    self.canvas, self.ax, "closure_point", dg.G, res.resampled.p,
                    commit_fn=commit_point, vline_gid="closure_vline", gate=gate))
                step_ctrls.append(picks.AnchorLineController(
                    self.canvas, ax2, gids={"segment": "closure_line_segment"},
                    get_pick=get_closure_pick, commit_fn=commit_line, curve=None,
                    allow_anchor=False, allow_body=False, gate=gate))
            self._controllers.extend(step_ctrls)
            if step_ctrls:
                self._controllers.append(picks.HoverCursorController(self.canvas, step_ctrls))
            self.hint_lbl.config(
                text="Rotate the through-origin line (the closure marker follows); "
                         "drag the closure marker or its vertical line.")
        elif step == "loglog" and pp_from_peak(self.state):
            # PC-E: no window; drag the peak marker the -1/2 extrapolation starts from.
            res = self.res
            dg = res.diagnostics
            if dg is not None and res.pce_peak_t is not None:
                good = (dg.t > 0) & (dg.tdpdt > 0)

                def commit_peak(x):
                    picks.commit_pce_peak(self.state, x)
                    self.refresh()

                ctrl = picks.DraggablePointController(
                    self.canvas, self.ax, "pce_peak", dg.t[good], dg.tdpdt[good],
                    commit_fn=commit_peak)
                self._controllers.extend([ctrl, picks.HoverCursorController(self.canvas, [ctrl])])
            self.hint_lbl.config(
                text=f"{self.state.postclosure_scenario}: drag the peak marker; "
                     "pore pressure = P_peak - 2*(t*dP/dt)_peak.")
        elif step == "loglog" and loglog_window_suppressed(self.state):
            # PC-F draws no window, so a drag would set a pick the analyst never sees.
            self.hint_lbl.config(
                text=f"No log-log window under {self.state.postclosure_scenario}; "
                     "pick another scenario to select one.")
        elif step == "loglog":
            def on_span(lo, hi):
                picks.handle_loglog_span(self.state, lo, hi)
                slope = compute_all(self.state, self.td).loglog_slope
                hint = picks.auto_assign_postclosure(self.state, slope)
                self.var_pcscen.set(self.state.postclosure_scenario)
                self.refresh()
                if hint:
                    # After refresh(): _attach_controllers resets the step's default hint.
                    self.hint_lbl.config(text=hint)
            self._controllers.append(picks.SpanController(self.ax, on_span))
            if self.state.postclosure_auto:
                self.hint_lbl.config(text=picks.postclosure_auto_hint(self.res.loglog_slope))
            else:
                self.hint_lbl.config(
                    text="Drag to select the late-time window; set postclosure scenario.")
        elif step == "porepressure" and pp_from_peak(self.state):
            self.hint_lbl.config(
                text="PC-E: pore pressure is the -1/2 line from the Log-log peak; no window.")
        elif step == "porepressure":
            def on_span(lo, hi):
                picks.handle_pp_span(self.state, lo, hi)
                self.refresh()
            self._controllers.append(picks.SpanController(self.ax, on_span))
            self.hint_lbl.config(text="Drag to select the late-time window; choose the axis.")
        elif step == "stiffness":
            res = self.res
            if closure_uninterpretable(self.state):
                self.hint_lbl.config(
                    text="G-function marked uninterpretable (C-X) -- no stiffness Shmin is "
                         "reported.")
            elif self.state.stiffness_no_upturn:
                # No line to drag -- the option is an explicit "there is no upturn to pick".
                self.hint_lbl.config(
                    text="No slope change apparent -- uncheck to restore the pick line.")
            elif res.stiffness_S is not None:
                p_eff = res.stiffness_p_eff[1:]

                def commit_stiffness(x):
                    # DragLineController commits the raw dragged x -- snap it to the nearest
                    # curve sample's pressure first (picks._nearest is a plain argmin over |diff|,
                    # so it works fine regardless of p_eff's order -- no longer guaranteed
                    # non-increasing now that rs.p can rise as well as fall).
                    idx = picks._nearest(p_eff, x)
                    picks.commit_stiffness_point(self.state, float(p_eff[idx]))
                    self.refresh()

                ctrl = picks.DragLineController(self.canvas, self.ax,
                                                handlers={"stiffness_pick": commit_stiffness})
                self._controllers.append(ctrl)
                self._controllers.append(picks.HoverCursorController(self.canvas, [ctrl]))
                self.hint_lbl.config(
                    text="Drag the blue dashed line to the stiffness upturn (fracture-wall "
                         "contact).")
            else:
                self.hint_lbl.config(
                    text="Stiffness plot unavailable until the min-dP/dG pick and a "
                         "pore-pressure estimate exist.")

    def _update_panel(self):
        r = self.res
        def s(v, f="{:.0f}"):
            return f.format(v) if v is not None else "-"
        # C-D rapid closure has no contact pick, so no compliance Shmin -- shmin_rapid stands in for
        # it in the same row, marked approximate by the label asterisk alone; the value column's
        # own "±75" half-range (from format_shmin_rapid's short form) is what signals the
        # approximation there, rather than sitting in a row of its own that read like a fourth
        # stress method. compute_all only ever sets shmin_rapid for C-D, and C-D always clears the
        # contact, so the two are never both set.
        use_rapid = r.shmin_compliance is None and r.shmin_rapid is not None
        vals = {
            "te (min)": s(r.te_s / 60 if r.te_s else None, "{:.2f}"),
            "Vinj (bbl)": s(r.vinj, "{:.1f}"),
            "qmax (bpm)": s(r.qmax_bpm, "{:.2f}"),
            "apparent ISIP": s(r.apparent_isip),
            "eff ISIP (compliance)": s(r.effective_isip_compliance),
            "NWB complexity": s(r.near_wellbore_complexity),
            "contact P": s(r.contact_pressure),
            "Shmin compliance": (interpret.format_shmin_rapid(r.shmin_rapid) if use_rapid
                                  else s(r.shmin_compliance)),
            "Shmin tangent": s(r.shmin_tangent),
            "Shmin variable": s(r.shmin_variable),
            "Shmin Liberty": s(r.shmin_liberty),
            "tc compliance (min)": s(r.closure_time_compliance_s / 60
                                      if r.closure_time_compliance_s is not None else None, "{:.2f}"),
            "tc tangent (min)": s(r.closure_time_tangent_s / 60
                                   if r.closure_time_tangent_s is not None else None, "{:.2f}"),
            "tc variable (min)": s(r.closure_time_variable_s / 60
                                    if r.closure_time_variable_s is not None else None, "{:.2f}"),
            "net (compliance)": s(r.net_pressure_compliance),
            "net (tangent)": s(r.net_pressure_tangent),
            "net (variable)": s(r.net_pressure_variable),
            "delta closure": s(r.delta_closure),
            "pore pressure": s(r.pore_pressure),
            "Shmin stiffness": s(r.shmin_stiffness),
        }
        for k, v in vals.items():
            self.value_lbls[k].config(
                text=v if summary.visited(self.state, FIELD_STEP[k]) else "-")
        # The asterisk tracks the value: gate it on the same not_visited check the loop above applies.
        gf_visited = summary.visited(self.state, "gfunction")
        self.name_lbls["Shmin compliance"].config(
            text="Shmin compliance*" if (use_rapid and gf_visited) else "Shmin compliance")
        # Complexity is referenced to the shared eff ISIP; mark it when that fell back to the tangent
        # one (C-C/C-D clear the contact, so there is no compliance eff ISIP to reference). Same
        # not_visited gate as the value, so the asterisk can never sit next to a "-".
        use_tangent_ref = (r.near_wellbore_complexity is not None
                           and r.net_pressure_isip_source == "tangent")
        self.name_lbls["NWB complexity"].config(
            text="NWB complexity*" if (use_tangent_ref and gf_visited) else "NWB complexity")
        self._issue_sections = issue_sections(r)
        self._warnings_expanded = False
        self._warn_lbl_hidden = False
        self.warn_lbl.config(text=format_warnings_text(self._issue_sections, False),
                             cursor="hand2" if self._issue_sections else "")

    def _toggle_warnings(self, event=None):
        """Click handler for warn_lbl: flips the collapsed/expanded summary in place, no
        recompute. No-ops on a click when there's nothing to show, or on Overview, where
        _update_issues_panel lists everything and blanks this label."""
        if not self._issue_sections or getattr(self, "_warn_lbl_hidden", False):
            return
        self._warnings_expanded = not self._warnings_expanded
        self.warn_lbl.config(text=format_warnings_text(self._issue_sections,
                                                        self._warnings_expanded))

    def _update_issues_panel(self):
        """Overview only: list every issue, grouped by level, in the slot the Results rows use on
        other steps, and blank the collapsed warn_lbl summary (it would repeat the list)."""
        if self.step != "overview":
            return
        for child in self.frm_issues.winfo_children():
            child.destroy()
        wrap = self._panel_wrap
        ttk.Label(self.frm_issues, text="Data checks", font=("", 10, "bold"),
                  wraplength=wrap).pack(anchor="w")
        if not self._issue_sections:
            ttk.Label(self.frm_issues, text="No warnings.", foreground=C.UI_MUTED,
                      wraplength=wrap).pack(anchor="w")
        colors = {title: color for _, title, _, color, _ in ISSUE_LEVELS}
        for title, msgs in self._issue_sections:
            ttk.Label(self.frm_issues, text=title, foreground=colors[title],
                      font=("", 9, "bold"), wraplength=wrap).pack(anchor="w", pady=(6, 0))
            for m in msgs:
                ttk.Label(self.frm_issues, text=f"\u2022 {m}", wraplength=wrap,
                          justify="left").pack(anchor="w", fill="x")
        if self._quest_lines:
            ttk.Label(self.frm_issues, text="Questionnaire", foreground=C.UI_MUTED,
                      font=("", 9, "bold"), wraplength=wrap).pack(anchor="w", pady=(6, 0))
            warn_color = next(c for key, _, _, c, _ in ISSUE_LEVELS if key == "warnings")
            for line in self._quest_lines:
                if not line.startswith(_QUEST_WARNING_PREFIX):
                    ttk.Label(self.frm_issues, text=f"• {line}", wraplength=wrap,
                              justify="left").pack(anchor="w", fill="x")
                    continue
                # ttk.Label can't mix fonts, so the bold prefix and the wrapping message are two
                # labels in a row frame; _on_panel_configure narrows the message by the prefix.
                row = ttk.Frame(self.frm_issues)
                row.pack(anchor="w", fill="x")
                prefix = ttk.Label(row, text=f"• {_QUEST_WARNING_PREFIX.strip()}",
                                   foreground=warn_color, font=("", 9, "bold"))
                prefix.pack(side="left", anchor="n")
                ttk.Label(row, text=line[len(_QUEST_WARNING_PREFIX):], justify="left",
                          wraplength=max(wrap - prefix.winfo_reqwidth() - 4, 50)
                          ).pack(side="left", anchor="n", padx=(4, 0))
        self.warn_lbl.config(text="", cursor="")
        self._warn_lbl_hidden = True

    def _update_unit_labels(self):
        """Gray "(detected unit)" hint beside each of the three unit dropdowns, from
        `td.unit_detections` -- refreshed every `refresh()` call (compute_all just rebuilt that
        cache) so it always reflects the current channel mapping/override, not just the
        load-time guess."""
        if self.td is None:
            return
        for col, lbl in ((self.state.pressure_col, self.lbl_pressure_unit),
                         (self.state.rate_col, self.lbl_rate_unit),
                         (self.state.volume_col, self.lbl_volume_unit)):
            det = self.td.unit_detections.get(col) if col else None
            lbl.config(text=f"({det.unit})" if det is not None else "")

    # ---- persistence ----------------------------------------------------------------------------
    def _save_picks(self):
        if self.td is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=[("JSON", "*.json")])
        if path:
            self.state.notes = self.txt_notes.get("1.0", "end").strip()
            # Capture any unapplied entry-widget edits so they aren't silently dropped on save.
            self._sync_state_from_widgets()
            self.state.to_json(path)

    def _finish(self):
        """Bound to the Finish button (_advance on the last step): a silent one-click export.

        Finish means "I completed this test", so it un-parks a whole-test Skip (clears
        explicit_status below) -- a test that was Skip-test-parked and later walked to
        completion should report done, not a stale skipped. Per-step "skipped" entries from
        "Skip >" are a different thing (a decision about that step, not the whole test) and are
        preserved, not overwritten by the "done" stamp below -- so a test finished with a
        skipped step still reports skipped overall.

        Single-file mode (current_entry is None): byte-for-byte today's behavior -- re-save the
        picks JSON next to the data file and write a PNG per step (current zoom) into a sibling
        subfolder; no log, no advance. Folder mode: save picks via store.save_picks_for (the
        store path is the single ground truth in folder mode -- no <stem>_picks.json
        duplicate), same PNG export, the dfit_log.csv row write, then _advance_queue() to the
        next "new" test.

        No success popup; only a failure raises a dialog."""
        if self.td is None:
            return
        # Preserve a skip recorded by "Skip >" on this step (reachable here on the last step,
        # or on loglog under PC-F where _goto redirects a porepressure destination back to
        # loglog) -- Finish must not silently promote an already-skipped step back to "done".
        if self.state.step_status.get(self.step) != "skipped":
            self.state.step_status[self.step] = "done"
        # Un-park a whole-test Skip: Finish is "I completed this test", so clear the flag before
        # saving so it doesn't survive into the picks JSON or feed a stale status below. Set
        # unconditionally rather than branching on folder/single-file mode -- single-file mode
        # never sets this field, so clearing it there is a no-op.
        self.state.explicit_status = None
        # Capture any unapplied entry-widget edits before refreshing, so self.res (feeding the
        # PNG export and, in folder mode, the log row below) reflects the synced state.
        self._sync_state_from_widgets()
        self.refresh()  # ensure current-step view stored in _views and self.res fresh
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        entry = self.current_entry
        parent = pathlib.Path(self.td.path).parent
        stem = pathlib.Path(self.td.path).stem
        try:
            if entry is None:
                self.state.to_json(str(parent / f"{stem}_picks.json"))
            else:
                store.save_picks_for(entry, self.state)
            out_dir = parent / f"{stem} DFIT plots"
            out_dir.mkdir(exist_ok=True)
            views = {k: ((v.xlim, v.ylim, v.y2lim) if v is not None else None)
                     for k, v in self._views.items()}
            plots.save_all_step_pngs(str(out_dir), self.td, self.state, self.res, views)
        except Exception as e:
            messagebox.showerror("Finish failed", str(e))
            return
        if entry is None:
            return
        entry.status = store.status_for(self.state)
        try:
            self._write_log_row(entry)
        except Exception as e:
            messagebox.showerror("Log write failed", str(e))
        self._refresh_queue_row(entry)
        self._advance_queue()

    def _apply_loaded_state(self, state: PickState):
        """Adopt `state` as the current PickState and reflect it into every widget -- shared by
        single-file _load_picks (state fresh off a file dialog) and folder-mode _load_test
        (state fresh off store.load_picks_for)."""
        self.state = state
        if getattr(self, "td", None) is not None:
            # Injection picks from a different/longer file: re-seed now, since seed_injection
            # otherwise runs only on the step's first visit and a stale index crashes the plots.
            picks.reseed_out_of_range_injection(self.state, self.td)
        self._views = {k: None for k, _ in STEPS}
        self._injection_full = False
        if not self.state.step_status:
            # An old save has real picks but no breadcrumb history -- infer it so the
            # breadcrumb doesn't present the whole workflow as unreached.
            self.state.step_status = infer_step_status(self.state)
        # reflect into widgets
        self.var_pressure.set(self.state.pressure_col)
        self.var_rate.set(self.state.rate_col or "")
        self.var_volume.set(self.state.volume_col or "")
        self.var_isbhp.set(self.state.pressure_is_bhp)
        self.var_pressure_unit.set(self.state.pressure_unit)
        self.var_rate_unit.set(self.state.rate_unit)
        self.var_volume_unit.set(self.state.volume_unit)
        self.var_density.set("" if self.state.density_ppg is None else str(self.state.density_ppg))
        self.var_tvd.set("" if self.state.tvd_ft is None else str(self.state.tvd_ft))
        self.var_well.set(self.state.well_name)
        self.var_formation.set(self.state.formation)
        # Density/TVD above now came from the picks file, not the questionnaire that was auto-
        # detected (if any) when the CSV was loaded -- clear the stale provenance label so it
        # doesn't misattribute these values.
        self._quest_lines = []
        self.var_alpha.set(str(self.state.alpha))
        self.var_step.set(str(self.state.resample_step))
        self.var_cscen.set(self.state.closure_scenario)
        self.var_pcscen.set(self.state.postclosure_scenario)
        self.var_ppaxis.set(self.state.pp_axis)
        self.var_showd2.set(self.state.show_d2pdg2)
        self.var_stiffness_no_upturn.set(self.state.stiffness_no_upturn)
        self.var_tangent_uninterpretable.set(self.state.tangent_uninterpretable)
        self.var_isip_at_shutin.set(self.state.isip_at_shutin)
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", self.state.notes)
        # Resume at the first not-yet-visited step so the breadcrumb picks up where the saved
        # workflow left off; if every step already has some status, there is no natural resume
        # point, so land on "overview" (first_not_visited_step's fallback).
        self._goto(first_not_visited_step(self.state.step_status))

    def _load_picks(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            state = PickState.from_json(path)
        except Exception as e:
            messagebox.showerror("Load picks failed", f"Could not read {path}: {e}")
            return
        self._apply_loaded_state(state)


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _num(text, current):
    """Parse a numeric entry, preserving `current` on unparseable input.

    An explicitly emptied box clears the value (returns None); a non-empty
    but garbled/partial string (e.g. mid-keystroke "8.") leaves `current`
    untouched rather than nulling out a previously-good value. Shared by
    `_sync_state_from_widgets` and `_apply_config`, so Apply on a garbled
    number also preserves the prior value instead of clearing it.
    """
    if text is None:
        return None
    if isinstance(text, str):
        text = text.strip()
        if not text:
            return None
    v = _to_float(text)
    return v if v is not None else current
