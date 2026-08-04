"""Tkinter/ttk shell: file open, channel mapping, the six-step canvas, live value panel.

Hosts the matplotlib canvas and wires the per-step pickers from picks.py to a recompute+redraw
loop. Holds no interpretation logic itself -- every number comes from model.compute_all.

There is no matplotlib toolbar: its sticky zoom/pan mode silently swallowed pick clicks/drags,
which is exactly the interaction this app depends on. View state (pan/zoom) is instead a
first-class per-step concept -- see ``ViewState`` and ``_views`` below -- restored on every
revisit to a step rather than only optionally preserved across one recompute.
"""

from __future__ import annotations

import math
import os
import pathlib
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import guide_content, interpret, io_load, picks, plots, sliders, store
from .model import (PickState, TangentPick, compute_all, infer_step_status,
                    porepressure_skipped, step_gate_error)
from .plots import D2_AXIS_GID, ViewDefaults
from .questionnaire import find_questionnaire, parse_questionnaire

_SLIDER_GID = "slider"

STEPS = [
    ("overview", "Overview"),
    ("isip", "Apparent ISIP"),
    ("gfunction", "G-function"),
    ("tangent", "Tangent"),
    ("loglog", "Log-log"),
    ("porepressure", "Pore pressure"),
]
CLOSURE_SCENARIOS = ["", "C-A clear", "C-B adequate", "C-C no-contact", "C-D rapid"]
POSTCLOSURE_SCENARIOS = ["", "PC-A linear", "PC-B false-radial",
                         "PC-C false radial to genuine linear",
                         "PC-D genuine linear to genuine radial",
                         "PC-E no trend", "PC-F no peak"]

# Advisory hints for the postclosure scenarios that don't fully dictate the pore-pressure axis
# (picks.suggest_pp_axis). Full explanatory text + figures live in the interpretation guide
# window (guide_content.py / _open_guide below); this dict is still consulted by _on_scenario.
_PC_HINTS = {
    "PC-D": "either axis valid -- choose t^(-1/2) or t^(-1) manually",
    "PC-E": "no clear slope -- t^(-1/2) set; treat pore pressure as low-confidence",
    "PC-F": "derivative still rising -- no reliable postclosure line; pore pressure step is "
            "skipped, Finish is available on this (log-log) step",
}

# Tabs for the single "Interpretation guide..." window (_open_guide), in display order.
GUIDE_TABS = [("closure", guide_content.CLOSURE_GUIDE), ("postclosure", guide_content.POSTCLOSURE_GUIDE)]
_GUIDE_ASSETS = pathlib.Path(__file__).parent / "assets" / "guide"

# The 19 result-panel rows, in display order -- module level (not just a literal inside
# _build_body) so FIELD_STEP below and tests can both refer to the same list.
PANEL_FIELDS = [
    "te (min)", "Vinj (bbl)", "qmax (bpm)", "apparent ISIP",
    "eff ISIP (compliance)", "NWB complexity",
    "contact P", "Shmin compliance", "Shmin tangent", "Shmin variable", "Shmin rapid",
    "tc compliance (min)", "tc tangent (min)", "tc variable (min)",
    "net (compliance)", "net (tangent)", "net (variable)",
    "delta closure", "pore pressure",
]

# Which step "owns" each panel field -- _update_panel shows "-" for a field whose step is still
# not_visited, even if compute_all already produced a value for it (e.g. a value carried over
# from a loaded JSON pick file the user hasn't actually visited yet this session). The three
# variable-method rows are guarded in compute_all on both the contact and closure picks, so they
# stay None (and display "-") until the gfunction step has actually been visited -- same
# precedent as "delta closure" owning only "tangent".
FIELD_STEP = {
    "te (min)": "overview",
    "Vinj (bbl)": "overview",
    "qmax (bpm)": "overview",
    "apparent ISIP": "isip",
    "eff ISIP (compliance)": "gfunction",
    # Needs the isip pick (apparent ISIP) and the gfunction pick (the reference eff ISIP);
    # gfunction is the later of the two, same precedent as "net (compliance)".
    "NWB complexity": "gfunction",
    "contact P": "gfunction",
    "Shmin compliance": "gfunction",
    "Shmin rapid": "gfunction",
    "tc compliance (min)": "gfunction",
    "net (compliance)": "gfunction",
    "Shmin tangent": "tangent",
    "tc tangent (min)": "tangent",
    "net (tangent)": "tangent",
    "delta closure": "tangent",
    "Shmin variable": "tangent",
    "tc variable (min)": "tangent",
    "net (variable)": "tangent",
    "pore pressure": "porepressure",
}


def step_index(key: str) -> int:
    """Position of ``key`` in ``STEPS``."""
    return next(i for i, (k, _) in enumerate(STEPS) if k == key)


def next_step(key: str) -> str:
    """The step after ``key``, or ``key`` itself if it is already the last one."""
    i = step_index(key)
    return STEPS[min(i + 1, len(STEPS) - 1)][0]


def prev_step(key: str) -> str:
    """The step before ``key``, or ``key`` itself if it is already the first one."""
    i = step_index(key)
    return STEPS[max(i - 1, 0)][0]


def first_not_visited_step(step_status: dict[str, str]) -> str:
    """Where ``_load_picks`` should land after loading a file: the first (in ``STEPS`` order)
    step that is still ``not_visited``, so the breadcrumb resumes wherever the saved workflow
    left off. If every step already has some status -- an old file whose picks cover the whole
    workflow -- there is no natural "resume point", so the simplest sensible fallback is the
    first step, "overview"."""
    for key, _ in STEPS:
        if step_status.get(key, "not_visited") == "not_visited":
            return key
    return "overview"


def _resolve_load_source(entry: store.TestEntry, saved: Optional[PickState]) -> str:
    """Which of ``entry.available_sources`` ``_load_test`` should open, absent an explicit
    ``source=`` override: the saved picks' ``active_source`` when there is a saved PickState
    naming a source that's actually available for this entry, else the first available source
    (``TestEntry.available_sources`` orders CSV before DBS when both exist)."""
    if saved is not None:
        for candidate in entry.available_sources:
            if candidate.lower() == saved.active_source:
                return candidate
    return entry.available_sources[0]


def _next_new_index(statuses: list[str], current_index: int) -> Optional[int]:
    """The index of the next ``"new"``-status entry in ``statuses``, scanning circularly
    starting just after ``current_index`` -- the pure selection logic behind Save & Next's
    auto-advance. Deliberately never revisits ``current_index`` itself even if its own status
    is ``"new"`` (its work was just saved this call), so "the only new entry is the current
    one" correctly reports no candidate. Returns None if ``statuses`` is empty or no other
    entry is ``"new"``."""
    n = len(statuses)
    for offset in range(1, n):
        i = (current_index + offset) % n
        if statuses[i] == "new":
            return i
    return None


@dataclass
class ViewState:
    """The resolved (non-optional) view actually applied to a step's Axes: primary xlim/ylim,
    and the twin axes' ylim if that step has one."""
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    y2lim: Optional[tuple[float, float]] = None


def _resolve_view(stored: Optional[ViewState], defaults: ViewDefaults,
                  full_x: tuple[float, float], full_y: tuple[float, float],
                  full_y2: Optional[tuple[float, float]]) -> ViewState:
    """First visit to a step (``stored`` is None): seed from the renderer's ``ViewDefaults``,
    falling back to the full autoscaled extent for whichever axis the renderer left ``None``.
    A revisit reuses the stored view unchanged, so user pan/zoom survives a recompute."""
    if stored is not None:
        return stored
    return ViewState(
        xlim=defaults.xlim if defaults.xlim is not None else full_x,
        ylim=defaults.ylim if defaults.ylim is not None else full_y,
        y2lim=defaults.y2lim if defaults.y2lim is not None else full_y2,
    )


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


class DfitApp:
    def __init__(self, root: tk.Tk, path: str | None = None):
        self.root = root
        self.root.title("DFIT interpretation (first build)")
        self.root.geometry("1400x850")

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
        self._guide_tab_index: dict[str, int] = {}

        # Folder mode: self.current_entry is the single mode flag -- None means single-file
        # mode (today's behavior, sidebar never packed). Task C adds dfit_log.csv writing on
        # top of log_df; this task only keeps it loaded.
        self.current_entry: store.TestEntry | None = None
        self.folder_root: str | None = None
        self.queue_entries: list[store.TestEntry] = []
        self.log_df = None

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
        ttk.Button(top, text="Open CSV…", command=self._open).pack(side="left")
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

        def combo(parent, label, var, width=24):
            ttk.Label(parent, text=label).pack(side="left", padx=(8, 2))
            c = ttk.Combobox(parent, textvariable=var, width=width, state="readonly")
            c.pack(side="left")
            return c

        self.cmb_pressure = combo(cfg, "Pressure:", self.var_pressure)
        ttk.Checkbutton(cfg, text="is BHP", variable=self.var_isbhp,
                        command=self._apply_config).pack(side="left", padx=4)
        self.cmb_rate = combo(cfg, "Rate:", self.var_rate, 20)
        self.cmb_volume = combo(cfg, "Volume:", self.var_volume, 18)

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

        # Folder mode only: mark the current test's queue status and roll it (plus every
        # computed value) into dfit_log.csv. Disabled/cleared in single-file mode --
        # _update_folder_controls is the one sync point for both widgets' state/values. Packed
        # right-to-left (each subsequent pack lands to the left of the one before it), so this
        # order reads left-to-right as "Mark: [combo] [Save & Next]", just left of Load/Save.
        self.btn_save_next = ttk.Button(cfg2, text="Save && Next", command=self._save_and_next,
                                        state="disabled")
        self.btn_save_next.pack(side="right", padx=4)
        self.var_mark = tk.StringVar()
        self.cmb_mark = ttk.Combobox(cfg2, textvariable=self.var_mark,
                                     values=["", "done", "skipped"], width=8, state="disabled")
        self.cmb_mark.pack(side="right")
        self.cmb_mark.bind("<<ComboboxSelected>>", lambda e: self._on_mark_change())
        ttk.Label(cfg2, text="Mark:").pack(side="right", padx=(8, 2))

        # Provenance for the density/TVD prefill above -- set by _load when a questionnaire xlsx
        # is auto-detected next to the CSV; empty when none was found. Density/TVD stay ordinary
        # editable entries either way, this is just so the user can see (and judge) the source. On
        # its own full-width row so long warning text isn't clipped by the buttons packed on cfg2.
        cfg3 = ttk.Frame(self.root, padding=(6, 2))
        cfg3.pack(side="top", fill="x")
        self.quest_lbl = ttk.Label(cfg3, text="", foreground="gray")
        self.quest_lbl.pack(fill="x", anchor="w")

    def _build_body(self):
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # left: folder-mode test queue -- built but never packed here. _show_queue()/_hide_queue()
        # (folder open / _load's exit-folder-mode path) own its visibility; single-file mode must
        # stay pixel-identical to today, so this frame starts hidden.
        self.queue_frame = ttk.Frame(body, width=240)
        self.queue_frame.pack_propagate(False)  # same fixed-width pattern as the right panel

        self.progress_lbl = ttk.Label(self.queue_frame, text="0/0", padding=(4, 4))
        self.progress_lbl.pack(side="top", fill="x")

        tree_frame = ttk.Frame(self.queue_frame)
        tree_frame.pack(side="top", fill="both", expand=True)
        self.queue_tree = ttk.Treeview(tree_frame, columns=("status",), show="tree headings")
        self.queue_tree.heading("#0", text="Test")
        self.queue_tree.heading("status", text="Status")
        self.queue_tree.column("#0", width=140)
        self.queue_tree.column("status", width=90, anchor="w")
        queue_vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.queue_tree.yview)
        self.queue_tree.configure(yscrollcommand=queue_vsb.set)
        self.queue_tree.pack(side="left", fill="both", expand=True)
        queue_vsb.pack(side="right", fill="y")
        self.queue_tree.bind("<<TreeviewSelect>>", self._on_queue_select)
        # Status is always readable as text (the "status" column); these tags are a secondary
        # color cue only, never the sole signal.
        self.queue_tree.tag_configure("done", foreground="#137333")
        self.queue_tree.tag_configure("skipped", foreground="gray")
        self.queue_tree.tag_configure("in_progress", foreground="#b35c00")
        self.queue_tree.tag_configure("new", foreground="black")

        # center: canvas
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)
        self._center_frame = center  # so _show_queue can pack the sidebar before= it
        self.fig = Figure(figsize=(9, 6))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=center)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # right: pick panel
        panel = ttk.Frame(body, padding=8, width=320)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        ttk.Label(panel, text="Results", font=("", 10, "bold")).pack(anchor="w")
        self.value_lbls: dict[str, ttk.Label] = {}
        for key in PANEL_FIELDS:
            row = ttk.Frame(panel); row.pack(fill="x")
            ttk.Label(row, text=key, width=16).pack(side="left")
            v = ttk.Label(row, text="-", width=14, anchor="e"); v.pack(side="right")
            self.value_lbls[key] = v

        ttk.Separator(panel).pack(fill="x", pady=6)

        # Closure-scenario and postclosure/pp-axis widgets are step-aware: only relevant once the
        # user has reached the step that produces the pick they annotate. Each cluster lives in
        # its own frame so _update_panel_visibility can pack/pack_forget it as a unit without
        # disturbing anything else in the panel. Neither frame is packed here -- refresh() ->
        # _update_panel_visibility() does that, always relative to sep_before_notes so re-showing
        # never reorders the panel.
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
        self.btn_gfunction_reset = ttk.Button(self.frm_cscen, text="Reset picks",
                                              command=self._on_reset_gfunction_picks)
        self.btn_gfunction_reset.pack(anchor="w", pady=(6, 0))
        ttk.Button(self.frm_cscen, text="Interpretation guide...",
                   command=lambda: self._open_guide("closure")).pack(anchor="w", pady=(6, 0))

        self.frm_pcscen = ttk.Frame(panel)
        ttk.Label(self.frm_pcscen, text="Postclosure scenario").pack(anchor="w")
        self.var_pcscen = tk.StringVar(value="")
        self.cmb_pcscen = ttk.Combobox(self.frm_pcscen, textvariable=self.var_pcscen,
                                       values=POSTCLOSURE_SCENARIOS, state="readonly")
        self.cmb_pcscen.pack(fill="x")
        self.cmb_pcscen.bind("<<ComboboxSelected>>", lambda e: self._on_scenario())

        ttk.Label(self.frm_pcscen, text="Pore-pressure axis").pack(anchor="w", pady=(6, 0))
        self.var_ppaxis = tk.StringVar(value="tm12")
        self.rb_ppaxis = []
        for txt, val in [("t^(-1/2)", "tm12"), ("t^(-1)", "tm1")]:
            rb = ttk.Radiobutton(self.frm_pcscen, text=txt, variable=self.var_ppaxis, value=val,
                                 command=self._on_scenario)
            rb.pack(anchor="w")
            self.rb_ppaxis.append(rb)

        ttk.Button(self.frm_pcscen, text="Interpretation guide...",
                   command=lambda: self._open_guide("postclosure")).pack(anchor="w", pady=(6, 0))

        self.sep_before_notes = ttk.Separator(panel)
        self.sep_before_notes.pack(fill="x", pady=6)
        ttk.Label(panel, text="Notes").pack(anchor="w")
        self.txt_notes = tk.Text(panel, height=5, width=36)
        self.txt_notes.pack(fill="x")
        self.hint_lbl = ttk.Label(panel, text="", wraplength=300, foreground="gray")
        self.hint_lbl.pack(anchor="w", pady=(6, 0))

    def _show_queue(self):
        """Pack the folder-mode sidebar leftmost, even though the center frame was already
        packed -- `before=` re-slots it ahead regardless of pack order."""
        self.queue_frame.pack(side="left", fill="y", before=self._center_frame)

    def _hide_queue(self):
        self.queue_frame.pack_forget()

    def _build_stepbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="bottom", fill="x")

        # Right side: Reset view, then the warning label at the far edge -- both pre-existing,
        # just not previously placed in the stepbar. Packed right-to-left, so pack the
        # rightmost-visually one (warn_lbl) first.
        self.warn_lbl = ttk.Label(bar, text="", foreground="red")
        self.warn_lbl.pack(side="right")
        ttk.Button(bar, text="Reset view", command=self._reset_view).pack(side="right", padx=4)

        # Left side: < Back, the six breadcrumbs, Next >, Skip >.
        ttk.Button(bar, text="< Back", command=self._back).pack(side="left", padx=2)
        self.step_buttons: dict[str, ttk.Button] = {}
        for key, label in STEPS:
            btn = ttk.Button(bar, text=label, command=lambda k=key: self._goto(k))
            btn.pack(side="left", padx=2)
            self.step_buttons[key] = btn
        self.next_btn = ttk.Button(bar, text="Next >", command=self._advance)
        self.next_btn.pack(side="left", padx=2)
        ttk.Button(bar, text="Skip >", command=self._skip).pack(side="left", padx=2)
        self.gate_lbl = ttk.Label(bar, text="", foreground="red")
        self.gate_lbl.pack(side="left", padx=8)

    # ---- data / config --------------------------------------------------------------------------
    def _open(self):
        path = filedialog.askopenfilename(
            filetypes=[
                ("DFIT data", "*.csv *.dbs"),
                ("CSV", "*.csv"),
                ("Fracpro DBS", "*.dbs"),
                ("All", "*.*"),
            ]
        )
        if path:
            self._load(path)

    def _load_common(self, path: str) -> bool:
        """Load `path` into a fresh PickState and land on "overview" -- shared by single-file
        _load and folder-mode _load_test. Returns False (leaving the previous self.td/state
        untouched) if the load failed, True on success."""
        try:
            self.td = io_load.load(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return False
        self.file_lbl.config(text=os.path.basename(path))
        cols = self.td.columns
        for cmb in (self.cmb_pressure, self.cmb_rate, self.cmb_volume):
            cmb["values"] = [""] + cols
        g = io_load.suggest_channels(cols)
        self.var_pressure.set(g["pressure"] or "")
        self.var_rate.set(g["rate"] or "")
        self.var_volume.set(g["volume"] or "")
        self.var_isbhp.set(bool(g["pressure_is_bhp"]))
        self.state = PickState()
        self._views = {k: None for k, _ in STEPS}
        self.var_cscen.set("")
        self.var_pcscen.set("")
        self.var_ppaxis.set("tm12")
        self.var_showd2.set(False)
        self.txt_notes.delete("1.0", "end")
        # Density/TVD are per-well; clear the stale previous well's values before (maybe)
        # prefilling from a questionnaire, so a well with no questionnaire doesn't inherit them.
        self.var_density.set("")
        self.var_tvd.set("")
        self.var_well.set("")
        self.var_formation.set("")
        self._load_questionnaire(path)
        self._sync_state_from_widgets()
        self._goto("overview")
        return True

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
        self.root.title("DFIT interpretation (first build)")
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

        # Surface scan warnings as a one-line summary rather than dialog-spamming per test.
        warn_count = sum(len(e.scan_warnings) for e in entries)
        entry_ids = {e.test_id for e in entries}
        orphan_ids = [tid for tid in log_df["test_id"].tolist() if tid not in entry_ids]
        parts = []
        if warn_count:
            parts.append(f"{warn_count} scan warning(s) -- see individual test folders")
        if orphan_ids:
            parts.append(f"{len(orphan_ids)} orphaned log row(s) with no matching test")
        if parts:
            self.warn_lbl.config(text=" | ".join(parts))

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
        if source_was_none:
            probed_picks = store.load_picks_for(entry)
            source = _resolve_load_source(entry, probed_picks)
        path = entry.data_path(source)
        if not self._load_common(path):
            return
        saved = None
        if not force_reset:
            # Reuse the source-resolution probe above rather than reading the same picks JSON
            # off disk twice -- only re-read when `source` was passed explicitly (no probe ran).
            saved = probed_picks if source_was_none else store.load_picks_for(entry)
            if saved is not None:
                self._apply_loaded_state(saved)
        self.state.active_source = source.lower()
        self.current_entry = entry
        self.root.title(f"DFIT interpretation — {entry.display_label}")
        entry.status = store.status_for(self.state if saved else None)
        self._refresh_queue_row(entry)
        self._update_folder_controls()

    def _update_folder_controls(self):
        """One sync point for the Source/Mark/Save & Next controls -- called from _load_test
        (after current_entry is set), the single-file _load wrapper (after clearing folder
        state), and _open_folder_path (which also covers _load_test's early return on a failed
        load, since that return happens before this call runs inside _load_test itself).

        Single-file mode (current_entry is None): all three cleared and disabled. Folder mode:
        cmb_source lists the entry's available sources (readonly only if there's more than
        one -- a single-source test has nothing to switch to), cmb_mark is readonly, and
        btn_save_next is enabled."""
        if self.current_entry is None:
            self.var_source.set("")
            self.cmb_source["values"] = []
            self.cmb_source.config(state="disabled")
            self.var_mark.set("")
            self.cmb_mark.config(state="disabled")
            self.btn_save_next.config(state="disabled")
            return
        entry = self.current_entry
        self.cmb_source["values"] = entry.available_sources
        self.var_source.set(self.state.active_source.upper())
        self.cmb_source.config(
            state="readonly" if len(entry.available_sources) > 1 else "disabled")
        self.cmb_mark.config(state="readonly")
        self.var_mark.set(self.state.explicit_status or "")
        self.btn_save_next.config(state="normal")

    def _on_source_change(self):
        """The Source combobox: switching CSV<->DBS resets all picks for this test (a fresh
        _load_test, not a resume), so confirm first -- reverting the combobox on decline."""
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

    def _on_mark_change(self):
        """The Mark combobox: an explicit done/skipped override on the current test's status.
        Persists only via Save & Next / navigation's picks-save (no log write here)."""
        self.state.explicit_status = self.var_mark.get() or None
        self.current_entry.status = store.status_for(self.state)
        self._refresh_queue_row(self.current_entry)

    def _write_log_row(self, entry: store.TestEntry):
        """Build and upsert one dfit_log.csv row for `entry` from the current state/res, then
        persist the whole log -- shared by _save_and_next and _finish's folder branch, the only
        two places dfit_log.csv is written. Callers wrap this in try/except (OneDrive file
        locks happen); the picks JSON save must already have succeeded independently before
        this runs."""
        row = store.build_log_row(entry, entry.data_path(self.state.active_source.upper()),
                                  self.folder_root, self.state, self.td, self.res)
        self.log_df = store.upsert_log_row(self.log_df, row)
        store.save_log(self.folder_root, self.log_df)

    def _save_and_next(self):
        """Bound to the Save & Next button -- only reachable in folder mode (the button is
        disabled otherwise), guarded anyway. Saves picks + the master log row, then advances to
        the next queue entry with status "new" (scanning circularly from just after the
        current one), or reports the queue is exhausted."""
        if self.current_entry is None or self.td is None:
            return
        entry = self.current_entry
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        self.state.explicit_status = self.var_mark.get() or None
        # Capture any unapplied entry-widget edits, then refresh so self.res (feeding the log
        # row below) is recomputed from the synced state rather than a stale prior compute.
        self._sync_state_from_widgets()
        self.refresh()
        store.save_picks_for(entry, self.state)
        entry.status = store.status_for(self.state)
        try:
            self._write_log_row(entry)
        except Exception as e:
            messagebox.showerror("Log write failed", str(e))
        self._refresh_queue_row(entry)
        statuses = [e.status for e in self.queue_entries]
        current_index = self.queue_entries.index(entry)
        next_index = _next_new_index(statuses, current_index)
        if next_index is None:
            messagebox.showinfo("Queue", "No new tests remain.")
            return
        self._load_test(self.queue_entries[next_index])

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
        self._save_current_queue_picks()
        self._load_test(entry)

    def _load_questionnaire(self, csv_path: str):
        """Auto-detect and parse a DFIT Questionnaire xlsx next to `csv_path`; prefill
        density/TVD/well name/formation.

        Best-effort only: a missing or malformed questionnaire must never block the CSV load
        already underway, so any failure here is swallowed and just leaves the provenance label
        empty. Density/TVD entries are prefilled even when the parse is uncertain (e.g. a coerced
        SG->ppg reading) -- the provenance label shows the raw source text so it can be checked.
        Well name/formation are plain free text, so there's no analogous "source" text to show.
        """
        self.quest_lbl.config(text="")
        try:
            xlsx_path, find_warnings = find_questionnaire(csv_path)
            if xlsx_path is None:
                return
            result = parse_questionnaire(xlsx_path)
        except Exception:
            return

        parts = []
        if result.density_ppg is not None:
            self.var_density.set(str(result.density_ppg))
            parts.append(f'density {result.density_ppg} ppg ["{result.density_source}"]')
        if result.tvd_ft is not None:
            self.var_tvd.set(str(result.tvd_ft))
            parts.append(f'TVD {result.tvd_ft} ft ["{result.tvd_source}"]')
        if result.well_name is not None:
            self.var_well.set(result.well_name)
            parts.append(f'well "{result.well_name}"')
        if result.formation is not None:
            self.var_formation.set(result.formation)
            parts.append(f'formation "{result.formation}"')
        all_warnings = find_warnings + result.warnings
        if all_warnings:
            parts.append("warnings: " + "; ".join(all_warnings))
        if parts:
            text = f"Questionnaire: {', '.join(parts)} — {os.path.basename(xlsx_path)}"
            self.quest_lbl.config(text=text)

    def _sync_state_from_widgets(self):
        self.state.pressure_col = self.var_pressure.get()
        self.state.rate_col = self.var_rate.get() or None
        self.state.volume_col = self.var_volume.get() or None
        self.state.pressure_is_bhp = self.var_isbhp.get()
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

    def _apply_config(self):
        if self.td is None:
            return
        self._sync_state_from_widgets()
        self.refresh()

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
            # picks.suggest_pp_axis); PC-D/PC-F leave the axis to the analyst.
            axis = picks.suggest_pp_axis(pcscen)
            if axis is not None:
                self.state.pp_axis = axis
            self.var_ppaxis.set(self.state.pp_axis)
            hint = _PC_HINTS.get(pcscen[:4]) or hint
        self._update_ppaxis_enabled()
        if pcscen_changed and self.step == "porepressure" and porepressure_skipped(self.state):
            # PC-F just got selected while sitting on the now-skipped pore-pressure step -- the
            # scenario combobox is visible on both loglog and porepressure, so this can happen
            # without ever leaving porepressure. _goto redirects to "loglog" and calls refresh()
            # itself; calling refresh() again here would just redo the same work.
            self._goto("loglog")
        else:
            self.refresh()
        if hint:
            # After refresh()/_goto(): _attach_controllers just set the step's default hint
            # text, and the scenario feedback must win.
            self.hint_lbl.config(text=hint)

    def _on_showd2(self):
        self.state.show_d2pdg2 = self.var_showd2.get()
        self.refresh()

    def _on_reset_gfunction_picks(self):
        """The G-function step's adaptive "Reset picks" button: re-run the active scenario's
        auto-pick, discarding any manual drags (decision 3)."""
        hint = picks.reset_gfunction_picks(self.state, compute_all(self.state, self.td))
        self.refresh()
        if hint:
            # refresh() -> _attach_controllers just set the step's default hint text; the
            # reset's own failure feedback must win, same pattern as _on_scenario above.
            self.hint_lbl.config(text=hint)

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
        win.title("DFIT interpretation guide")
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
                              foreground="red").pack(anchor="w", padx=10, pady=(0, 2))
                ttk.Label(inner, text=fig.caption, wraplength=900, justify="left",
                          font=("", 8, "italic"), foreground="gray").pack(
                    anchor="w", padx=10, pady=(0, 10))

        ttk.Label(inner, text=guide.source, wraplength=900, justify="left",
                  foreground="gray").pack(anchor="w", padx=10, pady=(6, 10))

    def _load_guide_image(self, name: str):
        try:
            return tk.PhotoImage(file=str(_GUIDE_ASSETS / name))
        except Exception:
            return None

    def _reconcile_pp_axis(self):
        """Force pp_axis to the value a postclosure scenario dictates (if any), so a locked
        axis can't disagree with its scenario. pp_axis feeds compute_all, so refresh() calls
        this before recomputing; PC-D/PC-F/unset return None and leave a manual choice intact."""
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

        A "porepressure" destination redirects to "loglog" whenever PC-F skips the pore-pressure
        step -- this one place covers the log-log Skip button, resume-on-load
        (first_not_visited_step), and any other programmatic jump."""
        if self.td is None:
            return
        if step == "porepressure" and porepressure_skipped(self.state):
            step = "loglog"
        if self.state.step_status.get(step, "not_visited") == "not_visited":
            self._seed_step(step)
            self.state.step_status[step] = "visited"
        self.step = step
        self.refresh()

    def _seed_step(self, key: str) -> None:
        """Pre-populate reasonable default picks for ``key`` on its first visit, via
        ``picks.SEEDERS``. "overview" and "isip" need ``self.td`` too; the rest take only
        (state, res)."""
        if self.td is None:
            return
        res = compute_all(self.state, self.td)
        seeder = picks.SEEDERS[key]
        if key == "overview":
            seeder(self.state, self.td)
        elif key == "isip":
            seeder(self.state, self.td, res)
        else:
            seeder(self.state, res)

    def _next(self):
        """Mark the current step done and advance. next_step() clamps at the last step, so at
        "porepressure" this simply re-marks it done and re-refreshes -- a no-op in terms of
        navigation, per the brief."""
        if self.td is None:
            return
        self.state.step_status[self.step] = "done"
        self._goto(next_step(self.step))

    def _skip(self):
        """Mark the current step skipped and advance, same clamping behavior as _next()."""
        if self.td is None:
            return
        self.state.step_status[self.step] = "skipped"
        self._goto(next_step(self.step))

    def _last_step(self) -> str:
        """The effective last step of the workflow: "loglog" when the postclosure scenario is
        PC-F (no peak, so there is no postclosure line and the pore-pressure step is skipped),
        else the actual last entry in STEPS ("porepressure")."""
        if porepressure_skipped(self.state):
            return "loglog"
        return STEPS[-1][0]

    def _advance(self):
        """Bound to the Next/Finish stepbar button. On the effective last step (_last_step(),
        normally "porepressure" but "loglog" when PC-F skips pore pressure) the button reads
        "Finish" and exports (_finish). Otherwise it advances (_next) -- but only once the
        current step's required scenario pick is present; step_gate_error gates the forward
        jump and the inline gate_lbl says what is missing. Back/Skip/breadcrumb navigation are
        NOT gated."""
        if self.td is None:
            return
        if self.step == self._last_step():
            self._finish()
            return
        msg = step_gate_error(self.state, self.step)
        if msg:
            self.gate_lbl.config(text=msg)
            return
        self._next()

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

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        defaults = plots.RENDERERS[self.step](self.ax, self.td, self.state, self.res)

        full_x = self.ax.get_xlim()
        full_y = self.ax.get_ylim()
        if self.step == "gfunction" and defaults.ylim is not None:
            # The Axes' own autoscale over full_y also picks up the effective-ISIP tangent's
            # dashed extension (drawn on this same Axes), which can swing the pressure axis to
            # extreme psi values far outside the real BHP data. The renderer's own data-driven
            # ylim is the true outer bound for this step's pressure axis.
            full_y = defaults.ylim
        twin = self._twin_axes()
        full_y2 = twin.get_ylim() if twin is not None else None
        if self.step == "gfunction" and full_y2 is not None:
            # Hard-clamp the derivative (dP/dG) slider's full range to 0-500 regardless of how
            # extreme the raw dPdG spike is, so the slider itself can never travel past it.
            full_y2 = (max(full_y2[0], 0.0), min(full_y2[1], 500.0))

        view = _resolve_view(self._views.get(self.step), defaults, full_x, full_y, full_y2)
        self._views[self.step] = view
        self.ax.set_xlim(view.xlim)
        self.ax.set_ylim(view.ylim)
        if twin is not None and view.y2lim is not None:
            twin.set_ylim(view.y2lim)
        # The d2P/dG2 axis gets no slider and no persisted view (decision D3) -- apply the
        # renderer's fresh default every refresh instead of folding it into ``view``/``_views``.
        d2_axes = self._d2_axes()
        if d2_axes is not None and defaults.y3lim is not None:
            d2_axes.set_ylim(defaults.y3lim)

        self._build_sliders(full_x, full_y, full_y2, view, twin)
        # tight_layout would fight the manually placed slider axes reserved on the right margin;
        # the d2 axis's offset third spine needs a wider right margin than usual when it's on.
        right = 0.70 if (self.step == "gfunction" and self.state.show_d2pdg2) else 0.84
        self.fig.subplots_adjust(left=0.10, right=right, bottom=0.16, top=0.90)
        self._attach_controllers()
        self.canvas.draw_idle()
        self._update_stepbar()
        self._update_panel_visibility()
        self._update_panel()

    def _update_stepbar(self):
        """Disable breadcrumb buttons for steps still ``not_visited`` (so a click is only ever
        honored for a reached step) and highlight the current step. Bold text rather than an
        Accent.TButton style -- that style name is theme-specific and not guaranteed to exist.

        The "porepressure" breadcrumb is force-disabled whenever PC-F skips that step, even if
        it was visited earlier in the session (e.g. the analyst picked PC-F after already
        reaching pore pressure) -- _goto redirects that destination to "loglog" regardless, so
        the button must not look reachable."""
        style = ttk.Style()
        style.configure("StepCurrent.TButton", font=("TkDefaultFont", 9, "bold"))
        skip_pp = porepressure_skipped(self.state)
        for key, btn in self.step_buttons.items():
            status = self.state.step_status.get(key, "not_visited")
            reachable = status != "not_visited" and not (key == "porepressure" and skip_pp)
            btn.state(["!disabled"] if reachable else ["disabled"])
            btn.configure(style="StepCurrent.TButton" if key == self.step else "TButton")
        # On the effective last step the Next button becomes Finish (bold, like the current-step
        # breadcrumb) -- _advance dispatches to _finish() instead of _next() in that case.
        if self.step == self._last_step():
            self.next_btn.configure(text="Finish", style="StepCurrent.TButton")
        else:
            self.next_btn.configure(text="Next >", style="TButton")

    def _update_panel_visibility(self):
        """Show the closure-scenario widgets only on "gfunction" and the postclosure/pp-axis
        widgets only on "loglog"/"porepressure" -- both packed relative to sep_before_notes so
        re-showing never reorders the panel."""
        self.frm_cscen.pack_forget()
        self.frm_pcscen.pack_forget()
        if self.step == "gfunction":
            self.frm_cscen.pack(fill="x", before=self.sep_before_notes)
            self.btn_gfunction_reset.configure(
                text=picks.gfunction_reset_button_label(self.state.closure_scenario))
        if self.step in ("loglog", "porepressure"):
            self.frm_pcscen.pack(fill="x", before=self.sep_before_notes)
            # refresh() already reconciled pp_axis with the scenario before recomputing; here
            # just lock/unlock the radios to match.
            self._update_ppaxis_enabled()

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

    def _build_sliders(self, full_x, full_y, full_y2, view, twin):
        """Per-axis RangeSlider zoom controls: one under the plot for x, one on the right edge
        for y, and (only when a twin exists) one further right for y2.

        Called from refresh() after the Axes are (re)built and ``view`` applied, so slider
        ranges/initial values always reflect the just-applied ViewState. The on_changed
        callbacks only set limits on the target Axes, mutate ``view`` in place, and draw_idle --
        never refresh() (fig.clf() would destroy the slider mid-drag).
        """
        self._x_slider = self._make_range_slider(
            rect=[0.10, 0.04, 0.68, 0.03], orientation="horizontal",
            full_range=full_x, cur_range=view.xlim, scale=self.ax.get_xscale(),
            apply=lambda lo, hi: self.ax.set_xlim(lo, hi),
            store=lambda lo, hi: setattr(view, "xlim", (lo, hi)),
        )
        self._y_slider = self._make_range_slider(
            rect=[0.87, 0.16, 0.02, 0.74], orientation="vertical",
            full_range=full_y, cur_range=view.ylim, scale=self.ax.get_yscale(),
            apply=lambda lo, hi: self.ax.set_ylim(lo, hi),
            store=lambda lo, hi: setattr(view, "ylim", (lo, hi)),
        )
        self._y2_slider = None
        if twin is not None:  # refresh() only sets full_y2 when there is a twin to read it from
            self._y2_slider = self._make_range_slider(
                rect=[0.93, 0.16, 0.02, 0.74], orientation="vertical",
                full_range=full_y2, cur_range=view.y2lim if view.y2lim is not None else full_y2,
                scale=twin.get_yscale(),
                apply=lambda lo, hi: twin.set_ylim(lo, hi),
                store=lambda lo, hi: setattr(view, "y2lim", (lo, hi)),
            )

    def _make_range_slider(self, rect, orientation, full_range, cur_range, scale, apply, store):
        """Build one PanRangeSlider, or return None for a degenerate/non-finite extent (a flat
        or single-sample axis has nothing to zoom).

        ``scale`` is the target Axes' actual xscale/yscale ("log" or "linear") -- for a log axis
        the slider itself operates in log10 space (clamped to a 1e-12 floor) with a valfmt that
        displays the linear value, and the callback exponentiates before applying limits.
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
        slider = sliders.PanRangeSlider(ax, "", lo_full, hi_full, valinit=(lo_cur, hi_cur),
                                        orientation=orientation, valfmt=valfmt)

        def _on_changed(val):
            lo, hi = val
            if is_log:
                lo, hi = sliders.from_log_bounds(lo, hi)
            apply(lo, hi)
            store(lo, hi)
            self.canvas.draw_idle()

        slider.on_changed(_on_changed)
        return slider

    def _reset_view(self):
        self._views[self.step] = None
        self.refresh()

    def _attach_controllers(self):
        for c in self._controllers:
            c.disconnect()
        self._controllers = []

        step = self.step
        if step == "overview":
            def _commit(idx_attr):
                def on_release(x_hours):
                    idx = picks._nearest(self.td.t_s / 3600.0, x_hours)
                    setattr(self.state, idx_attr, idx)
                    self.state.qmax_bpm = None  # re-derive from the new window
                    self.refresh()
                return on_release
            drag_ctrl = picks.DragLineController(
                self.canvas, self.ax,
                handlers={"start": _commit("start_idx"),
                          "shutin": _commit("shutin_idx")})
            self._controllers.append(drag_ctrl)
            self._controllers.append(picks.HoverCursorController(self.canvas, [drag_ctrl]))
            self.hint_lbl.config(
                text="Drag the injection-start and shut-in lines to adjust the window.")
        elif step == "isip":
            res = self.res
            step_ctrls = []
            if res.bhp_all is not None and res.t_shutin_s is not None:
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
                          "extension": "isip_tangent_extension"},
                    get_pick=get_pick, commit_fn=commit, curve=(t_min, res.bhp_all),
                    anchor_half=30, readout_fn=readout, gate=gate))
            self._controllers.extend(step_ctrls)
            if step_ctrls:
                self._controllers.append(picks.HoverCursorController(self.canvas, step_ctrls))
            self.hint_lbl.config(
                text="Drag the anchor along the curve, the body to pan, or an end to rotate "
                     "the ISIP tangent.")
        elif step == "gfunction":
            res = self.res
            ax2 = self._twin_axes()
            step_ctrls = []
            scenario = self.state.closure_scenario
            if res.diagnostics is not None and res.resampled is not None and ax2 is not None:
                G, p, dPdG = res.diagnostics.G, res.resampled.p, res.diagnostics.dPdG
                gate = picks._CaptureGate()

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

                # min-dP/dG-first ordering preserved (tests unpack step_ctrls by position) --
                # the triangle only applies to C-A (rel-min anchor) / C-B (inflection seed); the
                # contact marker applies to every scenario except C-C/C-D, which have no contact
                # rule at all (decision 4 / the CLAUDE.md closure-scenario table).
                if scenario.startswith(("C-A", "C-B")):
                    step_ctrls.append(picks.DraggablePointController(
                        self.canvas, ax2, "min_dpdg_point", G, dPdG, commit_fn=commit_min_dpdg,
                        gate=gate))
                if not scenario.startswith(("C-C", "C-D")):
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
                text="Rotate the through-origin line; drag the closure marker or its vertical line.")
        elif step == "loglog":
            def on_span(lo, hi):
                picks.handle_loglog_span(self.state, lo, hi)
                self.refresh()
            self._controllers.append(picks.SpanController(self.ax, on_span))
            self.hint_lbl.config(text="Drag to select the late-time window; set postclosure scenario.")
        elif step == "porepressure":
            def on_span(lo, hi):
                picks.handle_pp_span(self.state, lo, hi)
                self.refresh()
            self._controllers.append(picks.SpanController(self.ax, on_span))
            self.hint_lbl.config(text="Drag to select the late-time window; choose the axis.")

    def _update_panel(self):
        r = self.res
        def s(v, f="{:.0f}"):
            return f.format(v) if v is not None else "-"
        vals = {
            "te (min)": s(r.te_s / 60 if r.te_s else None, "{:.2f}"),
            "Vinj (bbl)": s(r.vinj, "{:.1f}"),
            "qmax (bpm)": s(r.qmax_bpm, "{:.2f}"),
            "apparent ISIP": s(r.apparent_isip),
            "eff ISIP (compliance)": s(r.effective_isip_compliance),
            "NWB complexity": s(r.near_wellbore_complexity),
            "contact P": s(r.contact_pressure),
            "Shmin compliance": s(r.shmin_compliance),
            "Shmin tangent": s(r.shmin_tangent),
            "Shmin variable": s(r.shmin_variable),
            "Shmin rapid": (interpret.format_shmin_rapid(r.shmin_rapid)
                            if r.shmin_rapid is not None else "-"),
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
        }
        for k, v in vals.items():
            owning_step = FIELD_STEP[k]
            visited = self.state.step_status.get(owning_step, "not_visited") != "not_visited"
            self.value_lbls[k].config(text=v if visited else "-")
        self.warn_lbl.config(text=" | ".join(r.warnings[:2]) if r.warnings else "")

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

        Single-file mode (current_entry is None): byte-for-byte today's behavior -- re-save the
        picks JSON next to the data file and write a PNG per step (current zoom) into a sibling
        subfolder; no log. Folder mode: save picks via store.save_picks_for (the store path is
        the single ground truth in folder mode -- no <stem>_picks.json duplicate), same PNG
        export, then the same dfit_log.csv row write as _save_and_next -- but no auto-advance.

        No success popup; only a failure raises a dialog."""
        if self.td is None:
            return
        self.state.step_status[self.step] = "done"
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

    def _apply_loaded_state(self, state: PickState):
        """Adopt `state` as the current PickState and reflect it into every widget -- shared by
        single-file _load_picks (state fresh off a file dialog) and folder-mode _load_test
        (state fresh off store.load_picks_for)."""
        self.state = state
        self._views = {k: None for k, _ in STEPS}
        if not self.state.step_status:
            # An old save has real picks but no breadcrumb history -- infer it so the
            # breadcrumb doesn't present the whole workflow as unreached.
            self.state.step_status = infer_step_status(self.state)
        # reflect into widgets
        self.var_pressure.set(self.state.pressure_col)
        self.var_rate.set(self.state.rate_col or "")
        self.var_volume.set(self.state.volume_col or "")
        self.var_isbhp.set(self.state.pressure_is_bhp)
        self.var_density.set("" if self.state.density_ppg is None else str(self.state.density_ppg))
        self.var_tvd.set("" if self.state.tvd_ft is None else str(self.state.tvd_ft))
        self.var_well.set(self.state.well_name)
        self.var_formation.set(self.state.formation)
        # Density/TVD above now came from the picks file, not the questionnaire that was auto-
        # detected (if any) when the CSV was loaded -- clear the stale provenance label so it
        # doesn't misattribute these values.
        self.quest_lbl.config(text="")
        self.var_alpha.set(str(self.state.alpha))
        self.var_step.set(str(self.state.resample_step))
        self.var_cscen.set(self.state.closure_scenario)
        self.var_pcscen.set(self.state.postclosure_scenario)
        self.var_ppaxis.set(self.state.pp_axis)
        self.var_showd2.set(self.state.show_d2pdg2)
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
        self._apply_loaded_state(PickState.from_json(path))


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
