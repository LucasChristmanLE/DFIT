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
from dataclasses import dataclass
from typing import Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from . import io_load, picks, plots, sliders
from .model import PickState, TangentPick, compute_all, infer_step_status
from .plots import ViewDefaults

_SLIDER_GID = "slider"

STEPS = [
    ("overview", "Overview"),
    ("isip", "Literal ISIP"),
    ("gfunction", "G-function"),
    ("tangent", "Tangent"),
    ("loglog", "Log-log"),
    ("porepressure", "Pore pressure"),
]
CLOSURE_SCENARIOS = ["", "C-A clear", "C-B adequate", "C-C no-contact", "C-D rapid"]
POSTCLOSURE_SCENARIOS = ["", "PC-A linear", "PC-B false-radial", "PC-C mixed",
                         "PC-D mixed", "PC-E none", "PC-F none"]

# The 13 result-panel rows, in display order -- module level (not just a literal inside
# _build_body) so FIELD_STEP below and tests can both refer to the same list.
PANEL_FIELDS = [
    "te (min)", "Vinj (bbl)", "qmax (bpm)", "literal ISIP", "effective ISIP",
    "contact P", "Shmin compliance", "Shmin tangent", "closure P",
    "net (compliance)", "net (tangent)", "delta closure", "pore pressure",
]

# Which step "owns" each panel field -- _update_panel shows "-" for a field whose step is still
# not_visited, even if compute_all already produced a value for it (e.g. picks.seed_defaults ran
# ahead of the user actually visiting that step).
FIELD_STEP = {
    "te (min)": "overview",
    "Vinj (bbl)": "overview",
    "qmax (bpm)": "overview",
    "literal ISIP": "isip",
    "effective ISIP": "gfunction",
    "contact P": "gfunction",
    "Shmin compliance": "gfunction",
    "net (compliance)": "gfunction",
    "Shmin tangent": "tangent",
    "closure P": "tangent",
    "net (tangent)": "tangent",
    "delta closure": "tangent",
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
    """Convert the stored literal-ISIP tangent -- ``anchor_x`` in seconds-since-file-start
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
    def __init__(self, root: tk.Tk, csv_path: str | None = None):
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

        self._build_top()
        self._build_body()
        self._build_stepbar()

        if csv_path:
            self._load(csv_path)

    # ---- layout ---------------------------------------------------------------------------------
    def _build_top(self):
        top = ttk.Frame(self.root, padding=6)
        top.pack(side="top", fill="x")
        ttk.Button(top, text="Open CSV…", command=self._open).pack(side="left")
        self.file_lbl = ttk.Label(top, text="(no file)")
        self.file_lbl.pack(side="left", padx=8)

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

        def combo(parent, label, var, width=18):
            ttk.Label(parent, text=label).pack(side="left", padx=(8, 2))
            c = ttk.Combobox(parent, textvariable=var, width=width, state="readonly")
            c.pack(side="left")
            return c

        self.cmb_pressure = combo(cfg, "Pressure:", self.var_pressure)
        ttk.Checkbutton(cfg, text="is BHP", variable=self.var_isbhp,
                        command=self._apply_config).pack(side="left", padx=4)
        self.cmb_rate = combo(cfg, "Rate:", self.var_rate, 14)
        self.cmb_volume = combo(cfg, "Volume:", self.var_volume, 12)

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

    def _build_body(self):
        body = ttk.Frame(self.root)
        body.pack(side="top", fill="both", expand=True)

        # center: canvas
        center = ttk.Frame(body)
        center.pack(side="left", fill="both", expand=True)
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

        self.frm_pcscen = ttk.Frame(panel)
        ttk.Label(self.frm_pcscen, text="Postclosure scenario").pack(anchor="w")
        self.var_pcscen = tk.StringVar(value="")
        self.cmb_pcscen = ttk.Combobox(self.frm_pcscen, textvariable=self.var_pcscen,
                                       values=POSTCLOSURE_SCENARIOS, state="readonly")
        self.cmb_pcscen.pack(fill="x")
        self.cmb_pcscen.bind("<<ComboboxSelected>>", lambda e: self._on_scenario())

        ttk.Label(self.frm_pcscen, text="Pore-pressure axis").pack(anchor="w", pady=(6, 0))
        self.var_ppaxis = tk.StringVar(value="tm12")
        for txt, val in [("t^(-1/2)", "tm12"), ("t^(-1)", "tm1")]:
            ttk.Radiobutton(self.frm_pcscen, text=txt, variable=self.var_ppaxis, value=val,
                            command=self._on_scenario).pack(anchor="w")

        self.sep_before_notes = ttk.Separator(panel)
        self.sep_before_notes.pack(fill="x", pady=6)
        ttk.Label(panel, text="Notes").pack(anchor="w")
        self.txt_notes = tk.Text(panel, height=5, width=36)
        self.txt_notes.pack(fill="x")
        self.hint_lbl = ttk.Label(panel, text="", wraplength=300, foreground="gray")
        self.hint_lbl.pack(anchor="w", pady=(6, 0))

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
        ttk.Button(bar, text="Next >", command=self._next).pack(side="left", padx=2)
        ttk.Button(bar, text="Skip >", command=self._skip).pack(side="left", padx=2)

    # ---- data / config --------------------------------------------------------------------------
    def _open(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self._load(path)

    def _load(self, path: str):
        try:
            self.td = io_load.load_csv(path)
        except Exception as e:
            messagebox.showerror("Load failed", str(e))
            return
        self._views = {}
        self.file_lbl.config(text=os.path.basename(path))
        cols = self.td.columns
        for cmb in (self.cmb_pressure, self.cmb_rate, self.cmb_volume):
            cmb["values"] = [""] + cols
        g = io_load.suggest_channels(cols)
        self.var_pressure.set(g["pressure"] or "")
        self.var_rate.set(g["rate"] or "")
        self.var_volume.set(g["volume"] or "")
        self.var_isbhp.set(bool(g["pressure_is_bhp"]))
        self._sync_state_from_widgets()
        picks.seed_defaults(self.state, self.td, lambda s: compute_all(s, self.td))
        self._goto("overview")

    def _sync_state_from_widgets(self):
        self.state.pressure_col = self.var_pressure.get()
        self.state.rate_col = self.var_rate.get() or None
        self.state.volume_col = self.var_volume.get() or None
        self.state.pressure_is_bhp = self.var_isbhp.get()
        self.state.density_ppg = _to_float(self.var_density.get())
        self.state.tvd_ft = _to_float(self.var_tvd.get())
        self.state.alpha = _to_float(self.var_alpha.get()) or 1.0
        self.state.resample_step = _to_float(self.var_step.get()) or 30.0

    def _apply_config(self):
        if self.td is None:
            return
        self._sync_state_from_widgets()
        self.refresh()

    def _on_scenario(self):
        self.state.closure_scenario = self.var_cscen.get()
        self.state.postclosure_scenario = self.var_pcscen.get()
        self.state.pp_axis = self.var_ppaxis.get()
        self.refresh()

    # ---- steps / render -------------------------------------------------------------------------
    def _goto(self, step: str):
        """Navigate to ``step``. Breadcrumb buttons for a ``not_visited`` step are disabled by
        _update_stepbar, so reaching one here means either it was already reached, or this is
        the programmatic first jump onto a step (initial load, or Next/Skip/Back stepping one
        further than the user has been). First-visit seeding lives here, not in Next/Skip/Back,
        so the seed always runs regardless of which control got the user there."""
        if self.td is None:
            return
        if self.state.step_status.get(step, "not_visited") == "not_visited":
            self._seed_step(step)
            self.state.step_status[step] = "visited"
        self.step = step
        self.refresh()

    def _seed_step(self, key: str) -> None:
        """No-op stub. Filled in by the seed-on-entry task (Task 6): pre-populate reasonable
        default picks for ``key`` the first time it is visited."""
        pass

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

    def _back(self):
        """Go to the previous step. No status change -- prev_step() clamps at the first step."""
        if self.td is None:
            return
        self._goto(prev_step(self.step))

    def refresh(self):
        if self.td is None:
            return
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        self.res = compute_all(self.state, self.td)

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        defaults = plots.RENDERERS[self.step](self.ax, self.td, self.state, self.res)

        full_x = self.ax.get_xlim()
        full_y = self.ax.get_ylim()
        twin = self._twin_axes()
        full_y2 = twin.get_ylim() if twin is not None else None

        view = _resolve_view(self._views.get(self.step), defaults, full_x, full_y, full_y2)
        self._views[self.step] = view
        self.ax.set_xlim(view.xlim)
        self.ax.set_ylim(view.ylim)
        if twin is not None and view.y2lim is not None:
            twin.set_ylim(view.y2lim)

        self._build_sliders(full_x, full_y, full_y2, view, twin)
        # tight_layout would fight the manually placed slider axes reserved on the right margin.
        self.fig.subplots_adjust(left=0.10, right=0.84, bottom=0.16, top=0.90)
        self._attach_controllers()
        self.canvas.draw_idle()
        self._update_stepbar()
        self._update_panel_visibility()
        self._update_panel()

    def _update_stepbar(self):
        """Disable breadcrumb buttons for steps still ``not_visited`` (so a click is only ever
        honored for a reached step) and highlight the current step. Bold text rather than an
        Accent.TButton style -- that style name is theme-specific and not guaranteed to exist."""
        style = ttk.Style()
        style.configure("StepCurrent.TButton", font=("TkDefaultFont", 9, "bold"))
        for key, btn in self.step_buttons.items():
            status = self.state.step_status.get(key, "not_visited")
            btn.state(["!disabled"] if status != "not_visited" else ["disabled"])
            btn.configure(style="StepCurrent.TButton" if key == self.step else "TButton")

    def _update_panel_visibility(self):
        """Show the closure-scenario widgets only on "gfunction" and the postclosure/pp-axis
        widgets only on "loglog"/"porepressure" -- both packed relative to sep_before_notes so
        re-showing never reorders the panel."""
        self.frm_cscen.pack_forget()
        self.frm_pcscen.pack_forget()
        if self.step == "gfunction":
            self.frm_cscen.pack(fill="x", before=self.sep_before_notes)
        if self.step in ("loglog", "porepressure"):
            self.frm_pcscen.pack(fill="x", before=self.sep_before_notes)

    def _twin_axes(self):
        """The step's twin (secondary y) Axes if it has one, else None.

        Excludes the slider Axes _build_sliders adds to the right margin -- those are tagged
        with gid ``_SLIDER_GID`` precisely so this scan doesn't mistake one of them for the
        step's twin.
        """
        for a in self.fig.axes:
            if a is not self.ax and a.get_gid() != _SLIDER_GID:
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
            self._controllers.append(picks.DragLineController(
                self.canvas, self.ax,
                handlers={"start": _commit("start_idx"),
                          "shutin": _commit("shutin_idx")}))
            self.hint_lbl.config(
                text="Drag the injection-start and shut-in lines to adjust the window.")
        elif step == "isip":
            res = self.res
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

                self._controllers.append(picks.AnchorLineController(
                    self.canvas, self.ax,
                    gids={"segment": "isip_tangent_segment", "tick": "isip_tangent_tick",
                          "extension": "isip_tangent_extension"},
                    get_pick=get_pick, commit_fn=commit, curve=(t_min, res.bhp_all),
                    anchor_half=30, readout_fn=readout, gate=gate))
            self.hint_lbl.config(
                text="Drag the anchor along the curve, the body to pan, or an end to rotate "
                     "the ISIP tangent.")
        elif step == "gfunction":
            res = self.res
            if res.diagnostics is not None and res.resampled is not None:
                G, p = res.diagnostics.G, res.resampled.p
                gate = picks._CaptureGate()

                def commit_line(kind, anchor_x, anchor_y, slope):
                    picks.commit_eff_isip_line(self.state, res, kind, anchor_x, anchor_y, slope)
                    self.refresh()

                def commit_point(x):
                    picks.commit_contact_point(self.state, x)
                    self.refresh()

                self._controllers.append(picks.AnchorLineController(
                    self.canvas, self.ax,
                    gids={"segment": "eff_isip_segment", "tick": "eff_isip_tick",
                          "extension": "eff_isip_extension"},
                    get_pick=lambda: self.state.eff_isip_line, commit_fn=commit_line,
                    curve=(G, p), gate=gate))
                self._controllers.append(picks.DraggablePointController(
                    self.canvas, self.ax, "contact_point", G, p, commit_fn=commit_point,
                    gate=gate))
            self.hint_lbl.config(text="Drag the effective-ISIP line or the contact marker.")
        elif step == "tangent":
            res = self.res
            ax2 = self._twin_axes()
            if res.diagnostics is not None and ax2 is not None:
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

                self._controllers.append(picks.AnchorLineController(
                    self.canvas, ax2, gids={"segment": "closure_line_segment"},
                    get_pick=get_closure_pick, commit_fn=commit_line, curve=None,
                    allow_anchor=False, allow_body=False, gate=gate))
                self._controllers.append(picks.DraggablePointController(
                    self.canvas, ax2, "closure_point", dg.G, dg.GdPdG, commit_fn=commit_point,
                    gate=gate))
            self.hint_lbl.config(text="Rotate the through-origin line; drag the closure marker.")
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
            "literal ISIP": s(r.literal_isip),
            "effective ISIP": s(r.effective_isip),
            "contact P": s(r.contact_pressure),
            "Shmin compliance": s(r.shmin_compliance),
            "Shmin tangent": s(r.shmin_tangent),
            "closure P": s(r.closure_pressure),
            "net (compliance)": s(r.net_pressure_compliance),
            "net (tangent)": s(r.net_pressure_tangent),
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
            self.state.to_json(path)

    def _load_picks(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        self.state = PickState.from_json(path)
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
        self.var_alpha.set(str(self.state.alpha))
        self.var_step.set(str(self.state.resample_step))
        self.var_cscen.set(self.state.closure_scenario)
        self.var_pcscen.set(self.state.postclosure_scenario)
        self.var_ppaxis.set(self.state.pp_axis)
        self.txt_notes.delete("1.0", "end")
        self.txt_notes.insert("1.0", self.state.notes)
        # Resume at the first not-yet-visited step so the breadcrumb picks up where the saved
        # workflow left off; if every step already has some status, there is no natural resume
        # point, so land on "overview" (first_not_visited_step's fallback).
        self._goto(first_not_visited_step(self.state.step_status))


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
