"""Tkinter/ttk shell: file open, channel mapping, the six-step canvas, live value panel.

Hosts the matplotlib canvas and wires the per-step pickers from picks.py to a recompute+redraw
loop. Holds no interpretation logic itself -- every number comes from model.compute_all.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import io_load, picks, plots
from .model import PickState, compute_all

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
        self.toolbar = NavigationToolbar2Tk(self.canvas, center)
        self.toolbar.update()

        # right: pick panel
        panel = ttk.Frame(body, padding=8, width=320)
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)
        ttk.Label(panel, text="Results", font=("", 10, "bold")).pack(anchor="w")
        self.value_lbls: dict[str, ttk.Label] = {}
        for key in ["te (min)", "Vinj (bbl)", "qmax (bpm)", "literal ISIP", "effective ISIP",
                    "contact P", "Shmin compliance", "Shmin tangent", "closure P",
                    "net (compliance)", "net (tangent)", "delta closure", "pore pressure"]:
            row = ttk.Frame(panel); row.pack(fill="x")
            ttk.Label(row, text=key, width=16).pack(side="left")
            v = ttk.Label(row, text="-", width=14, anchor="e"); v.pack(side="right")
            self.value_lbls[key] = v

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="Closure scenario").pack(anchor="w")
        self.var_cscen = tk.StringVar(value="")
        self.cmb_cscen = ttk.Combobox(panel, textvariable=self.var_cscen,
                                      values=CLOSURE_SCENARIOS, state="readonly")
        self.cmb_cscen.pack(fill="x")
        self.cmb_cscen.bind("<<ComboboxSelected>>", lambda e: self._on_scenario())

        ttk.Label(panel, text="Postclosure scenario").pack(anchor="w", pady=(6, 0))
        self.var_pcscen = tk.StringVar(value="")
        self.cmb_pcscen = ttk.Combobox(panel, textvariable=self.var_pcscen,
                                       values=POSTCLOSURE_SCENARIOS, state="readonly")
        self.cmb_pcscen.pack(fill="x")
        self.cmb_pcscen.bind("<<ComboboxSelected>>", lambda e: self._on_scenario())

        ttk.Label(panel, text="Pore-pressure axis").pack(anchor="w", pady=(6, 0))
        self.var_ppaxis = tk.StringVar(value="tm12")
        for txt, val in [("t^(-1/2)", "tm12"), ("t^(-1)", "tm1")]:
            ttk.Radiobutton(panel, text=txt, variable=self.var_ppaxis, value=val,
                            command=self._on_scenario).pack(anchor="w")

        ttk.Separator(panel).pack(fill="x", pady=6)
        ttk.Label(panel, text="Notes").pack(anchor="w")
        self.txt_notes = tk.Text(panel, height=5, width=36)
        self.txt_notes.pack(fill="x")
        self.hint_lbl = ttk.Label(panel, text="", wraplength=300, foreground="gray")
        self.hint_lbl.pack(anchor="w", pady=(6, 0))

    def _build_stepbar(self):
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(side="bottom", fill="x")
        self.warn_lbl = ttk.Label(bar, text="", foreground="red")
        self.warn_lbl.pack(side="right")
        for key, label in STEPS:
            ttk.Button(bar, text=label, command=lambda k=key: self._goto(k)).pack(side="left", padx=2)

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
        self.step = step
        self.refresh()

    def refresh(self, preserve_view: bool = False):
        if self.td is None:
            return
        prev = None
        if preserve_view and self.ax is not None:
            prev = (self.ax.get_xlim(), self.ax.get_ylim())
        self.state.notes = self.txt_notes.get("1.0", "end").strip()
        self.res = compute_all(self.state, self.td)

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        plots.RENDERERS[self.step](self.ax, self.td, self.state, self.res)
        if prev is not None:
            self.ax.set_xlim(prev[0]); self.ax.set_ylim(prev[1])
        self.fig.tight_layout()
        self._attach_controllers()
        self.canvas.draw_idle()
        self._update_panel()

    def _attach_controllers(self):
        for c in self._controllers:
            c.disconnect()
        self._controllers = []

        def guarded(fn):
            def wrapped(*a):
                if self.toolbar.mode:  # zoom/pan active -> don't place picks
                    return
                fn(*a)
                self.refresh()
            return wrapped

        step = self.step
        if step == "overview":
            def _commit(idx_attr):
                def on_release(x_hours):
                    idx = picks._nearest(self.td.t_s / 3600.0, x_hours)
                    setattr(self.state, idx_attr, idx)
                    self.state.qmax_bpm = None  # re-derive from the new window
                    self.refresh(preserve_view=True)
                return on_release
            self._controllers.append(picks.DragLineController(
                self.canvas, self.ax,
                handlers={"start": _commit("start_idx"),
                          "shutin": _commit("shutin_idx")},
                guard=lambda: bool(self.toolbar.mode)))
            self.hint_lbl.config(
                text="Drag the injection-start and shut-in lines to adjust the window.")
        elif step == "isip":
            self._controllers.append(picks.ClickController(
                self.canvas, self.ax,
                guarded(lambda x, y, b: picks.handle_isip_click(self.state, self.td, self.res, x, b))))
            self.hint_lbl.config(text="Click on the decline to place the ISIP tangent anchor.")
        elif step == "gfunction":
            self._controllers.append(picks.ClickController(
                self.canvas, self.ax,
                guarded(lambda x, y, b: picks.handle_gfunction_click(self.state, self.res, x, b))))
            self.hint_lbl.config(text="Left-click = effective-ISIP line, right-click = contact.")
        elif step == "tangent":
            self._controllers.append(picks.ClickController(
                self.canvas, self.ax,
                guarded(lambda x, y, b: picks.handle_tangent_click(self.state, self.res, x, b))))
            self.hint_lbl.config(text="Click the departure from the through-origin line = closure.")
        elif step == "loglog":
            self._controllers.append(picks.SpanController(
                self.ax, guarded_span(self, picks.handle_loglog_span)))
            self.hint_lbl.config(text="Drag to select the late-time window; set postclosure scenario.")
        elif step == "porepressure":
            self._controllers.append(picks.SpanController(
                self.ax, guarded_span(self, picks.handle_pp_span)))
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
            self.value_lbls[k].config(text=v)
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
        self.refresh()


def guarded_span(app: "DfitApp", fn):
    def wrapped(lo, hi):
        fn(app.state, lo, hi)
        app.refresh()
    return wrapped


def _to_float(s: str):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None
