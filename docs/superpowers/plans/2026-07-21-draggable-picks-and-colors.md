# Draggable Start/Shut-in Lines + Plot Colors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recolor the overview/diagnostic plots to rate=blue, surface-pressure=red, BHP=black, and replace click-to-place of the injection-start/shut-in markers with drag-to-move that snaps to the nearest sample and preserves the current view.

**Architecture:** `plots.py` render functions get color changes and gid tags on the two marker lines. A new `DragLineController` in `picks.py` captures a tagged line on mouse-press, moves it live on motion, and commits the snapped index on release. `ui.py` wires the controller for the overview step, adds a `preserve_view` flag to `refresh()`, and updates the hint text. The overview click handler is retired.

**Tech Stack:** Python 3, numpy, matplotlib (TkAgg for the app, Agg for headless tests), Tkinter, pytest.

## Global Constraints

- Marker picks stay integer sample indices: `PickState.start_idx` / `shutin_idx` (`int | None`). No continuous placement.
- `plots.py` renderers must stay Tkinter-free and headless-renderable (Agg), per the module docstring.
- Overview x-axis is in **hours from file start**; conversions use `td.t_s / 3600.0`.
- Colors: rate + rate axis = `tab:blue`; BHP curve/axis = `black`; surface-pressure curve/axis = `tab:red`. Marker colors (orange dashed start, red solid shut-in) unchanged.
- Do not recolor `render_loglog` — its blue `dp` curve is a derived pressure change, not a raw BHP trace.
- Project is not a git repo; the "Commit" steps below use git only if a repo is later initialized. If `git` reports "not a repository", skip the commit step and continue.

---

### Task 1: Headless test harness + synthetic TestData

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/helpers.py`
- Create: `tests/test_harness_smoke.py`

**Interfaces:**
- Produces: `tests/helpers.py::make_testdata(n=600, dt=1.0) -> io_load.TestData` — a `TestData` with a linear time base and columns `"PRESSURE"` (a rise-then-decline curve), `"RATE"` (zero, ramp to ~8 bpm between indices 100 and 300, then zero), `"VOLUME"` (cumulative). Column names discoverable via `td.columns`.
- Produces: `tests/helpers.py::overview_state(td) -> PickState` — a `PickState` with `pressure_col="PRESSURE"`, `rate_col="RATE"`, `volume_col="VOLUME"`, `start_idx=100`, `shutin_idx=300`.

- [ ] **Step 1: Inspect `io_load.TestData` to match the real constructor**

Run: read `dfit_tool/io_load.py` and note the exact `TestData` fields (`t_s`, `columns`, `column(name)`, `n`) and how `load_csv` builds it. Build `make_testdata` to construct a `TestData` the same way (construct directly from arrays/dict — do not write a temp CSV).
Expected: you can list the constructor signature and the `column()` lookup mechanism.

- [ ] **Step 2: Write `tests/conftest.py` to force the Agg backend**

```python
import matplotlib
matplotlib.use("Agg")
```

- [ ] **Step 3: Write `tests/__init__.py` (empty) and `tests/helpers.py`**

Implement `make_testdata` and `overview_state` per the Interfaces block, constructing `TestData` exactly as inspected in Step 1. Rate array: `np.zeros(n)`, then `rate[100:300]` ramps 0->8->0 (use `np.linspace` up then down). Pressure: `np.concatenate` of a rise over `[100:300]` then an exponential-ish decline; any smooth curve is fine. Volume: `np.cumsum(rate) * dt / 60.0`.

- [ ] **Step 4: Write the smoke test**

```python
from tests.helpers import make_testdata, overview_state

def test_harness_builds_testdata():
    td = make_testdata()
    assert td.n == 600
    assert "PRESSURE" in td.columns
    st = overview_state(td)
    assert st.start_idx == 100 and st.shutin_idx == 300
```

- [ ] **Step 5: Run it**

Run: `python -m pytest tests/test_harness_smoke.py -v`
Expected: PASS. (If pytest is missing: `python -m pip install pytest` first.)

- [ ] **Step 6: Commit**

```bash
git add tests/
git commit -m "test: headless Agg harness + synthetic TestData"
```

---

### Task 2: Colors in the render functions

**Files:**
- Modify: `dfit_tool/plots.py` (`render_overview` ~L34-70; `render_isip` ~L82; `render_gfunction` ~L110-118; `render_porepressure` ~L207)
- Create: `tests/test_plot_colors.py`

**Interfaces:**
- Consumes: `make_testdata`, `overview_state` from Task 1.
- Produces: no new symbols; behavior change only.

- [ ] **Step 1: Write the failing color test**

```python
import numpy as np
from matplotlib.figure import Figure
from dfit_tool.model import compute_all
from dfit_tool import plots
from tests.helpers import make_testdata, overview_state

def _overview_axis(state):
    td = make_testdata()
    res = compute_all(state, td)
    fig = Figure(); ax = fig.add_subplot(111)
    plots.render_overview(ax, td, state, res)
    return fig, ax

def test_rate_trace_is_blue_and_bhp_black_when_bhp():
    st = overview_state(make_testdata()); st.pressure_is_bhp = True
    fig, ax = _overview_axis(st)
    # pressure trace on primary axis is black
    press_line = ax.get_lines()[0]
    assert press_line.get_color() == "black"
    # rate trace lives on the twin axis; find it among figure axes
    twins = [a for a in fig.axes if a is not ax]
    rate_line = twins[0].get_lines()[0]
    assert rate_line.get_color() == "tab:blue"

def test_pressure_trace_is_red_when_surface():
    st = overview_state(make_testdata()); st.pressure_is_bhp = False
    _, ax = _overview_axis(st)
    assert ax.get_lines()[0].get_color() == "tab:red"
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_plot_colors.py -v`
Expected: FAIL (current colors are `tab:blue` pressure / `tab:green` rate).

- [ ] **Step 3: Recolor `render_overview`**

In `dfit_tool/plots.py` `render_overview`, replace the pressure plot color and the rate axis colors:

```python
    press_color = "black" if state.pressure_is_bhp else "tab:red"
    ax.plot(xt, xp, color=press_color, lw=0.8,
            label="BHP" if state.pressure_is_bhp else "pressure")
    ax.set_xlabel("time from file start (h)")
    ax.set_ylabel("pressure (psi)")
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        _, xr = _decimate(t_h, res.rate_all)
        ax2.plot(xt, xr, color="tab:blue", lw=0.7, alpha=0.7)
        ax2.set_ylabel("rate (bpm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
```

- [ ] **Step 4: Recolor BHP curves in the diagnostic renderers**

In `render_isip`, change the BHP decline plot from `color="tab:blue"` to `color="black"`.
In `render_gfunction`, change the BHP-vs-G plot from `color="tab:blue"` to `color="black"` and its `ax.set_ylabel("BHP (psi)")` stays (add nothing); leave `dP/dG` red.
In `render_porepressure`, change the BHP-vs-x plot from `color="tab:blue"` to `color="black"`.
Leave `render_loglog` and `render_tangent` unchanged.

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest tests/test_plot_colors.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dfit_tool/plots.py tests/test_plot_colors.py
git commit -m "feat: rate=blue, BHP=black, surface pressure=red plot colors"
```

---

### Task 3: gid tags on the marker lines + DragLineController

**Files:**
- Modify: `dfit_tool/plots.py` (`render_overview` axvlines ~L52-55)
- Modify: `dfit_tool/picks.py` (add `DragLineController`; remove `handle_overview_click`)
- Create: `tests/test_drag_controller.py`

**Interfaces:**
- Consumes: `make_testdata`, `overview_state`.
- Produces:
  - `render_overview` sets `gid="start"` on the injection-start axvline and `gid="shutin"` on the shut-in axvline, with linewidths `1.6` and `1.8`.
  - `picks.DragLineController(canvas, ax, handlers, guard=None, tol_px=6)` where `handlers: dict[str, Callable[[float], None]]` maps gid -> on-release callback receiving the released x (data coords); `guard: Callable[[], bool] | None` returns True to block interaction. Methods: `disconnect()`. On release it calls the matching gid's handler with the final xdata.

- [ ] **Step 1: Add gids + linewidths in `render_overview`**

Replace the two axvline calls:

```python
    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color="tab:orange", ls="--", lw=1.6,
                   label="injection start", gid="start")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color="tab:red", ls="-", lw=1.8,
                   label="shut-in", gid="shutin")
```

- [ ] **Step 2: Write the failing controller test**

```python
import types
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dfit_tool.model import compute_all
from dfit_tool import plots, picks
from tests.helpers import make_testdata, overview_state

def _built_overview():
    td = make_testdata(); st = overview_state(td); res = compute_all(st, td)
    fig = Figure(); ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_overview(ax, td, st, res)
    fig.canvas = canvas
    return td, st, ax, canvas

def _press_at_line(ax, canvas, gid, button=1):
    line = next(l for l in ax.get_lines() if l.get_gid() == gid)
    x = line.get_xdata()[0]
    px, py = ax.transData.transform((x, np.mean(ax.get_ylim())))
    ev = types.SimpleNamespace(inaxes=ax, xdata=x, ydata=np.mean(ax.get_ylim()),
                               x=px, y=py, button=button)
    return line, ev

def test_press_captures_and_release_calls_handler():
    td, st, ax, canvas = _built_overview()
    got = {}
    ctrl = picks.DragLineController(
        canvas, ax, handlers={"start": lambda xd: got.__setitem__("start", xd),
                              "shutin": lambda xd: got.__setitem__("shutin", xd)})
    line, press = _press_at_line(ax, canvas, "start")
    ctrl._on_press(press)
    assert ctrl._active is line
    # move: new xdata halfway across the axis
    newx = float(np.mean(ax.get_xlim()))
    ctrl._on_motion(types.SimpleNamespace(inaxes=ax, xdata=newx, x=0, y=0))
    assert line.get_xdata()[0] == newx
    ctrl._on_release(types.SimpleNamespace(inaxes=ax, xdata=newx, x=0, y=0, button=1))
    assert got["start"] == newx and ctrl._active is None

def test_guard_blocks_capture():
    td, st, ax, canvas = _built_overview()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None},
                                    guard=lambda: True)
    line, press = _press_at_line(ax, canvas, "start")
    ctrl._on_press(press)
    assert ctrl._active is None

def test_press_far_from_any_line_captures_nothing():
    td, st, ax, canvas = _built_overview()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    # cursor pixel far from both lines
    ev = types.SimpleNamespace(inaxes=ax, xdata=ax.get_xlim()[0], ydata=0,
                               x=-9999, y=-9999, button=1)
    ctrl._on_press(ev)
    assert ctrl._active is None
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/test_drag_controller.py -v`
Expected: FAIL with `AttributeError: module 'dfit_tool.picks' has no attribute 'DragLineController'`.

- [ ] **Step 4: Implement `DragLineController` in `picks.py`**

Add after `SpanController`:

```python
class DragLineController:
    """Drag gid-tagged vertical lines within an Axes; commit the released x per gid.

    ``handlers`` maps an axvline gid to ``on_release(x_data)``. During a drag the line is moved
    live (no recompute); on release the matching handler is called with the final x. ``guard()``
    returning True blocks capture (e.g. while the toolbar zoom/pan mode is active).
    """

    def __init__(self, canvas, ax, handlers, guard=None, tol_px: float = 6.0):
        self.canvas = canvas
        self.ax = ax
        self.handlers = handlers
        self.guard = guard or (lambda: False)
        self.tol_px = tol_px
        self._active = None
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def _lines(self):
        return [l for l in self.ax.get_lines() if l.get_gid() in self.handlers]

    def _on_press(self, event):
        if event.inaxes is not self.ax or event.button != 1 or event.x is None:
            return
        if self.guard():
            return
        best, best_d = None, self.tol_px
        for line in self._lines():
            lx = line.get_xdata()[0]
            px = self.ax.transData.transform((lx, 0.0))[0]
            d = abs(px - event.x)
            if d <= best_d:
                best, best_d = line, d
        self._active = best

    def _on_motion(self, event):
        if self._active is None or event.inaxes is not self.ax or event.xdata is None:
            return
        self._active.set_xdata([event.xdata, event.xdata])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        line = self._active
        self._active = None
        x = event.xdata if (event.inaxes is self.ax and event.xdata is not None) \
            else line.get_xdata()[0]
        handler = self.handlers.get(line.get_gid())
        if handler is not None:
            handler(float(x))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
```

- [ ] **Step 5: Remove the retired overview click handler**

Delete `handle_overview_click` from `picks.py` (the function defined ~L83-90). Leave `_nearest` (still used by the ui release handlers and other handlers).

- [ ] **Step 6: Run tests to verify pass**

Run: `python -m pytest tests/test_drag_controller.py tests/test_plot_colors.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
git add dfit_tool/plots.py dfit_tool/picks.py tests/test_drag_controller.py
git commit -m "feat: DragLineController + gid-tagged marker lines; retire overview click"
```

---

### Task 4: Wire drag into the app + preserve view on release

**Files:**
- Modify: `dfit_tool/ui.py` (`_attach_controllers` overview branch ~L238-242; `refresh` ~L210-222)

**Interfaces:**
- Consumes: `picks.DragLineController`, `picks._nearest`.
- Produces: `DfitApp.refresh(self, preserve_view: bool = False)`.

- [ ] **Step 1: Add `preserve_view` to `refresh`**

Replace the `refresh` method body so it captures and restores limits when asked:

```python
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
```

- [ ] **Step 2: Replace the overview branch in `_attach_controllers`**

```python
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
```

Leave the `isip`, `gfunction`, `tangent`, `loglog`, `porepressure` branches unchanged.

- [ ] **Step 3: Confirm nothing else references the removed handler**

Run: `grep -rn "handle_overview_click" dfit_tool/`
Expected: no matches.

- [ ] **Step 4: Import check**

Run: `python -c "import dfit_tool.ui, dfit_tool.picks, dfit_tool.plots"`
Expected: no error.

- [ ] **Step 5: Full test run**

Run: `python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Manual verification (Tkinter — needs a display)**

Run: `python -m dfit_tool.app <path-to-a-DFIT-csv>`
Verify: (1) overview shows rate in blue, pressure in black (BHP mode) or red (surface, uncheck "is BHP"); (2) grab the orange start line and the red shut-in line and drag each — they follow the cursor and snap to a sample on release; (3) te/Vinj/qmax update after a drag; (4) zoom in with the toolbar, drag a line, release — the view stays put; (5) with zoom/pan mode active, clicking does not move a line (drag is blocked by the guard).
Expected: all behaviors hold. Note any failures.

- [ ] **Step 7: Commit**

```bash
git add dfit_tool/ui.py
git commit -m "feat: drag start/shut-in lines, preserve view on release"
```

---

## Self-Review Notes

- Spec coverage: colors (Task 2), gids + linewidth + DragLineController + retire click (Task 3), app wiring + preserve_view + hint (Task 4), test harness (Task 1). Log-log deliberately excluded per spec.
- Types consistent: `DragLineController(canvas, ax, handlers, guard, tol_px)` and `refresh(preserve_view)` used identically where defined and consumed.
- No placeholders; every code step shows full code.
