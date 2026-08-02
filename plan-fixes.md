# DFIT Tool — Interaction, Zoom, Workflow, and Environment Fixes

## Context

The first build of the DFIT interpretation tool (`dfit_tool/`) works end to end but has five problems found in real use:

1. **Interactive picks stop working after pan/zoom.** Root causes confirmed in code:
   - `NavigationToolbar2Tk` zoom/pan mode is *sticky*: after one zoom/pan, `toolbar.mode` stays set until the button is clicked again. `guarded()` ([ui.py:234-240](dfit_tool/ui.py#L234-L240)) and the `DragLineController` guard ([ui.py:255](dfit_tool/ui.py#L255)) then silently no-op every click/drag — picks "die" after the first zoom.
   - The guard is inconsistent: `guarded_span` ([ui.py:337-341](dfit_tool/ui.py#L337-L341)) never checks toolbar mode, so log-log spans fight an active zoom rectangle.
   - `refresh()` destroys/recreates the Axes every time ([ui.py:219-220](dfit_tool/ui.py#L219-L220)); only the overview drag passes `preserve_view=True`, so every other step snaps back to default limits after any pick. The toolbar's nav stack also goes stale on the dead Axes.
2. **Rectangular zoom is the wrong tool** — replace with per-axis range sliders.
3. **Apparent-ISIP tangent has no drag/pan/rotate** — `handle_isip_click` ([picks.py:162-171](dfit_tool/picks.py#L162-L171)) refits the whole tangent from any click. Same click-refits-everything pattern on the gfunction/tangent steps.
4. **Tangent plot lacks pressure** — `render_tangent` ([plots.py:147-170](dfit_tool/plots.py#L147-L170)) is a single axis with only G·dP/dG.
5. **Workflow is not sequential** — all six step buttons freely clickable, all 13 result labels + both scenario comboboxes always visible, and `seed_defaults` ([picks.py:212-249](dfit_tool/picks.py#L212-L249)) fills every pick at load.
6. **Venv confusion** — `start-app.ps1` hard-codes the venv python with no provisioning and no `requirements.txt`. (Venv is outside the project deliberately: OneDrive must not sync package files.)

## Decisions (user-confirmed)

- **Zoom**: two-thumb `RangeSlider` per axis (x below plot, y beside; second y-slider for twin axes). Dragging the bar middle pans. Remove `NavigationToolbar2Tk` entirely. Add a Reset-view button.
- **Navigation**: breadcrumb step bar + Back/Next/Skip; visited steps re-clickable, future steps disabled.
- **Seeding**: nothing at load; each step's auto-suggestion seeds on first entry to that step.
- **Environment**: keep the venv at `C:\Users\LucasChristman\.venvs\dfit`; `start-app.ps1` auto-creates it and installs a new pinned `requirements.txt` when missing.

## End-state architecture

- `plots.py` renderers become **pure drawing + default-view suggestion**: they never call `set_xlim/set_ylim`; they return a `ViewDefaults(xlim, ylim, y2lim)` dataclass. `ui.py` owns applying either the stored per-step view or the renderer default, and owns slider bounds (= full data extent).
- New `dfit_tool/sliders.py` (matplotlib-only, headless-testable): `PanRangeSlider(RangeSlider)`.
- `picks.py` gains two generalized twinx-safe controllers (`AnchorLineController`, `DraggablePointController`) replacing all click-to-place handlers; stays Tkinter-free.
- `PickState` gains `step_status: dict[str, str]`; `_decode` hardened for old JSONs.
- Task order: 1 → 2, 3 → 4, 5 → 6; 7 independent.

---

## Task 1 — Remove toolbar; per-step ViewState; renderer contract

**Files:** `dfit_tool/plots.py`, `dfit_tool/ui.py`

**plots.py** — add `ViewDefaults` dataclass (`xlim`, `ylim`, `y2lim`, all `Optional[tuple[float,float]]`, `None` = autoscaled full extent). Every `render_*` returns one:
- `render_overview`: delete the auto-zoom block (plots.py:63-69); return the same `span_h`-based window as `ViewDefaults(xlim=...)`.
- `render_isip`: currently pre-filters data to `t_min ∈ [-2, 30]` (plots.py:85) — plot the **full** decimated post-shut-in series instead and return `ViewDefaults(xlim=(-2.0, window_min))`, so the x-slider has a real full extent to zoom within.
- `render_gfunction`: delete `ax2.set_ylim(...)` (plots.py:124-127); return the 95th-percentile clip as `y2lim`.
- `render_tangent`: after the Task 4 twinx relayout, same percentile `y2lim` for G·dP/dG.
- `render_loglog`: `ViewDefaults()`.
- `render_porepressure`: delete `ax.set_xlim(left=0)` (plots.py:216); return `ViewDefaults(xlim=(0.0, autoscaled_xmax))`.

**ui.py** — delete the toolbar (ui.py:105-106), `guarded()` (ui.py:234-240), and `guarded_span` (ui.py:337-341). Keep `DragLineController`'s `guard` param (tests depend on it) but pass none.
- `self._views: dict[str, Optional[ViewState]]` (`ViewState` = xlim/ylim/y2lim), reset on file load.
- Rewrite `refresh()`: compute → `fig.clf()` → render (capturing `ViewDefaults`) → read full autoscaled extents from the axes → if `self._views[step]` is `None`, initialize it from defaults-or-full → apply the view to `ax` (+ twin) → `_build_sliders(...)` → replace `tight_layout()` with a fixed `subplots_adjust(left=0.10, right=0.84, bottom=0.16, top=0.90)` (tight_layout fights the manually placed slider axes) → `_attach_controllers()` → draw → panel updates. `preserve_view` is deleted — the view dict makes preservation unconditional and correct on every step; the overview drag-release call site (ui.py:249) becomes plain `self.refresh()`.
- `_twin_axes()` helper = the one figure axes that isn't `self.ax` and isn't a slider axes (tag slider axes with a gid or keep references to exclude them).
- `_reset_view()`: `self._views[self.step] = None; self.refresh()`.

## Task 2 — Per-axis range-slider zoom

**Files:** new `dfit_tool/sliders.py`, `dfit_tool/ui.py`

**sliders.py** — `PanRangeSlider(matplotlib.widgets.RangeSlider)`: stock `RangeSlider` jumps the *nearest thumb* on a track click, so bar-drag pan must be added. Override the press handling: if the press falls strictly between the two thumbs (beyond a pixel tolerance from each), enter pan mode — record the offset, and on motion `set_val((lo+d, hi+d))` clamped to `[valmin, valmax]` preserving width; near a thumb, defer to stock behavior. Headless-testable with synthetic `MouseEvent`s.

**ui.py** — `_build_sliders(full_x, full_y, full_y2, view, twin)` called from `refresh()` after `fig.clf()` (sliders are rebuilt each refresh, re-initialized from the stored `ViewState`):
- x-slider: `fig.add_axes([0.10, 0.04, 0.68, 0.03])`; y-slider: `[0.87, 0.16, 0.02, 0.74]` vertical; y2-slider (only when a twin exists): `[0.93, 0.16, 0.02, 0.74]` vertical.
- Slider range = full data extent, `valinit` = current view. For the `loglog` step, operate in `log10` space with a `valfmt` converting back (clamp bounds to `1e-12` first); exponentiate in the callback.
- `on_changed` callbacks **only** `set_xlim/set_ylim` on the target axes, mutate the current `ViewState` in place, and `draw_idle()`. They must **never** call `refresh()` — `fig.clf()` would destroy the slider mid-drag. Full refresh (which rebuilds sliders from `ViewState`) happens only on step change, pick commit, Apply, scenario change, Reset view.
- Hold `self._x_slider/_y_slider/_y2_slider` references — matplotlib keeps no strong reference; GC silently kills callbacks otherwise.
- Re-entrancy: callbacks never call `set_val` on themselves/siblings, so no recursion guard is needed; if cross-slider syncing is ever added, gate with an `self._updating` flag (`set_val` always fires observers).

## Task 3 — Generalized draggable-line / point controllers

**Files:** `dfit_tool/picks.py`

- Promote `DragLineController._in_axes/_x_from_pixel` (picks.py:94-102) to module-level `_axes_contains_pixel(ax, event)` / `_data_from_pixel(ax, event)`; all controllers hit-test via `event.x/event.y` through their own `ax.bbox`/`ax.transData`, never `event.inaxes` (a twin axes owns `inaxes` over shared regions — pattern already proven by `test_overview_rate_twin_owns_inaxes_regression`).
- `_CaptureGate` — per-refresh press arbiter shared by all controllers on a step: `try_claim(owner)` at press (first wins), `release()` at release. Prevents double-capture when the eff-ISIP anchor sits near the contact marker.
- `AnchorLineController(canvas, ax, gids, commit_fn, curve=None, allow_anchor=True, allow_body=True, allow_rotate=True, tol_px=8.0, readout_fn=None, gate=None)`:
  - `gids` names the renderer-drawn artists: segment, anchor tick, dashed extension.
  - Hit priority: anchor → ends → body. Motion: **anchor** snaps to nearest `curve` sample and refits slope (`_tangent_from_index`, extracted from the duplicated logic in the old handlers/seeds); **body** translates the anchor by the drag delta, slope unchanged; **end** rotates about the anchor (`slope = (y−ay)/(x−ax)`, divide-by-zero → keep prior slope). Artists updated live; nothing written to `PickState` until release.
  - `readout_fn` (optional) drives a live Text artifact during drag — used for the live apparent-ISIP intercept readout.
  - `curve=None, allow_anchor=False, allow_body=False` = pinned-anchor rotate-only mode for the through-origin line.
- `DraggablePointController(canvas, ax, gid, curve_x, curve_y, commit_fn, tol_px=8.0, gate=None)` — drags a marker snapped along a curve; commit on release.
- Delete `handle_isip_click` / `handle_gfunction_click` / `handle_tangent_click` (picks.py:162-198); replace with pure commit functions: `commit_isip_tangent`, `commit_eff_isip_line`, `commit_closure_line`, `commit_contact_point`, `commit_closure_point`. Spans (`handle_loglog_span`/`handle_pp_span`, `SpanController`) untouched.

## Task 4 — Tangent-step twinx relayout + controller wiring

**Files:** `dfit_tool/plots.py`, `dfit_tool/ui.py`

- `render_tangent`: mirror `render_gfunction` — **P vs G on the left axis (black)**, `ax2 = ax.twinx()` with **G·dP/dG on the right (red)**; through-origin line and closure marker move to `ax2`, gid-tagged (`closure_line_segment`, `closure_point`); return the percentile `y2lim`.
- `render_isip` / `render_gfunction`: draw the tangent constructions as gid-tagged pieces per plan.md step 3 — finite segment + short perpendicular tick at the anchor + dashed extension to the reference vertical (shut-in line for apparent ISIP; G=0 for effective ISIP), replacing today's single continuous line.
- `_attach_controllers` wiring:
  - `isip`: `AnchorLineController` on `self.ax`, `curve=(td.t_s, res.bhp_all)`, readout = live intercept at shut-in; commit → `commit_isip_tangent` + `refresh()`.
  - `gfunction`: `AnchorLineController` (eff-ISIP line, `curve=(G, p)`) + `DraggablePointController` (contact on the P curve), one shared gate, both on `self.ax` (pressure axis).
  - `tangent`: on `ax2` (the twin): rotate-only `AnchorLineController` (through-origin) + `DraggablePointController` (closure on the G·dP/dG curve), shared gate.
  - Update the hint labels accordingly.

## Task 5 — step_status; breadcrumb Back/Next/Skip; step-aware panel

**Files:** `dfit_tool/model.py`, `dfit_tool/ui.py`

- `PickState.step_status: dict[str, str] = field(default_factory=dict)` — values `not_visited` (absent) / `visited` / `done` / `skipped`.
- Harden `_decode` (model.py:100-107): filter the input dict to known dataclass field names (`dataclasses.fields`) so old/foreign JSONs never raise; missing `step_status` falls to the default.
- `_build_stepbar` rewrite: `< Back` | six breadcrumb buttons | `Next >` | `Skip >` … `Reset view` + warning label on the right. Breadcrumb click only honored when that step's status ≠ not_visited; `_update_stepbar()` (called from `refresh`) disables unreached steps and highlights the current one (Accent style if the theme has it, else bold text).
- `_goto(key)`: if the step is `not_visited`, run `_seed_step(key)` (Task 6) and mark `visited`; then `refresh()`. `Next` marks current `done` then advances; `Skip` marks `skipped` then advances.
- Step-aware panel: `FIELD_STEP` map (te/Vinj/qmax → overview; apparent ISIP → isip; effective ISIP/contact P/Shmin compliance/net compliance → gfunction; Shmin tangent/closure P/net tangent/delta closure → tangent; pore pressure → porepressure). In `_update_panel`, show `-` for any field whose owning step is still `not_visited`.
- Scenario widget visibility: wrap the closure-scenario combobox in `frm_cscen` and the postclosure combobox + pp-axis radios in `frm_pcscen`; `_update_panel_visibility()` packs `frm_cscen` only on `gfunction` and `frm_pcscen` only on `loglog`/`porepressure`, using `pack(before=self.sep_before_notes)` so re-showing never reorders the panel.
- `_load_picks` backfill: an old JSON has picks but no `step_status`, which would lock the whole breadcrumb. After `from_json`, if `step_status` is empty, infer it: mark a step `done` when its picks exist (`start_idx/shutin_idx` → overview, `isip_tangent` → isip, `eff_isip_line/contact_G` → gfunction, `closure_G` → tangent, `loglog_window` → loglog, `pp_window` → porepressure).

## Task 6 — Seed-on-entry

**Files:** `dfit_tool/picks.py`, `dfit_tool/ui.py`

- Split `seed_defaults` into `seed_overview(state, td)`, `seed_isip(state, td, res)`, `seed_gfunction(state, res)`, `seed_tangent(state, res)`, `seed_loglog(state, res)`, `seed_pp(state, res)` + a `SEEDERS` dict; delete `seed_defaults`. Each keeps its existing guards (early return when `res.diagnostics` etc. is `None`) so out-of-order entry degrades to "nothing seeded," never a crash. Reuse `_tangent_from_index` from Task 3.
- `ui.py._seed_step(key)`: `res = compute_all(self.state, self.td)` then dispatch (overview/isip get `td` too). Remove the load-time `picks.seed_defaults(...)` call (ui.py:180).
- `_load`: reset `self.state = PickState()` and `self._views = {k: None ...}` before `_sync_state_from_widgets()` (which must run after the reset since it writes into state), then `_goto("overview")`. This intentionally stops picks leaking between files opened in one session.

## Task 7 — Venv auto-provision

**Files:** new `requirements.txt`, `start-app.ps1`

- `requirements.txt` pinned to the proven venv versions: `numpy==2.5.1`, `pandas==3.0.3`, `matplotlib==3.11.1`, `scipy==1.18.0`, `openpyxl==3.1.5`, `pytest==9.1.1`.
- `start-app.ps1`: keep the existing try/catch + pause structure. Before launch: if the venv python is missing, create the venv (`py -3.14 -m venv` if `py` exists, else `C:\Python314\python.exe -m venv`); then probe `& $python -c "import numpy, pandas, matplotlib, scipy, openpyxl"` and only when that fails run `pip install -r requirements.txt`. Cheap probe (~0.3 s) keeps normal launches fast while self-healing an incomplete venv.

---

## Tests (existing headless pattern: Agg via conftest, synthetic data via `tests/helpers.make_testdata`, real `MouseEvent`s driving `_on_press/_on_motion/_on_release`)

- `test_view_state.py` — each renderer returns `ViewDefaults`; axes keep the unclipped full extent (gfunction `ax2` includes the spike; porepressure xmin not forced to 0 by the renderer); applying defaults vs a stored view yields the expected limits.
- `test_pan_range_slider.py` — thumb drag vs bar-drag pan (width preserved, clamped at bounds); log-space round-trip for the loglog step.
- `test_anchor_line_controller.py` — anchor drag snaps + refits slope; body drag translates without slope change; end drag rotates without moving anchor; pinned-anchor mode ignores anchor/body; twinx regression on the tangent step's `ax2` (analog of `test_overview_rate_twin_owns_inaxes_regression`).
- `test_draggable_point_controller.py` — snap-to-nearest-sample; commit receives correct x.
- `test_capture_gate.py` — two controllers, overlapping hit zones: first claimer wins.
- `test_step_status.py` — `from_json` on JSON missing `step_status` and on JSON with unknown extra keys; `_load_picks`-style backfill inference.
- `test_seed_steps.py` — each `seed_*` against partial states (e.g. `seed_isip` with no injection window) no-ops gracefully.
- Re-run `test_drag_controller.py` / `test_plot_colors.py` unchanged as regression (neither asserts on axis limits).

## Risks / edge cases

- **Slider lifecycle**: `on_changed` must never trigger `refresh()` (clf() kills the widget mid-gesture); commits happen only on `button_release_event`. `set_val` always fires observers — no self/sibling `set_val` in callbacks.
- **Vertical RangeSlider feel**: supported in matplotlib 3.11.1 but verify thumb behavior near bounds in the live Tk app; documented fallback is horizontal pan+zoom sliders.
- **Twinx event ownership**: all hit-testing via own-ax pixel transforms, never `event.inaxes`.
- **Toolbar removal loses the Save-figure button** — accepted; PNG export arrives later with `report.py` per plan.md.
- **Renderer contract change**: direct callers no longer get baked-in zoom; note in `plots.py` docstring.
- **Double `compute_all` on step entry** (seed then refresh) — negligible at a few hundred resampled points; don't pre-optimize.

## Verification

1. `pytest` — full suite green (old + new tests) using the venv python.
2. Launch via `start-app.cmd` after renaming the venv dir aside — confirm it recreates and pip-installs, then launches; restore/relaunch — confirm the probe skips installation.
3. Load `2019.02.12_PDC_Argentine State 7170 4U B4H_Final Data.csv`, then walk the workflow:
   - Overview: zoom/pan with all three sliders, then drag start/shut-in lines — drags still work after any slider use; view survives the pick; Reset view restores the default window.
   - ISIP: tangent appears on first entry (seeded); drag anchor along the curve, pan the body, rotate an end — intercept readout updates live; zoom in tight and repeat.
   - G-function: closure-scenario combo visible here (and only here); drag eff-ISIP line and contact marker; right-axis slider clips the dP/dG spike.
   - Tangent: pressure (left, black) + G·dP/dG (right, red); rotate the through-origin line; drag the closure point.
   - Log-log: postclosure combo + pp-axis radios visible here; span select still works; log-space sliders behave.
   - Panel: fields show `-` until their step is reached; breadcrumb blocks unvisited steps; Back/Next/Skip statuses persist through Save picks → reopen → Load picks (old pre-`step_status` JSONs load with inferred statuses).
