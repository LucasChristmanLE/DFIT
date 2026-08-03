# CLAUDE.md

Guidance for working in this repository.

## Overview

An interactive tool for interpreting a single diagnostic fracture injection test (DFIT)
by the compliance method. A Tkinter/ttk shell hosts an embedded matplotlib canvas. The
interpreter opens one data file (CSV or Fracpro `.DBS`), maps its channels, and walks six
workflow steps, making draggable picks on each plot. Every reported number is derived from
one pure function, `model.compute_all`.

Scope is one well at a time. There is no batch queue, master results log, or cross-test
aggregation. Permeability is out of scope.

The six steps (`ui.py:STEPS`): overview → isip → gfunction → tangent → loglog →
porepressure.

## Commands

Dependencies are not on the system Python. Always use the project venv:

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe

Python 3.14. The venv is kept outside the project on purpose so OneDrive does not sync
thousands of package files.

Run the app:

    start-app.cmd                                                   # double-click, no file loaded
    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app [path/to/file]

Run the tests (from the repo root):

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest

`start-app.ps1` self-provisions: if the venv python is missing it creates the venv
(`py -3.14 -m venv`, falling back to `C:\Python314\python.exe`), then probes
`import numpy, pandas, matplotlib, scipy, openpyxl` and runs `pip install -r requirements.txt`
only when that probe fails. `start-app.cmd` runs the `.ps1` with
`-NoProfile -ExecutionPolicy Bypass` and keeps the window open on error.

Pinned deps (`requirements.txt`): numpy 2.5.1, pandas 3.0.3, matplotlib 3.11.1,
scipy 1.18.0, openpyxl 3.1.5, pytest 9.1.1.

## Architecture

The package `dfit_tool/` is layered. Lower layers never import higher ones.

- **Leaf math (numpy only).** `gfunction.py` is the Nolte G-function and G-time (α=1
  default, α=0.5 option). `interpret.py` is the interpretation math: te, apparent/effective
  ISIP, Shmin, net pressure, pore pressure, plus the `suggest_*` auto-pick helpers. These
  take arrays and pick parameters and return numbers.
- **Compute core (pure python/numpy).** `resample.py` does the 30-psi pressure-increment
  resampling, the diagnostic derivatives (dP/dG, G·dP/dG, d²P/dG², t·dP/dt), and the
  tail guard. `model.py` holds `PickState` (the serializable set of interpreter choices),
  `DerivedResults`, and `compute_all(state, td)`. `compute_all` is the single source of
  truth: it produces every reported value and every array the plots need.
- **IO.** `io_load.py` loads CSV and the reverse-engineered Fracpro `.DBS` binary format
  (`load()` dispatches on extension), parses datetimes including leaked Excel serials,
  suggests channel roles, and converts surface pressure to BHP hydrostatically
  (`BHP = WHP + 0.052·mw·tvd`, valid post-shut-in where flow → 0). `questionnaire.py`
  parses a `*questionnaire*.xlsx` next to the data file for fluid density and TVD.
- **Interaction (matplotlib only, no Tkinter).** `picks.py` has the event controllers
  (`DragLineController`, `AnchorLineController`, `DraggablePointController`,
  `SpanController`, `HoverCursorController`, and the `_CaptureGate` press arbiter), the
  pure `commit_*` functions that translate finished geometry into `PickState` changes, and
  the per-step `seed_*` functions. `plots.py` has the `render_*` renderers. `sliders.py`
  has `PanRangeSlider`.
- **Shell.** `ui.py` (`DfitApp`) is the only Tkinter consumer. It wires the per-step
  pickers to a recompute-and-redraw loop and holds no interpretation logic. `app.py` just
  launches it.

Import graph:

    app → ui → {io_load, picks, plots, sliders, model, interpret, questionnaire}
    picks, plots → model, io_load, interpret
    model → interpret, resample, io_load
    resample → gfunction

`picks.py`, `plots.py`, and `sliders.py` never import Tkinter, so the whole interaction
layer runs headless under the Agg backend and the shell could be ported off Tkinter without
touching them.

Persistence is per-test JSON only, via `PickState.to_json`/`from_json`, saved and loaded
through a file dialog (`DfitApp._save_picks`/`_load_picks`). `DerivedResults` is never
serialized. `model._decode` migrates legacy saves: it maps the old `eff_isip_line` pick to
`min_dpdg_G`, rebuilds tuples and `TangentPick`, and filters unknown keys so old or foreign
JSON never raises. `step_status` (the breadcrumb history) rides along in the same JSON;
`infer_step_status` backfills it for saves made before it existed.

On the last step (`porepressure`) the stepbar's "Next >" button becomes a bolded "Finish"
button (`ui.py:_advance`/`_update_stepbar`). One click (`ui.py:_finish`) silently re-saves
the picks JSON to `<stem>_picks.json` and writes a PNG of all six step plots, in their
current zoom state, to a `<stem> DFIT plots/` subfolder -- both next to the loaded data
file. The PNG export itself is headless: `plots.render_step_figure`/`save_all_step_pngs`
take no Tkinter and replicate `ui.refresh`'s view-resolution logic (including the
gfunction-specific clamps) against an offscreen `Figure`, so `ui._finish` just resolves
`self._views` into the plain tuples that function expects. There is still no CSV results
log across tests -- `_finish` has a placeholder comment for that.

## Conventions and invariants

Preserve these when changing the code.

- **`compute_all` is the single source of truth.** New reported values are computed there,
  not in the UI. `ui.py` only displays what `compute_all` returns.
- **Keep `picks.py`, `plots.py`, and `sliders.py` free of Tkinter.** They are the
  headless-testable layer.
- **Renderers never set view limits.** `render_*` leave the Axes autoscaled and return a
  `ViewDefaults(xlim, ylim, y2lim)`. `ui.py` owns per-step view state (`_views`) and applies
  either the stored view or the renderer default. Do not call `set_xlim`/`set_ylim` inside a
  renderer.
- **Hit-test through own-axes pixel transforms, never `event.inaxes`.** A twin axes (e.g.
  the overview's rate `twinx`) owns `inaxes` over the shared region, so an identity check
  against a specific Axes never matches. Use `_axes_contains_pixel`/`_data_from_pixel`.
- **Slider `on_changed` callbacks must never call `refresh()`.** `refresh()` calls
  `fig.clf()`, which destroys the slider mid-drag. Callbacks only `set_xlim`/`set_ylim`,
  mutate the current `ViewState`, and `draw_idle()`. A full refresh happens only on step
  change, pick commit, Apply, scenario change, or Reset view. Hold slider references on
  `self` — matplotlib keeps no strong reference and GC otherwise kills the callbacks.
- **Diagnostic derivatives are reported positive-up for a declining pressure** (negated),
  matching how G-function and log-log plots are conventionally drawn.

## Testing

Tests live in `tests/` (~21 files). `tests/conftest.py` forces the Agg backend before
`matplotlib.pyplot` is imported, so the suite runs with no display or Tk. Synthetic data
comes from `tests/helpers.make_testdata` (a DFIT-shaped `TestData`) and
`helpers.overview_state`; tests drive controllers with real synthesized `MouseEvent`s.
GUI-only paths are exercised by binding real `DfitApp` methods onto duck-typed stand-ins
rather than constructing a real `tk.Tk()`.

New logic should land in a headless-testable layer (`model`, `interpret`, `resample`,
`picks` commits/controllers, `plots` renderers) and get a test there, rather than inside the
Tkinter shell.

## Domain and methodology

Compliance-method interpretation per McClure et al. (URTeC-2019-123) and the ResFrac
practical guidelines. Refs are in `Refs/` (gitignored).

Per-test deliverables:

- **Apparent ISIP** — early BHP-decline tangent extrapolated back to the shut-in instant
  (FracPro-style construction; a manual pick on the isip step).
- **Effective ISIP** — the P-vs-G straight line extrapolated to G = 0.
- **Shmin, compliance** — contact pressure − 75 psi (`interpret.COMPLIANCE_OFFSET_PSI`).
- **Shmin, tangent** — BHP at the G·dP/dG through-origin departure (closure) point.
- **Shmin, variable** — BHP at the G-time midpoint of the contact and closure picks. This
  third "variable-compliance" method is computed by `compute_all` and reported alongside the
  other two (effective ISIP, Shmin, closure time, and net pressure each have compliance,
  tangent, and variable columns in the panel).
- **Pore pressure** — intercept of the late-time postclosure line on the t^(−1/2) or t^(−1)
  axis chosen by the postclosure scenario.

**Net pressure** = reference ISIP − Shmin, using each method's own effective ISIP (falling
back to apparent ISIP per method when that effective ISIP is unavailable).

**Resampling.** After shut-in, keep one (time, pressure) point each time BHP has dropped
≥ 30 psi below the last kept point. This collapses ~10⁶ raw rows to a few hundred, dense
early and sparse late, which stabilizes the numerical derivatives. It replaces time-domain
smoothing. A tail guard stops resampling once the pressure rises above its running minimum
(non-monotonic late data).

**G-function.** α = 1 (low-leakoff) is the default; α = 0.5 only if a test exceeds ~1 md.

Closure scenarios drive the contact pick and effective ISIP:

| Scenario | dP/dG shape | Stress pick |
|---|---|---|
| C-A clear | clear "S" (min then rise) | contact at min + 10%, − 75 psi |
| C-B adequate | monotonic with inflection | contact at the inflection, − 75 psi |
| C-C no-contact | monotonic, no inflection | none (no Shmin) |
| C-D rapid | monotonic, concave-up, no tortuosity | apparent ISIP − 100–250 psi (methodology; see note) |

Note: in the current code (`picks.apply_closure_scenario`) C-C and C-D both clear the contact
pick, so `compute_all` produces no compliance Shmin and no effective ISIP for either. C-D
additionally reports `shmin_rapid` = apparent ISIP − 175 psi (the midpoint of the 100–250 psi
range; `interpret.RAPID_CLOSURE_OFFSET_PSI`), shown as its own "Shmin rapid" panel row and in
the G-function title (`interpret.format_shmin_rapid`). Net pressure is deliberately not
derived from it -- there is no `net_pressure_rapid`.

Postclosure scenarios drive the pore-pressure axis:

| Scenario | log-log signature | Pore-pressure axis |
|---|---|---|
| PC-A linear | bends to −1/2 | t^(−1/2) |
| PC-B false-radial | −1 after peak | t^(−1) |
| PC-C false radial to genuine linear | −1 then −1/2 | t^(−1/2) |
| PC-D genuine linear to genuine radial | −1/2 then −1 | either |
| PC-E no trend | peak, no clear slope | t^(−1/2), low confidence |
| PC-F no peak | derivative still rising | pore-pressure step skipped |

This linkage is implemented: `picks.suggest_pp_axis` maps a postclosure scenario string to
`pp_axis` ("tm12"/"tm1") by its `scenario[:4]` prefix, so the mapping is label-independent, and
`ui._on_scenario` applies it whenever the postclosure-scenario combobox changes, also
re-syncing on step entry/load via `_update_panel_visibility`. The side panel's pore-pressure-
axis radios lock (disabled) whenever a scenario dictates the axis. PC-D ("either") and PC-F
("no peak") are intentionally absent from the mapping, so the radios stay enabled and the axis
is left to the analyst -- moot for PC-F, since that scenario skips the axis-picking step
entirely (below). `model._decode` normalizes the four old pre-rename labels (`"PC-C mixed"`,
`"PC-D mixed"`, `"PC-E none"`, `"PC-F none"`) found in older saved picks JSON to their current
form; unrecognized strings pass through untouched.

**PC-F skip.** `model.porepressure_skipped(state)` is true whenever `postclosure_scenario`
starts with `"PC-F"`: the derivative never peaks, so no postclosure line exists and
`compute_all` leaves `pore_pressure` `None` even if a stale `pp_window` pick exists. When it's
true, the pore-pressure step is skipped end to end: `ui._last_step()` reports `"loglog"` so
`_advance`'s Next button becomes "Finish" there instead of on pore pressure; `_goto` redirects
any `"porepressure"` destination (the log-log Skip button, resume-on-load, a breadcrumb click)
to `"loglog"`; `_update_stepbar` force-disables the porepressure breadcrumb even if that step
was visited earlier in the session; and `plots.save_all_step_pngs` omits the porepressure PNG
(the other five keep their `RENDERERS`-order numbering).

An in-app "Interpretation guide" window (opened from the "Interpretation guide..." buttons on
the G-function step's closure-scenario panel and the log-log/pore-pressure steps'
postclosure-scenario panel) shows the ResFrac closure (C-A...C-D) and postclosure (PC-A...PC-F)
figures and explanatory text side by side with the pickers. Content lives in
`dfit_tool/guide_content.py` (headless, no Tkinter); figures are vendored PNGs under
`dfit_tool/assets/guide/`. Both buttons open the same reused `ttk.Notebook` window and just
select their tab (`ui.py:_open_guide`).

## Not built / notes

- No master results log or batch queue/resume. PNG export exists (the Finish button, see
  "Persistence" above) but there is still no CSV results log rolling up multiple tests.
- `scipy` is pinned in `requirements.txt` and probed by `start-app.ps1` but is not currently
  imported anywhere in the package.
- Sample data folders, `Refs/`, and `.superpowers/` are gitignored. Design specs and plans
  from the build are under `docs/superpowers/`.
