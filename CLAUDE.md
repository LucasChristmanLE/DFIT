# CLAUDE.md

Guidance for working in this repository.

## Overview

An interactive tool for interpreting a single diagnostic fracture injection test (DFIT)
by the compliance method. A Tkinter/ttk shell hosts an embedded matplotlib canvas. The
interpreter opens one data file (CSV or Fracpro `.DBS`), maps its channels, and walks six
workflow steps, making draggable picks on each plot. Every reported number is derived from
one pure function, `model.compute_all`.

Interpretation is one well at a time. Two ways to get there: single-file mode (open one CSV
or `.DBS` directly, today's original flow) or folder mode ("Open Folder…"), which scans a
root folder into a queue of tests, shows that queue in a left sidebar, and rolls every test's
results up into a per-root `dfit_log.csv` master log. The sidebar and the log exist only in
folder mode; single-file mode is otherwise unchanged. There is still no cross-test
aggregation beyond that one log (no charts, no rollup stats). Permeability is out of scope.

The six steps (`ui.py:STEPS`): overview → isip → gfunction → tangent → loglog →
porepressure.

## Commands

Dependencies are not on the system Python. Always use the project venv:

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe

Python 3.14. The venv is kept outside the project on purpose so OneDrive does not sync
thousands of package files.

Run the app:

    start-app.cmd                                                   # double-click, no file loaded
    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app [path/to/file|folder]

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
- **Folder-mode persistence.** `store.py` is Tk-free, like `model.py`, so it is unit-testable
  headless. `scan_root` does the depth-1 scan of an opened root: one `TestEntry` per immediate
  subdirectory holding data files (folder layout), plus one per loose data file or same-stem
  csv+dbs pair directly in the root (flat layout); a loose-file entry whose test_id collides
  with a subfolder entry is dropped in favor of the subfolder (with a warning attached) rather
  than crashing the queue on a duplicate iid. Picks persist to a per-test
  `<folder>/<test_id>.dfit_picks.json`, written atomically (temp file + `os.replace`), same
  contract as `PickState.to_json`/`from_json`. `status_for` derives a test's queue status
  ("new"/"in_progress"/"done"/"skipped") from its saved `PickState`: `"done"` and
  `"in_progress"` are purely derived from `step_status` -- ground-truthing it, including the
  PC-F clause (`porepressure` counts as complete under PC-F even though that scenario never
  gets a `step_status` entry for it, since the step is skipped end to end; see the PC-F section
  below) -- but `"skipped"` can also come from `state.explicit_status`, the whole-test
  Skip-test button's override, which short-circuits the derivation regardless of how far the
  steps got. `load_log`/`save_log` read and atomically write the
  per-root `dfit_log.csv`; `build_log_row` maps one test's `PickState`/`DerivedResults` into a
  `LOG_COLUMNS`-shaped row (it computes nothing itself), and `upsert_log_row` replaces-or-
  appends by `test_id`.
- **Shell.** `ui.py` (`DfitApp`) is the only Tkinter consumer. It wires the per-step
  pickers to a recompute-and-redraw loop and holds no interpretation logic. `app.py` just
  launches it, accepting either a file or a folder path on the command line.

Import graph:

    app → ui → {io_load, picks, plots, sliders, model, interpret, questionnaire, store}
    picks, plots → model, io_load, interpret
    model → interpret, resample, io_load
    resample → gfunction
    store → model, questionnaire

`picks.py`, `plots.py`, and `sliders.py` never import Tkinter, so the whole interaction
layer runs headless under the Agg backend and the shell could be ported off Tkinter without
touching them.

Persistence is per-test JSON, via `PickState.to_json`/`from_json`. In single-file mode it is
saved and loaded through a file dialog (`DfitApp._save_picks`/`_load_picks`) to a path the
analyst chooses. In folder mode there is no file dialog for picks: `ui.py` saves through
`store.save_picks_for`/`store.load_picks_for` to the fixed per-test
`<test_id>.dfit_picks.json` next to the test's data files -- on queue navigation
(`_save_current_queue_picks`), Finish, and Skip test. `DerivedResults` is never serialized
either way. `model._decode` migrates legacy saves: it maps the old `eff_isip_line` pick to
`min_dpdg_G`, rebuilds tuples and `TangentPick`, and filters unknown keys so old or foreign
JSON never raises. `step_status` (the breadcrumb history) rides along in the same JSON;
`infer_step_status` backfills it for saves made before it existed.

On the last step (`porepressure`) the stepbar's "Next >" button becomes a bolded "Finish"
button (`ui.py:_advance`/`_update_stepbar`). One click (`ui.py:_finish`) saves picks and
writes a PNG of all six step plots, in their current zoom state, to a `<stem> DFIT plots/`
subfolder next to the loaded data file. In single-file mode (`current_entry` is None) that is
byte-for-byte the original behavior: picks re-save to `<stem>_picks.json`, no log write. In
folder mode, picks save through `store.save_picks_for` instead (no `<stem>_picks.json`
duplicate), Finish also upserts and writes the current test's `dfit_log.csv` row
(`ui._write_log_row`, shared with Skip test), and then advances to the next `"new"` queue
entry via `ui._advance_queue` (or reports the queue is exhausted). The PNG export itself is
headless: `plots.render_step_figure`/`save_all_step_pngs` take no
Tkinter and replicate `ui.refresh`'s view-resolution logic (including the gfunction-specific
clamps) against an offscreen `Figure`, so `ui._finish` just resolves `self._views` into the
plain tuples that function expects.

**Folder mode.** "Open Folder…" (`ui._open_folder_path`) scans the chosen root via
`store.list_tests`, populates the sidebar queue Treeview (row iid = `test_id`), and
auto-opens the first `"new"`-status test (or the first entry if none are new). There are
exactly two ways to end a test, both in the bottom stepbar: Finish (completed it) and Skip
test (park it) -- `ui._advance_queue` is their shared auto-advance tail, scanning circularly
from just after the current entry for the next `"new"`-status one and reporting when the
queue is exhausted rather than looping forever. `"done"` and `"in_progress"` are purely
derived from `step_status` (all six steps accounted for and none skipped is `"done"`, all
accounted for with >=1 skipped is `"skipped"`, otherwise `"in_progress"`); `"done"` is never a
manual choice. Skip test (`ui._skip_test`) is the one capability that has no other
expression: a toggle button that flags the whole test `state.explicit_status = "skipped"`
regardless of how far its steps got, saves picks, writes the log row, refreshes the queue row,
and advances -- or, when the test is already flagged (the button then reads "Unskip test"),
clears the flag, saves/logs/refreshes, and stays put. `store.status_for` checks
`explicit_status` before falling back to the `step_status` derivation. Finish clears the flag
too -- completing a test un-parks it -- but it preserves a per-step `"skipped"` written by
"Skip >" on the step it is invoked from (reachable on the last step, where `next_step` clamps,
and on `loglog` under PC-F), so a test finished with a skipped step still reports `"skipped"`.
The Source dropdown
(CSV/DBS) is enabled only when a test has both files available; switching sources is treated
as a different data file, so it resets that test's picks after a confirm dialog
(`ui._on_source_change`).

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
  other two (Shmin, closure time, and net pressure each have compliance, tangent, and
  variable rows in the panel; effective ISIP shows only the compliance row there).
- **Near-wellbore complexity** — apparent ISIP − the shared reference effective ISIP. The
  near-wellbore friction and tortuosity that is in the early-decline extrapolation but has
  dissipated by the time the P-vs-G line is fit. Shown as the "NWB complexity" panel row and
  logged to the `near_wellbore_complexity` column.
- **Pore pressure** — intercept of the late-time postclosure line on the t^(−1/2) or t^(−1)
  axis chosen by the postclosure scenario.

**Net pressure** = shared reference ISIP − Shmin. All three methods (compliance, tangent,
variable) subtract their own Shmin from one shared reference ISIP: the compliance effective
ISIP, falling back to the tangent effective ISIP, else undefined (no apparent-ISIP fallback,
so a net pressure is blank when neither effective ISIP exists or that method's Shmin is
absent). `compute_all` records the source that fed the reference in
`DerivedResults.net_pressure_isip_source` ("compliance"/"tangent"/""), logged to the
`net_pressure_isip_source` column of `dfit_log.csv`. The tangent and variable effective ISIPs
are kept in the CSV log but are no longer shown in the sidebar panel. Near-wellbore complexity
subtracts that same shared reference from the apparent ISIP
(`interpret.near_wellbore_complexity`), which closes the identity `Shmin + net pressure +
complexity = apparent ISIP` for all three methods. It is one value per test, not one per
method, set by `model._resolve_net_pressures` alongside the net pressures and guarded on the
apparent ISIP being present. A negative value is reported as-is -- no warning, no clamp --
since clamping would break the identity. C-C and C-D clear the contact pick, so neither gets
a compliance effective ISIP -- but once the closure pick is made the tangent effective ISIP
still exists, so the shared reference falls back to tangent and complexity IS reported for
both. `shmin_rapid` never feeds the shared reference, so for C-D the reported complexity is
referenced to the tangent effective ISIP and composes with `shmin_tangent`, not with
`shmin_rapid` -- there is no `net_pressure_rapid`, so for C-D the complexity participates in
no reported identity.

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

- `scipy` is pinned in `requirements.txt` and probed by `start-app.ps1` but is not currently
  imported anywhere in the package.
- Sample data folders, `Refs/`, and `.superpowers/` are gitignored. Design specs and plans
  from the build are under `docs/superpowers/`.
- The master log is CSV only; a parquet mirror alongside `dfit_log.csv` is a deferred
  extension point (`store.py`'s module docstring), not yet implemented.
- Folder-mode scanning is depth-1 only: nested subfolders (depth 2+) are never scanned.
- No concurrency control: folder mode assumes a single interpreter working a root at a time.
  Two people (or two windows) open on the same root last-write-wins on both the picks JSON and
  `dfit_log.csv` -- there is no lock file or merge.
- Accepted known limitations in the folder-mode Source switch/resume logic: if a test's saved
  picks name a source the resume logic silently falls back to applying those picks to whatever
  source actually loads, with no warning; a source switch (`ui._on_source_change`) only
  persists to disk on the next save (queue navigation, Finish, or Skip test), not
  immediately; and using the manual "Load picks…" file-dialog button while in folder mode
  loads that JSON into the workspace but does not re-sync the Source combobox to it (the
  Skip-test button does re-sync, via `_apply_loaded_state`'s `_goto` -> `refresh` ->
  `_update_stepbar` -> `_update_skip_test_btn` chain).

## TODO
- Apparent ISIP tool has 3 move methods: drag along line (with the vertical segment), pan, and rotate (activated when hovering over either end of the tool). Rotate seems to be gone/impossible because the tangent line is too long. Make it shorter and extend back to shut in time with dotted line
- Apparent ISIP tool needs to be a perfect tangent to pressure. Seems maybe it's taking an average slope over interval instead.
- Some tests suddenly drop off to zero pressure towards end, need to be able to trim tail, probably on G-function plot.
- 0 surface pressure means bottom hole can no longer be calc'd accurately because the head falls. After trimming tail, warn user if surface pressure ever falls below 100 psi.
- Would like to be able to resize both sidebars. Field names in right sidebar are getting cut off, even though there's lots of space.
- When no rate is auto-detected, the start/shut-in vlines never appear
- Some datasets don't have rate. Fallback in this case should simply set injection time by the true time between user-marked start and shut-in