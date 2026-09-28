# CLAUDE.md

Guidance for working in this repository.

## Overview

An interactive tool for interpreting a single diagnostic fracture injection test (DFIT)
by the compliance method. A Tkinter/ttk shell hosts an embedded matplotlib canvas. The
interpreter opens one data file (CSV or Fracpro `.DBS`), maps its channels, and walks eight
workflow steps, making draggable picks on each plot. Every reported number is derived from
one pure function, `model.compute_all`.

Interpretation is one well at a time. Two ways to get there: single-file mode (open one CSV
or `.DBS` directly, today's original flow) or folder mode ("Open Folder…"), which scans a
root folder into a queue of tests, shows that queue in a left sidebar, and rolls every test's
results up into a per-root `dfit_log.csv` master log. The sidebar and the log exist only in
folder mode; single-file mode is otherwise unchanged. There is still no cross-test
aggregation beyond that one log (no charts, no rollup stats). Permeability is out of scope.

The eight steps (`ui.py:STEPS`): overview → injection → isip → gfunction → tangent → loglog →
porepressure → stiffness. Overview shows the entire dataset, unclamped, and hosts the always-on
tail-trim line; its y-axis is labeled "pressure (psi)", and when BHP is converted from surface
pressure it also overlays the raw surface trace (`DerivedResults.p_surface_all`, set by
`compute_all`) as a thin red line on the same axis; Injection is the zoomed injection-window view with the draggable start/shut-in
lines and the te/Vinj/qmax title; stiffness is a semilog-y relative-stiffness-vs-effective-
pressure plot (URTeC-2019-123 A.8/A.9) that needs the pore-pressure estimate, so it comes last
and is skipped end to end under PC-F exactly like porepressure (`model.stiffness_skipped`).

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
  take arrays and pick parameters and return numbers. `units.py` is the unit-conversion leaf
  below both `io_load` and `questionnaire`: conversion constants, factor tables, and
  header/alias token lookups for pressure/rate/volume — no `dfit_tool` imports of its own.
- **Compute core (pure python/numpy).** `resample.py` does the 30-psi pressure-increment
  resampling, the diagnostic derivatives (dP/dG, G·dP/dG, d²P/dG², t·dP/dt), and the
  tail guard. `model.py` holds `PickState` (the serializable set of interpreter choices),
  `DerivedResults`, and `compute_all(state, td)`. `compute_all` is the single source of
  truth: it produces every reported value and every array the plots need.
- **IO.** `io_load.py` loads CSV and the reverse-engineered Fracpro `.DBS` binary format
  (`load()` dispatches on extension), parses datetimes including leaked Excel serials,
  suggests channel roles (the pressure pick is ranked, not first-match: surface/WHP-named beats
  treating-named beats generic beats demoted aux/pump/annulus/max/avg-type channels, though a
  BHP-named channel still wins outright; when given a loaded `TestData.column` callable, it also
  filters to "live" candidates first, since a dead/backside gauge or a locked-constant channel
  can otherwise outrank a real signal by name alone), converts surface pressure to BHP
  hydrostatically
  (`BHP = WHP + 0.052·mw·tvd`, valid post-shut-in where flow → 0), and detects/converts
  per-channel units via `units.py` (see Unit detection and conversion below). `questionnaire.py`
  parses a `*questionnaire*.xlsx` next to the data file for fluid density and TVD, also using
  `units.py` for meter/kg-m3 conversions.
- **Interaction (matplotlib only, no Tkinter).** `picks.py` has the event controllers
  (`DragLineController`, `AnchorLineController`, `DraggablePointController`,
  `SpanController`, `ModifierSpanController`, `HoverCursorController`, and the
  `_CaptureGate` press arbiter), the
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
    io_load, questionnaire → units

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
writes a PNG of all step plots, in their current zoom state, to a `<stem> DFIT plots/`
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
derived from `step_status` (all eight steps accounted for and none skipped is `"done"`, all
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
  the injection's rate `twinx`) owns `inaxes` over the shared region, so an identity check
  against a specific Axes never matches. Use `_axes_contains_pixel`/`_data_from_pixel`.
- **Slider `on_changed` callbacks must never call `refresh()`.** `refresh()` calls
  `fig.clf()`, which destroys the slider mid-drag. Callbacks only `set_xlim`/`set_ylim`,
  mutate the current `ViewState`, and `draw_idle()`. A full refresh happens only on step
  change, pick commit, Apply, scenario change, or Reset view. Hold slider references on
  `self` — matplotlib keeps no strong reference and GC otherwise kills the callbacks.
- **Diagnostic derivatives are reported positive-up for a declining pressure** (negated),
  matching how G-function and log-log plots are conventionally drawn.

## Testing

Tests live in `tests/` (~22 files). `tests/conftest.py` forces the Agg backend before
`matplotlib.pyplot` is imported, so the suite runs with no display or Tk. Synthetic data
comes from `tests/helpers.make_testdata` (a DFIT-shaped `TestData`) and
`helpers.injection_state`; tests drive controllers with real synthesized `MouseEvent`s.
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
- **Shmin, tangent** — BHP at the G·dP/dG through-origin departure (closure) point. The
  through-origin line's seed (`interpret.suggest_closure_tangent`, `picks.seed_tangent`)
  anchors at an interior local max of dP/dG (G >= `g_min=1.0`, relative prominence >=
  `CLOSURE_TANGENT_MIN_PROMINENCE`) with the largest G·dP/dG, rejecting both the early
  water-hammer spike (masked below `g_min`) and a small noise-driven local max further out on
  the curve (low prominence). Falls back to a through-origin least-squares fit over the first
  third of the `G >= g_min` samples when no candidate survives (unmasked when that segment has
  fewer than 2 finite, positive-`G` samples). Closure = the last sample within 5%
  (`CLOSURE_TANGENT_TOL_FRAC`) of the line before the first departure, walking forward from the
  tangent point -- a later re-crossing (e.g. a rising tail) is never picked up. The pick is a
  plain `DraggablePointController`: only its seeded starting position comes from this rule, so
  it stays fully draggable and a reload's saved pick is never overwritten.
- **Shmin, variable** — BHP at the G-time midpoint of the contact and closure picks. This
  third "variable-compliance" method is computed by `compute_all` and reported alongside the
  other two (Shmin, closure time, and net pressure each have compliance, tangent, and
  variable rows in the panel; effective ISIP shows only the compliance row there).
- **Shmin, Liberty** — Liberty's internal variant of the compliance method, minus 200 psi
  (`interpret.LIBERTY_OFFSET_PSI`). C-A (and blank) anchors on the min-dP/dG pick
  (`state.min_dpdg_G`); C-B anchors on the contact pick instead (the inflection) — the min pick
  there is only the inflection's seed and can sit at a nearby rel-min, not the inflection
  itself. Gated on `state.contact_G` being set, so it blanks in every state that blanks the
  compliance row (including a C-A/C-B whose contact construction failed); blank for C-C/C-D.
  Shown as the "Shmin Liberty" panel row and logged to the `Shmin_liberty` column only — it is
  not drawn on any plot and feeds no other derived value (no net pressure, no shared reference
  ISIP, no complexity).
- **Shmin, stiffness** — picked pressure − 75 psi (`interpret.shmin_compliance`), the same
  offset as Shmin compliance but a different construction: the pick sits on the 8th
  ("stiffness") step's relative-stiffness-vs-effective-pressure plot (URTeC-2019-123 A.8/A.9),
  at the upturn where the fracture walls come into contact (`interpret.h_function`/
  `relative_stiffness`/`suggest_stiffness_upturn_index`, auto-seeded then draggable). Needs the
  pore-pressure estimate (the h-function's Pres term), so it is skipped end to end under PC-F
  exactly like porepressure (`model.stiffness_skipped`). Comparison-only, like Shmin Liberty:
  shown as the "Shmin stiffness" panel row and logged to the `Shmin_stiffness`/
  `Shmin_stiffness_gradient` columns only — never drawn into net pressure, the shared reference
  ISIP, or complexity.
- **"No slope change apparent"** (`state.stiffness_no_upturn`, a side-panel checkbox visible
  only on the stiffness step) is the explicit negative finding: some tests show no abrupt
  slope change on the relative-stiffness plot, and this records that as a legitimate outcome
  rather than endorsing the auto-seeded line — same precedent as closure scenario C-C's "no
  contact → no Shmin". Checking it blanks `shmin_stiffness` (`model.compute_all` gates the
  value, not the arrays: `stiffness_p_eff`/`stiffness_S` still compute and the curve still
  renders) and suppresses the stale-pick warning for that pick (a suppressed pick reports
  nothing, so it can't be stale). The pick itself (`state.stiffness_pick_P`) is left alone —
  unchecking restores whatever was already picked, or, if none exists yet (the step's
  first-visit seeder already ran and won't re-fire), `ui._on_stiffness_no_upturn` calls
  `picks.seed_stiffness` on uncheck so the analyst isn't left with no line and no way to get
  one. `picks.seed_stiffness` itself no-ops when the flag is set, so a reloaded save with it
  doesn't get a line parked on next visit. `render_stiffness` draws no pick vline/marker while
  the flag is set and titles the plot "Relative stiffness — no slope change apparent (Shmin
  not reported)"; the curve stays in the step's PNG export either way. The checkbox itself
  never touches `step_status` — checking it and clicking Finish walks `step_status` normally,
  the same "visited this step, then advanced" path any other step takes, no pick required.
  Only `model.infer_step_status` (the legacy backfill for saves made before `step_status`
  existed) treats the flag alone as sufficient to mark the step `"done"` on reload, since a
  pre-existing save has no walk to replay. A checked box followed by navigating away *without*
  Finish therefore leaves the step merely `"visited"`, so the whole test reads `"in_progress"`
  in `store.status_for` — correct, since the analyst hasn't actually finished the test yet, and
  distinct from the per-step Skip button, which marks the whole test `"skipped"` regardless of
  how far it got. Logged to the tail-appended `stiffness_no_upturn` column (`store.LOG_COLUMNS`)
  as `True`/`False` — except once `model.stiffness_skipped(state)` is true (PC-F), when
  `store.build_log_row` logs blank instead of the stored flag: switching to PC-F after checking
  the box makes the step unreachable (so the box can never be unchecked again), and the flag is
  left alone in the picks JSON on purpose (switching back off PC-F revives it) rather than
  cleared, so the log row must blank it itself, matching every other stiffness output going
  blank under PC-F.
- **Near-wellbore complexity** — apparent ISIP − the shared reference effective ISIP. The
  near-wellbore friction and tortuosity that is in the early-decline extrapolation but has
  dissipated by the time the P-vs-G line is fit. Shown as the "NWB complexity" panel row and
  logged to the `near_wellbore_complexity` column.
- **Pore pressure** — intercept of the late-time postclosure line on the t^(−1/2) or t^(−1)
  axis chosen by the postclosure scenario.
- **Pressure gradients** — depth-normalized (psi/ft) form of a pressure, `value / state.tvd_ft`
  (`interpret.pressure_gradient`, `model._resolve_gradients`). Three rows in the panel (apparent
  ISIP grad, Shmin compliance grad, pore pressure grad, each inline under its parent) plus five
  more logged only to `dfit_log.csv` (Shmin variable/tangent/Liberty/rapid/stiffness gradient) —
  eight columns total. `tvd_ft` must be not-None, finite, and `> 0`, or every gradient blanks to
  `None`/`"-"`; nothing upstream enforces that (`io_load.bhp_inputs_ready` accepts `tvd_ft =
  0.0`), so this guard is the only thing standing between the feature and a divide-by-zero.
  Every gradient field is strictly its own source value / TVD — no cross-field fallback in the
  model layer, so `Shmin_compliance_gradient` stays blank under C-D exactly as `Shmin_compliance`
  does, and `Shmin_rapid_gradient` is what carries that case in the CSV. The panel's rapid
  substitution (below) is a `ui.py` display choice on top of that, not a property of these
  fields. An unusable `tvd_ft` (missing, non-numeric, zero, negative, or non-finite) is never
  silent once there's something to report: `_resolve_gradients` appends a warning naming the
  problem ("TVD not set..." or "TVD `<value>` is not a positive number..."), gated on at least
  one of the eight source values being non-None so a freshly-opened test with no picks yet
  stays quiet. It deliberately coexists with the separate "Surface pressure selected but
  density/TVD not set" warning (`compute_all`) rather than being suppressed by it — that one is
  about BHP reliability, this one is about the gradients not being reported, and both land in
  the same collapsed warnings count (see the Tail trim section below for the display itself).

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
smoothing. A tail guard flags a default cutoff once the pressure sustains a rise above its
running minimum (non-monotonic late data) for long enough, and in enough samples, to rule out
noise -- see Tail trim below for the exact thresholds and how an explicit override can move
the cutoff past it.

**No-rate fallback.** When a dataset has no rate channel (or a dead one that never exceeds
the detection threshold), `picks.seed_injection` seeds the start/shut-in vlines from the
pressure shape instead (`interpret.suggest_injection_window_pressure`: shut-in at the
pressure max, start at the last upcross of a 10%-of-rise threshold, non-finite samples
ignored so dropouts can't fake an upcross, positional defaults for degenerate shapes), so the
draggable lines always exist. `picks.seed_overview` delegates to the same seeder on the
Overview step's first visit, so the reference lines exist there too. `compute_all` sets
`t_shutin_s` from the picks alone and, when the effective te (Vinj/qmax) is unavailable,
falls back to te = wall-clock pump duration (shut-in − start) with an appended warning. Vinj
and qmax stay blank without rate; the Injection title shows only the pieces that exist. When a
rate channel does exist, the rate-based seed (`interpret.suggest_injection_window`) splits the
rate-on samples into contiguous runs, sizes them all by volume gain (or all by summed rate
when there is no volume channel or any run's gain is non-finite or non-positive), drops any run smaller than
`INJECTION_MIN_RUN_FRAC` (10%) of the largest run's size, and returns the last surviving run
(start = its first sample, shut-in = one past its last). This skips both an earlier breakdown
pulse or step-rate cycle and a trailing post-shut-in rate blip, landing on the main, sustained
injection.

**Tail trim.** The tail guard only catches a late rise, and now only a *sustained* one: it
fires when a run of samples stays continuously more than a fixed 30 psi
(`resample.RISE_GUARD_PSI`, independent of the resample step) above the running minimum for
both >= 60 s and >= 5 samples (`resample.RISE_GUARD_SUSTAIN_S`/`RISE_GUARD_SUSTAIN_SAMPLES` --
the sample-count floor matters at coarse (>= 60 s) sample spacing, where duration alone would
already be satisfied by the run's second sample), so a brief water-hammer rebound or noise
spike no longer trips it. A non-finite sample mid-run resets the run rather than being skipped
through it -- continuity can't be confirmed across a dropout, so two excursions separated by
missing data can't bridge into a false fire. `guard_dt`/`guarded_at` are pinned to the FIRST run
that satisfies both sustain conditions -- a second, later qualifying excursion further out in the
same record never overwrites them, regardless of `stop_at_guard` (below). When the guard does
fire, the excluded raw tail is drawn as a faint gray preview capped at 2x the G-range that was
actually kept BEFORE the guard fired (the last `resampled_full` sample strictly before `guard_dt`,
not the whole record -- `stop_at_guard=False`, below, can extend real kept samples well past
`guard_dt`, which would otherwise loosen this cap considerably) on the G-function plot alongside a
warning inserted at the front of `DerivedResults.warnings` (not appended), so it's the first line
if the analyst expands the list -- `ui.py`'s `warn_lbl` (under the Notes box) collapses the whole
warnings list to a one-line count by default (`"N warnings (click to expand)"`,
`ui.format_warnings_text`), expanding in place on click (`DfitApp._toggle_warnings`) and
re-collapsing on the next recompute -- rather than getting buried below an earlier-queued warning
once expanded. A
monotone crash to ~0 psi (gauge pulled, well opened) still sails through the guard untouched and
pollutes the derivatives -- the Overview step's tail-trim line is the defense against that.

The trim line is always on: there is no more "Show trim tool" toggle. `picks.seed_tail_trim`
(called explicitly from `ui._seed_step`, not a `SEEDERS` entry -- it needs a `res` recomputed
*after* the injection window is seeded) parks-and-applies a default cut on the Overview step's
first visit, via `interpret.suggest_tail_trim_dt(dt_post, p_surface_post, guard_dt)`: the
earliest of the rise-guard boundary (`res.resampled_full.guard_dt`) or the first post-shut-in
sample where surface pressure drops below 100 psi (`interpret.MIN_SURFACE_PRESSURE_PSI`; skipped
when the mapped channel is already BHP, where a sub-100-psi test is meaningless), tie going to
the rise guard. A rise-guard boundary sets **no pick at all** (`PickState.tail_trim_dt` stays
`None`) -- the guard boundary is already the effective cutoff by default
(`interpret.resolve_tail_cut_dt`, below, with no trim in state), so only the rendered line
position and gray-out need to reflect it, not a stored trim; a sub-100-psi crash
does snap to the last full-resample sample **strictly before** the cut (`searchsorted` with
`side="left"`, not `"right"`, and not `picks._nearest`) and sets `PickState.tail_trim_reason =
"low_pressure"` (logged to `tail_trim_reason`, appended after `tail_trim_s` in `LOG_COLUMNS`).
Strictly-before matters and is not a rounding nicety: the resampler keeps a point at every >=30 psi
drop, so the crash cliff is almost always kept *and is usually the last point kept* (the flat ~0
psi tail after it never drops another 30 psi), so an "at or before" snap would land on
`dt_full[-1]`, trip the seeder's own past-the-end bail, and make the auto-trim a near-no-op on the
ordinary crashed record. The seeder BAILS (sets no pick) rather than clamps whenever
`idx = searchsorted(dt_full, cut_dt, side="left") - 1` falls outside `[2, len(dt_full) - 2]`:
`idx < 2` means fewer than 3 resampled samples precede the crash, so no cut can both keep >=3
points and exclude it -- clamping `idx` UP to 2 (the old behavior) could set a trim at
`dt_full[2]` even when that sample sits at or past the crash, keeping a sub-floor sample inside
a record the "Tail auto-trimmed" message claims is clean; `idx >= len(dt_full) - 1` means the
cut sits past the last kept point (crash beyond where resampling reached), so there's nothing in
the record to remove. Both bail cases are still covered by the separate low-surface-pressure
warning (which scans raw, not kept, samples), so neither goes silent -- setting no trim is
strictly better than setting a wrong one. Neither candidate existing (clean record) leaves the
line parked at the end of the data, nothing trimmed. The seeder is non-destructive (a pre-existing
trim, e.g. from a reloaded save, is left alone).

Dragging the line (`DragLineController`, gid `"tail_trim"`, in time-domain hours, its own private
gate since nothing else is draggable on that axes) commits `PickState.tail_trim_dt`
(shut-in-relative seconds, `None` = no trim) via `picks.commit_tail_trim`, which always resets
`tail_trim_reason` to `""` -- a manual drag or clear overrides whatever auto-attribution put the
trim where it was. `compute_all` resamples the full post-shut-in record via
`resample.resample_pressure_increment(..., stop_at_guard=False)`, so `DerivedResults.resampled_full`/
`G_full` always span the whole record regardless of whether the rise guard fired -- `stop_at_guard`
(default `True`, preserving the historic hard-stop for `tests/test_rise_guard.py`'s direct-call
tests and any other caller that still wants the old "guard ends the record" behavior) still
detects `guard_dt`/`guarded_at` from the FIRST run that satisfies both sustain conditions, pinned
there for good (see above) regardless of which mode is active. What `False` changes is what
happens at the moment of firing: instead of breaking, the run simply resets (exactly like an
ordinary run that dips back below tolerance on its own) and resampling continues, still relative
to the ORIGINAL running minimum -- never reset, never frozen at a fresh anchor. This needs no
special re-anchoring logic, because a rising excursion never satisfies the ordinary keep rule
(`pi <= last_kept - step`) in the first place: rejecting it as part of a guard run and rejecting
it via the ordinary rule are the same outcome. If a genuine further decline resumes later
(relative to the TRUE historical minimum), it gets picked up completely normally, with no
monotonicity issue -- every kept point, before and after the guard, is still by construction a
genuine new low. If the rise never comes back down at all (a stuck sensor or an ongoing leak, the
common real shape), `resampled_full` correctly keeps ZERO additional points past `guard_dt` --
mathematically honest, since there is no new information to resample there, not a bug to paper
over with a fake anchor point. (An earlier version of this feature tried exactly that -- appending
the confirming sample as a fresh anchor and resetting `running_min`/`last_kept` to it -- but that
both failed to fix the "never comes back" case in the one place that mattered, the anchor still
sat at `dt_full[-1]` in the UI's drag-commit closure, and broke the resampler's monotonicity
invariant for every OTHER caller by admitting a point that was, by definition, above the running
minimum. It was reverted.) `compute_all` then masks to the cutoff returned by
`interpret.resolve_tail_cut_dt(state.tail_trim_dt, guard_dt, state.tail_guard_override)` before
computing diagnostics (see below) -- so the trim propagates to every downstream value (effective
ISIP, Shmin, log-log, pore pressure) with no other plumbing. Whenever `tail_trim_dt` is set, `compute_all` also emits an
explanatory warning (`insert(0)`, same front-of-stack treatment as the guard warning) so an
auto-applied trim is never silent: for `"low_pressure"`, `"Tail auto-trimmed ... surface pressure
crashes below 100 psi shortly after this point. Drag the Overview trim line to the right edge to
undo."` -- worded to point at the crash beginning just past the cut, not at the cut itself, since
by construction of the `side="left"` snap the trim dt is a sample where pressure is still ABOVE
the floor -- with an appended clause, `" Doing so also overrides the tail guard, which fires
further into the tail, extending the cutoff to the end of the record."`, whenever
`resampled_full.guard_dt` is also set for this record (a guard existing at all here necessarily
fires later than this low-pressure cut, per `suggest_tail_trim_dt`'s earliest-wins rule), so
"drag to the right edge" doesn't read as a plain undo when it also overrides a fired guard; else
`"Tail trimmed ... (N raw samples excluded)"` for a manual trim (see the override-related
exception to that message in "Overriding the guard" below). This matters
because the seeded trim usually *clears* the separate "Surface pressure fell below 100 psi"
warning below (its mask is narrowed by `tail_trim_dt`) -- without this line an auto-applied trim
would otherwise report nothing having changed.

**Resyncing shut-in moves (`picks.resync_auto_tail_trim`).** `tail_trim_dt` is shut-in-relative,
but the shut-in pick can move on the Injection step after the trim was seeded on Overview.
Dragging shut-in later by delta moves the auto trim's absolute-time cut delta later too (nothing
else re-derives it), quietly re-admitting whatever crash it was supposed to exclude while the
"Tail auto-trimmed" warning keeps claiming the crash is handled. `ui.py`'s Injection controller
wiring calls `resync_auto_tail_trim(state, td, res)` after committing a shut-in drag (never a
start-only drag: `tail_trim_dt`/`guard_dt` both live in dt-from-shut-in space, and
`resample_pressure_increment` takes only `(dt, p)` built from shut-in onward, so a start-only
change can't move either). The resync is a no-op unless `tail_trim_reason == "low_pressure"` --
a `""` reason means no trim or the analyst's own manual pick, neither of which resync ever
touches -- otherwise it clears the stale pick and re-runs `seed_tail_trim` against the new window,
which correctly sets no trim at all if the new window turns out to have no crash.

**Overriding the guard (`PickState.tail_guard_override`, `interpret.resolve_tail_cut_dt`).**
`resampled_full` spans the whole record (`stop_at_guard=False`, above) whenever a genuine further
decline resumes after the guard fires, but on the shape that actually matters most -- a tail that
never comes back down at all -- it correctly keeps nothing new past `guard_dt`, so there may be no
resampled sample there to snap to at all. The override therefore lives in `ui.py`'s Overview
drag-commit closure (`_attach_controllers`), not in the resampler: when the drag target is at or
before `guard_dt` (or there's no guard), it snaps to the nearest full-resample sample among only
those at/before `guard_dt` (`picks._nearest` against `DerivedResults.resampled_full.dt` masked to
`<= guard_dt` when a guard exists) -- restricted, not the bare unmasked array, because
`resampled_full` can now carry points PAST `guard_dt` too (`stop_at_guard=False`), and an
unmasked nearest search can pick one of those whenever it happens to sit numerically closer to
the drag target than any pre-guard sample (a sparse pre-guard decline next to a denser post-guard
one), silently setting `tail_guard_override` even though the analyst never dragged past the
guard. Masking first reproduces exactly the kept-point set `stop_at_guard=True` would have
produced, so this branch is provably equivalent to the tool's pre-override behavior. When the drag
target is PAST `guard_dt`, it instead snaps against the RAW post-shut-in samples
(`self.td.t_s` from shut-in onward) -- resolvable unconditionally regardless of whether the
resampler found any new points past the guard, since a raw sample always exists to snap to even
when `resampled_full` doesn't. That said, "resolvable" is not "meaningful": the override always
changes the DISPLAY (the gray-out boundary, and which of the two guard-related warnings shows,
below) but only changes the actual `res.resampled`/`res.diagnostics` numbers -- and therefore
Shmin/effective ISIP/pore pressure -- when the resampler genuinely had something new to offer
past the guard, i.e. the tail later resumes a decline below its pre-guard historical minimum. On
a permanently-elevated or still-rising tail (the "never comes back" shape below), dragging past
the guard moves the line and the gray-out but leaves every reported number exactly as the
guard-clamped default, and `compute_all` says so explicitly (see the warning-suppression
paragraph further down). Critically, the raw-snap branch never clears the trim to `None`, even when the drag lands
on the raw record's very last sample -- it always sets an explicit `tail_trim_dt` there instead
(functionally "include everything," since nothing exists past it to exclude), because clearing to
`None` would fall back through `resolve_tail_cut_dt` to `guard_dt` again, defeating the whole
point of the override. This is the actual bug fix: previously (an earlier, reverted version of
this feature) a drag past a permanently-elevated guard had nothing to snap to, clamped to the last
resampled index, hit the "released at/past the end -> clear the trim" branch, and the line snapped
right back to `guard_dt` on the next render no matter where the analyst dropped it.

A set `tail_trim_dt` sitting past `guard_dt` is still ambiguous on its own -- it's either that
deliberate drag, or a stale auto-trim left behind by a shut-in move before
`resync_auto_tail_trim` ran (above). `picks.commit_tail_trim` takes an optional `guard_dt` param
(one existing call site omits it -- `seed_tail_trim`'s own commit, since a seeded trim is never
past the guard by construction; `ui.py`'s Overview drag handler is the other call site, and it
always passes `res.resampled_full.guard_dt`) and sets `state.tail_guard_override = tail_trim_dt
is not None and guard_dt is not None and tail_trim_dt > guard_dt` every time it runs, so a later
commit that no longer lands past the guard clears a stale `True` back to `False`.
`interpret.resolve_tail_cut_dt(tail_trim_dt, guard_dt, override)` is the single place both
`compute_all`'s masking and `render_overview`'s display cut resolve this: no trim set ->
`guard_dt` (or `None`); a trim at/before the guard (or no guard at all) -> the trim, unchanged; a
trim past the guard -> the trim, unchanged, only when `override` is `True`, else clamped back to
`guard_dt`. `store.LOG_COLUMNS`/`build_log_row` log the flag too, as `tail_guard_override`,
tail-appended at the very end of `LOG_COLUMNS` (not adjacent to `tail_trim_s`/`tail_trim_reason`,
which sit earlier in the column order, per the append-only convention).
`compute_all` resolves this cutoff once and reuses it for the masking above and for one more
thing: the "Tail guard stopped resampling ... later data excluded" warning fires in its original
form only while the guard is still actually binding, i.e. the resolved cutoff is `<= guard_dt`
(the no-override default, and a stale trim that got clamped back). Once an override lets the
trim stick past `guard_dt`, `compute_all` checks whether that override actually admitted new
resampled points there (`np.any((rs_full.dt > guard_dt) & (rs_full.dt <= cutoff))`) before
deciding what to say instead of this warning -- it is never just silently dropped:
  - **Data genuinely admitted** (a real further decline resumes past the guard): the original
    warning is suppressed outright. The separate "Tail trimmed ... " warning (from the
    `tail_trim_dt is not None` block below) already reports the real, later cutoff, and showing
    both at once would contradict each other.
  - **Nothing admitted** (the "never comes back" shape -- `resampled_full` keeps zero points past
    `guard_dt` no matter how far past it the override reaches): the original warning is replaced
    with a distinct, honest one -- `"Tail-guard override requested past N min, but the resampler
    found no further usable data there (the tail never resumes a decline) -- Shmin/effective
    ISIP/pore pressure still reflect the guard's original cutoff at M min."` The ordinary "Tail
    trimmed ... (0 raw samples excluded)" message is also suppressed in this specific case (it
    would otherwise sit right next to the honest warning and read as a confusing, near-
    contradictory pair claiming both "nothing changed" and "0 excluded" about the same cut) --
    everything the plain message would have said is already covered by the honest one.

`plots.render_overview`'s `show_trim` kwarg is now `interactive` (default `True`, meaning "this is
the live canvas, not an export" rather than "the analyst toggled the tool on"); `render_step_figure`
passes `interactive=False` for `"overview"` (mirroring the `step_key == "gfunction"` special
cases), so an exported PNG never carries a line the analyst can't actually drag. The effective
display cut is `interpret.resolve_tail_cut_dt(state.tail_trim_dt, res.resampled_full.guard_dt,
state.tail_guard_override)` -- both `tail_trim_dt` and `guard_dt` live in shut-in-relative dt, so a
stale trim left behind by a shut-in move (before a resync runs, or for any other caller that builds
`res` without going through the Injection wiring) can sit PAST the guard; taking `tail_trim_dt`
alone (the old "a set pick is always <= guard_dt, no clamp needed" reasoning) would then render the
guard-excluded region as kept, the opposite of this feature's purpose, so the clamp-back-to-`guard_dt`
is required whenever `tail_guard_override` isn't set -- and skipped, on purpose, when it is, so a
deliberate drag past the guard actually sticks. The
Overview renderer grays out the raw trace past that cut (`gid="tail_excluded"`) whenever it's not
`None`, which is what makes a **guard-excluded** tail finally visible on Overview too --
previously only the G-function plot's own `guard_excluded` preview showed it there; the
G-function plot itself
still carries no trim artifacts at all. The draggable vline itself (`gid="tail_trim"`, drawn only
when `interactive`) sits at that same cut when set, else the last raw sample time. Releasing a
drag at/past the last point clears the trim, and the commit clamps to >=3 kept points -- a record
whose full resample already yields <=3 points can therefore only ever clear the trim, never set
one (a pathological saved trim hits a warning directing the analyst back to the Overview tab's
trim line, instead of a dead plot). The trim is never touched by a scenario change or by either
min-dP/dG correction (the triangle drag, the Shift+drag window) -- this replaces the old
"manual-only (no seeder) ... never auto-set" rule, which no longer holds.

`render_overview`'s `ViewDefaults.ylim` is pinned to `(0.0, p_hi + pad)` unconditionally --
including a converted-BHP record, where it squashes the trace into the top of the axes; the
y-slider and Reset view are the escape. Because that can reach below the Axes' own autoscaled
extent, both `ui.refresh` and `plots.render_step_figure` union a renderer's `ViewDefaults.ylim`
into `full_y` (the slider's outer range) for every step except gfunction, which still *replaces*
`full_y` (it must keep shielding the y-slider from the effective-ISIP tangent's dashed extension,
which can swing the Axes' own autoscale to extreme psi) -- otherwise `_make_range_slider`'s
`valinit` clamping would silently pull the view back up into the autoscaled extent on the first
slider touch, losing the 0 baseline.

Warnings: WHP below 100 psi (`interpret.MIN_SURFACE_PRESSURE_PSI`) anywhere the resampler
actually consumed post-shut-in data -- up to the rise guard's own cutoff
(`resample.Resampled.guard_dt`), unless an override is actively extending past it (see
"Overriding the guard" below, which this scan's window also respects), further narrowed by a
trim if one is set, deliberately *not*
bounded by the last resampled point kept (which can sit up to one `resample_step` above the true
minimum and so miss a crash just past it) -- flags BHP as unreliable there (only when the mapped
channel is surface pressure). The `guard_dt` bound is itself skipped when an active
`tail_guard_override` has actually extended the effective cutoff past it (reusing the same
`cutoff > guard_dt` check `compute_all` already made for the guard-warning logic above, not
re-derived) -- otherwise a sub-100-psi crash the override genuinely admits into the diagnostics
would never get its own warning just because it sits past `guard_dt`; the `tail_trim_dt` bound
still narrows the scan by the explicit trim either way. A stale-pick warning (gated on a trim actually being set) covers two
distinct failure modes: the G-function picks (contact, min-dP/dG, closure) left beyond the trim
would otherwise silently interp-clamp to the trimmed edge, while a pore-pressure window affected
by the trim gets the same warning by outcome -- a finite upper bound beyond the trimmed edge just
shrinks its fit (still returns a value), and a window with fewer than 2 surviving samples empties
it and blanks `pore_pressure` outright (an open-ended upper bound shrinks benignly with the trim
and is exempt from the shrunk check) -- so neither failure mode goes silent.

**Pressure dropouts.** A brief gauge dropout -- pressure reads near zero for a few seconds, then
returns -- otherwise poisons everything downstream of it: it pins `resample_pressure_increment`'s
`running_min` at the dip (nothing later is ever `RISE_GUARD_PSI` below that, so resampling stops
dead there) and the recovery itself fires the rise guard. `resample.detect_dropouts(dt, p)` finds
and masks these automatically, no per-test toggle, on the raw mapped pressure channel (before any
hydrostatic offset, so "near zero" means near zero on the gauge, not on a converted BHP) --
`model.compute_all` runs it once `res.t_shutin_s` is known, before the resample block, and stores
the result on `DerivedResults.dropout_mask` (bool, full length over `td` samples, `False` before
shut-in) and `DerivedResults.dropouts` (a `resample.Dropout(dt_start, dt_end, n_samples, p_min)`
per event, describing the dip run itself, not any lead-in extension). Every consumer of raw
post-shut-in pressure applies the mask by treating a masked sample as missing (NaN) --
`res.bhp_all` itself is never mutated. Onset is a sample falling to <=
`resample.DROPOUT_FLOOR_FRAC` (10%) of the most recent finite, unmasked level, gated on that level
being above `resample.DROPOUT_MIN_REF_PSI` (100 psi) -- without that gate, noise on an
already-dead channel re-triggers onset endlessly. The dip run is masked only if it recovers
(returns to within `RISE_GUARD_PSI` of the pre-dip level) before the record ends and the recovery
is "momentary": not sustained by the rise guard's own definition (`>= RISE_GUARD_SUSTAIN_S` AND
`>= RISE_GUARD_SUSTAIN_SAMPLES`) and under `resample.DROPOUT_MAX_S` (10 min) total duration -- a
crash that never recovers is the low-pressure tail trim's job, not this detector's, and a
genuinely sustained dip (a dead channel, a real quiet period) is left alone. A lead-in extension
(a slide toward zero just before onset, seen on the motivating Encore record) is walked backward
from onset and masked too, but only when the walk stops on its own -- within `RISE_GUARD_SUSTAIN_S`
of onset, at a level within `resample.DROPOUT_LEADIN_MATCH_PSI` (150 psi) of the recovery level --
so a real step change that happens to straddle a one-sample glitch (a long plateau at a level
nowhere near the recovery) masks only the glitch, not the plateau. `detect_dropouts` is inherently
sequential state (not vectorizable outright), but it spends almost all of its length outside any
dip even on a record that DOES have one, so `resample._dropout_candidates(p, floor_frac, min_ref)`
computes every possible onset position -- and the `ref` each one is checked against -- in one
vectorized pass over consecutive pairs in the finite-only subsequence of `p`, and
`resample._dropout_scan_candidates` jumps directly between them instead of stepping sample by
sample there. This is exact, not approximate: outside an active dip (the only place onset is ever
checked -- a dip's own interior samples are compared only against the frozen pre-dip `ref`, never
re-checked for a fresh onset), `ref` is always exactly the immediately preceding finite sample --
including right after a dip resolves, where it resets to the recovered sample, which is already
that relationship in the same global computation -- so every position the scan would ever need to
examine is one of these candidates, with the correct `ref` already attached. A candidate that
happens to sit inside a dip a still-earlier candidate already resolved is simply never reached: the
scan skips straight past the whole resolved range (`np.searchsorted` against the sorted candidate
list). Only the in-dip recovery scan and the lead-in walk still step sample by sample -- both are
bounded by the event (a handful to a few hundred samples), not by the whole record. On Encore's
1.2M post-shut-in samples this drops `detect_dropouts` from ~0.7 s (the original per-sample loop,
which used to run in full even once a dropout was found) to roughly the vectorized pre-check's own
cost. The original per-sample loop (`resample._dropout_scan_loop`) is kept as a reference
implementation, called only by a test that fuzzes it against the candidate-jump path over
thousands of randomized series. Consumers: the resample block
masks `res.bhp_all` to NaN before calling `resample_pressure_increment` (already handles NaN --
skipped, resets any rise run -- no resampler change needed) and reuses that same masked array for
the guard-excluded preview; the low-surface-pressure scan excludes masked samples from its window
(`m &= ~res.dropout_mask`) so a masked glitch never fires "Surface pressure fell below 100 psi";
and `picks.seed_tail_trim` masks `p_surface_post` to NaN before calling
`interpret.suggest_tail_trim_dt` (which already ignores non-finite samples) so a dropout never
triggers the `"low_pressure"` auto-trim. A firing detector is never silent: `compute_all`
`insert(0)`s one warning line -- on the motivating Encore record, `"Pressure dropout masked at 11
min after shut-in (23 samples, 22 s, to 12 psi) -- treated as a gauge glitch"` -- or, for several
events, `"3 pressure dropouts masked (first at 11 min after shut-in) -- treated as gauge
glitches"` -- silent when there are none. Both the minute and (single-event) duration figures are
gated on their FORMATTED (rounded) value, not the raw number, so what's checked always matches
what's printed: `"at 0 min"` reads as "no time elapsed at all" and is misleading for anything in
the first ~30 s, so it becomes `"at <1 min"` instead when `f"{dt_start/60:.0f}"` rounds to `"0"`;
likewise the single-event duration clause is dropped entirely (not printed as the useless `"0
s"`) when `f"{duration:.0f}"` rounds to `"0"` -- true for an exact-zero single-sample dip
(`dt_end == dt_start`) and for a sub-0.5 s multi-sample one on a sub-1-Hz channel alike. The
single-event line also grammar-checks the common case: `"1 sample"` not `"1 samples"` for a
single-sample glitch -- e.g. `"Pressure dropout masked at <1 min after shut-in (1 sample, to 15
psi) -- treated as a gauge glitch"`.
`plots._split_dropouts(p, mask)` splits a trace into `(p_clean, p_masked)` (NaN in the
other array); `render_overview`/`render_isip` plot `p_clean` as the main trace (so the line breaks
across a masked gap instead of spiking to it) and `p_masked`'s finite samples as small magenta
markers (`gid="dropout_masked"`, label "masked dropout") -- undecimated, since the handful of
masked samples in a 10^5+-point record would almost certainly fall between the main trace's
decimation stride and never get drawn. The markers are plotted with `scalex=False, scaley=False`
and the Axes' `dataLim` is snapshotted before and restored after, so a masked sample's real value
(a -9999 psi sentinel, or Encore's ~12 psi) never pulls the autoscaled view -- or anything else
that reads `ax.get_ylim()`/`dataLim` afterward, e.g. the ISIP step's default view or the Overview
y-slider's full range -- down to it; the marker is still a real plotted artist at its true (and
possibly off-screen) position, just excluded from the bounding box. Exported PNGs
(`render_step_figure`) get this automatically, since it isn't gated on `interactive`.

**G-function.** α = 1 (low-leakoff) is the default; α = 0.5 only if a test exceeds ~1 md.

The dP/dG (twin) axis has two independent limits. The **default view** autoscales to the data:
the max of dP/dG over `G >= plots.Y2_SCALE_G_MIN` (1.0, the same g_min convention
`interpret.suggest_min_dpdg_index` and the d2P/dG2 block use), +10%, floored at 1.0, capped at
`plots.DPDG_VIEW_MAX`. Masking by G is what keeps the early water-hammer spike out of it -- a
percentile over all samples was dominated by that spike, since the resampled grid is densest
across it, and the old hard 50-psi/G cap squashed any record whose real derivative ran higher.
A record entirely below G=1 falls back to all finite samples. The **slider's full range** is the
twin Axes' own autoscale unioned with that default, then hard-clamped into
`(0, plots.DPDG_VIEW_MAX)` -- both steps in `ui.refresh` and again in
`plots.render_step_figure`, kept in textual lockstep. The clamp is what stops the slider
traveling past 500 no matter how extreme the raw spike is. The union is what keeps the default
view inside the travel: the raw autoscale is inflated at the top by the near-G=0 spike and
lifted off zero at the bottom by a nonzero data minimum, so the default can fall outside it in
either direction, and `_make_range_slider`'s valinit pinning would then snap the view off the
default on the first slider touch -- the same failure the `full_y` union prevents on every other
step. Note the full range is an *intersection* with `(0, 500)`, not that interval itself.
`render_tangent`'s own G·dP/dG default is still the 95th-percentile rule and has no cap.

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
range; `interpret.RAPID_CLOSURE_OFFSET_PSI`), shown in the G-function title
(`interpret.format_shmin_rapid`, verbose form). Net pressure is deliberately not
derived from it -- there is no `net_pressure_rapid`.

**Panel asterisks.** Three result-panel rows carry a trailing `*` on the label when the number in
them came from a fallback rather than the primary construction. All three are display-only, set in
`ui._update_panel` by mutating `self.name_lbls[...]`. The actual rule is that the gate matches
whatever makes the value column render a real number instead of `"-"`, so an asterisk can never
sit next to a `"-"`: two of the three rows blank only on the same `not_visited` check the value
column uses, but `"Shmin compliance grad*"` has a second blanking path the value column also
respects (see below) and so needs a second clause in its gate. All reset to plain text on the
else branch, since the label widgets persist across refreshes. None is explained in the panel
itself.

- `"Shmin compliance*"` -- `shmin_rapid` has no panel row of its own; it stands in for the
  compliance Shmin in that row when `use_rapid = shmin_compliance is None and shmin_rapid is not
  None`. The value is the short `format_shmin_rapid` form, `"9325 ±75"` -- no tilde, the ±
  half-range is what signals the approximation in the value column. The two are never both set
  (C-D clears the contact, and `compute_all` only sets `shmin_rapid` for C-D), so the fallback is
  unambiguous. The G-function title separately carries the verbose
  `Shmin(rapid)=9325 ±75 (ISIP − 100–250)` whenever this is showing.
- `"Shmin compliance grad*"` -- shadows the `"Shmin compliance*"` row directly above it, same
  `use_rapid` gate, selecting `shmin_rapid_gradient` instead of `shmin_compliance_gradient` for
  the value. Unlike the parent row there is no `±75` half-range in a gradient value (not worth
  rendering), so the label asterisk is the only marker here. `shmin_rapid_gradient` itself gets
  no panel row of its own, mirroring `shmin_rapid`. This row has a second blanking path the
  parent row does not: `model._resolve_gradients`'s `tvd_ft > 0` guard, which blanks every
  gradient (including `shmin_rapid_gradient`) regardless of `use_rapid` -- reachable on a
  C-D-rapid test with no TVD entered (e.g. a `pressure_is_bhp` downhole-gauge record, where TVD
  is never needed). So this row's gate adds a third clause, `r.shmin_rapid_gradient is not
  None`, on top of `use_rapid and gf_visited` -- without it the label would show the asterisk
  over a `"-"` value in that state.
- `"NWB complexity*"` -- complexity is apparent ISIP minus the *shared reference* effective ISIP,
  and this marks the case where that reference fell back to the tangent effective ISIP:
  `use_tangent_ref = near_wellbore_complexity is not None and net_pressure_isip_source ==
  "tangent"`. The reachable condition is "no compliance effective ISIP but a tangent one exists",
  which C-C and C-D both produce (they clear the contact, so there is nothing to reference), but
  they are not the only way there -- e.g. selecting C-C and then C-A, where
  `re_derive_contact_from_min` finds nothing at >=110% of the min and leaves `contact_G` `None`,
  shows the mark under a C-A scenario. It is truthful in every such state; the gate is the
  reference, not the scenario. Note the two asterisks do not mean the same thing -- this one says
  "referenced to the fallback", not "approximate". Under C-D both show at once **once the closure
  pick exists**: without it there is no tangent effective ISIP either, so complexity is `None` and
  only the Shmin asterisk shows.

The CSV log is unaffected by either: `store.LOG_COLUMNS` keeps its own `Shmin_rapid` column, and
`net_pressure_isip_source` is already logged.

**Min-dP/dG pick.** The red triangle seeds from `interpret.suggest_min_dpdg_index`: interior
local minima of dP/dG over the whole record (no fixed G threshold), preferring candidates
before the fracture-contact hump. `interpret.suggest_hump_index` locates that hump as the
interior local max of dP/dG with the largest G·dP/dG -- the water-hammer spike near G=0 loses
on the G factor, the decaying tail has no local max -- falling back to the raw argmax on a
monotone record; the same helper seeds the blank-scenario contact placeholder. A record with
no interior local min at all (C-C shape) falls back to the G>=1-masked global min. Two
corrections exist: dragging the triangle commits on release and re-derives the contact via
`re_derive_contact_from_min` (C-A: +10% rule; C-B: g_min-masked inflection nearest the drag);
Shift+drag selects a window instead (`picks.ModifierSpanController`, sharing the step's
capture gate and registered ahead of the point controllers so a Shift-press wins the gesture).
The window commit, `picks.handle_min_dpdg_window`, sets both picks itself: C-A puts the
triangle at the window's dP/dG min and the contact by the +10% rule from there; C-B puts the
triangle *and* the contact at the inflection found inside the window
(`suggest_contact_inflection_index(g_range=...)`, no g_min mask, so a below-G=1 inflection
stays where the analyst put the window). The span is a live gesture only -- never persisted,
never rendered in exports. When C-A is active and dP/dG never reaches 110% of the picked min
(no clear contact), `render_gfunction` draws a dashed hline at that threshold on the dP/dG
axis (`gid="clear_threshold_line"`, skipped when the threshold is non-finite) with a
"consider C-B" label; it appears in exported PNGs whenever the condition holds.

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
(the other steps keep their `RENDERERS`-order numbering).

**Stiffness skip.** `model.stiffness_skipped(state)` mirrors `porepressure_skipped(state)`
exactly (it is the same PC-F check): the relative-stiffness plot's h-function needs a
pore-pressure estimate as its `Pres` term, and PC-F never yields one. Every place the
pore-pressure skip is consulted has an independent stiffness counterpart -- `_goto` redirects a
`"stiffness"` destination to `"loglog"` the same way it redirects `"porepressure"`;
`_update_stepbar` force-disables the stiffness breadcrumb under the same condition;
`plots.save_all_step_pngs` also omits `"8_stiffness.png"`; and `store.status_for`'s
`_accounted`/`_user_skipped` PC-F carve-outs are OR'd with `stiffness_skipped` alongside the
existing `porepressure_skipped` clause, so a PC-F test with neither step's `step_status` entry
still reports `"done"`. `ui._last_step()` needs no separate stiffness case: it already reports
`"loglog"` under PC-F, which is correct since PC-F skips both steps.

PC-F is not the only way to land on an empty stiffness plot -- `plots.render_stiffness` shows
an instructional title in place of the plot (no exception, no blank axes) whenever
`res.stiffness_S` is `None`, which `model.compute_all`'s stiffness block leaves unset in any of
four gate failures: `state.min_dpdg_G` not yet picked, no pore-pressure estimate (this is where
the PC-F case actually surfaces, transitively, per `stiffness_skipped` above), `res.te_s`
unavailable, or fewer than 4 resampled points surviving (the tail trim, an aggressive resample
step, or a short record can all produce this). A separate guard inside the same renderer handles
a fifth, arrays-present-but-degenerate case: `stiffness_S` computed but entirely
non-positive/non-finite (no positive sample to log-scale against) also renders an instructional
title rather than an axis-only log plot.

An in-app "Interpretation guide" window (opened from the "Interpretation guide..." buttons on
the G-function step's closure-scenario panel and the log-log/pore-pressure steps'
postclosure-scenario panel) shows the ResFrac closure (C-A...C-D) and postclosure (PC-A...PC-F)
figures and explanatory text side by side with the pickers. Content lives in
`dfit_tool/guide_content.py` (headless, no Tkinter); figures are vendored PNGs under
`dfit_tool/assets/guide/`. Both buttons open the same reused `ttk.Notebook` window and just
select their tab (`ui.py:_open_guide`).

**Unit detection and conversion.** Everything downstream of the IO boundary (fixed psi
offsets, the hydrostatic constant, plot labels, panel rows, `dfit_log.csv` columns) stays
field-unit (psi/bpm/bbl); metric input is normalized at the boundary instead. Detection order
per channel (`io_load.detect_channel_unit`): a per-channel UI override (anything but `"auto"`)
→ the column header's parenthesized suffix (`_unit_of`, e.g. `"CASING Pressure (KPAg)"`) →
a kind-specific fallback. Pressure's fallback is the magnitude heuristic
(`classify_pressure_magnitude`): 99th percentile of finite samples, `< 15,000` → psi/high,
`>= 20,000` → kpa/high, the 15-20k overlap band → psi/low (no clean single-unit reading, so it
defaults to the historically-assumed unit). Rate and volume get no standalone heuristic (the
bpm/m3-min ranges overlap with no clean threshold) and instead inherit the file-level pressure
verdict -- any non-psi pressure unit means metric rate/volume too, always at low confidence
(an inference from another channel, not direct evidence) -- but a channel's own header suffix
always wins over that inheritance. Conversion is lazy: `TestData.column()` multiplies the raw,
never-mutated df column by a cached factor in `td.unit_factors` (default 1.0 = no-op for any
caller that never invokes detection); `io_load.refresh_unit_detection` rebuilds
`unit_factors`/`unit_detections`/`unit_warnings` from the raw columns and is idempotent, so
`model.compute_all` calls it at the top of every recompute (before any channel read, including
picks seeders) rather than caching a detection result across state changes. Warning policy
(`_maybe_warn`) is never-silent-on-a-real-conversion: always warn when a resolved factor isn't
1.0, also warn on a low-confidence pressure heuristic call even at factor 1.0 (the classification
itself is uncertain even though it landed on the default unit) -- silent only for the ordinary
factor-1.0 field-units load. `PickState` carries `pressure_unit`/`rate_unit`/`volume_unit`
(each `"auto"` by default; old saves lacking the keys take the dataclass default, no migration
needed). `ui._on_unit_change` mirrors `_on_source_change`: changing a channel's unit away from
its current value pops a confirm dialog (an override rescales the same column under existing
picks -- `isip_tangent` stores an absolute psi anchor, `te_s` feeds `g_time`), reverting the
combobox on decline; accept resets picks via `_reset_picks_keep_mapping` (channel mapping, unit
overrides, density/TVD, well/formation, alpha, resample step, notes, and active source carry
forward; numeric picks/step_status/tail_trim reset) and returns to Overview. With no file
loaded (`self.td is None`) it just reverts the combobox before ever reaching the dialog.
`store.LOG_COLUMNS` carries one `units_note` column (tail-appended, per the append-only
convention), mapped from `DerivedResults.unit_conversion_note` -- a compact summary of every
non-1.0 factor applied (e.g. `"pressure: kpa×0.145038 (header)"`).

`questionnaire.py` gained matching metric handling: a TVD cell is read as meters, detected any of
three ways -- an explicit meter suffix (`mTVD`/`mKB`/`mMD`/bare `m`/`metre(s)`/`meter(s)`, checked
immediately after the matched number so a bare number stays feet); the label itself spelled
`mTVD` (e.g. `"mTVD: 3368"`); or a delimiter-bounded meter token in the gap between the label and
the number (e.g. `"TVD (mKB): 3368"`, `"TVD in mKB = 3368"`) -- and is converted via
`units.M_TO_FT` *before* the `_TVD_MIN`/`_TVD_MAX` range filter -- ordering matters, since a
meters value like 3368 sits inside the feet-assumed 1000-25000 window and would otherwise be
silently accepted as feet; density gained a kg/m3 unit (`units.KGM3_TO_PPG`) alongside the
existing ppg/SG/psi-per-ft forms, and a fluid-name fallback (`"fresh water"`/`"freshwater"`/
`"water"`, exact trimmed whole-cell match) that assumes 8.34 ppg with a warning when a cell names
the fluid but gives no number at all. Multi-well workbooks
(`parse_questionnaire(path, well_hint=...)`) disambiguate a
multi-sheet workbook by matching `well_hint` against each sheet's title, then its "Well Name"
answer cell; a positive match narrows to that one sheet, but a missing hint or no match reads
the *whole* workbook (concatenating every sheet's cells) rather than guessing `sheets[0]`
alone -- narrowing only ever happens on a positive match, so a `scripts/` caller that passes no
hint at all still finds data that happens to live on a later sheet (e.g. behind a cover/
instructions sheet). `well_hint` itself comes from `ui.py`: folder mode passes the queue
entry's `test_id`, single-file mode passes the data-file stem (`ui._load_questionnaire`). The
match (`questionnaire._sheet_match_length`) is delimiter-bounded containment in both
directions -- the contained string must sit between non-alphanumeric characters or a string edge
in the container, so a short hint like "1H" can't false-match inside "21H" -- and when more than
one sheet matches, the one with the longest matched text wins, with a warning naming the chosen
sheet.

Accepted gaps, deliberately not solved: a psi-pressure file with an unlabeled metric rate/volume
is undetectable in-band (the per-channel override dropdown is the escape hatch); and there is no
filename-based unit sniffing, ever -- the corpus has a "Metric"-named DBS file that holds psi
data byte-identical to its non-metric-named sibling, so a filename heuristic would misclassify it.
In `questionnaire.py`'s TVD label matching, a digit-ending token right before an `mTVD` label
(e.g. `"Zone 2 mTVD: 3368"`) and a suffix-only TVD-first cell with no colon label (e.g.
`"3368 mTVD / mMD 6656"`) are still read as feet/MD respectively -- inherent label-vs-suffix
ambiguity, not fixed.

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
- A per-channel unit override (`ui._on_unit_change`) mirrors the Source-switch limitation above:
  it only persists to disk on the next save (queue navigation, Finish, or Skip test), not
  immediately.
- A pre-`step_status` legacy picks JSON, reloaded, can silently re-run the Overview auto-trim
  seeder and change reported numbers. `infer_step_status` backfills `step_status` for saves made
  before it existed, but it omits `"overview"` from that backfill, so `"overview"` still reads
  `not_visited` on reload -- and since the tail-trim seeder is park-and-apply on that step's first
  visit (Tail trim, above), re-visiting Overview can re-seed a trim (or a different one than
  whatever was in effect when that save's `dfit_log.csv` row was written), moving Shmin/effective
  ISIP/net pressure/pore pressure with no analyst action. This is a known consequence of the
  park-and-apply design colliding with an unrelated legacy-migration gap, not something the tail
  trim feature itself can detect or guard against.
- Adding "stiffness" as an 8th workflow step is an accepted migration consequence for saves
  finished under the old 7-step workflow: those have a 7-key `step_status` with no
  `"stiffness"` entry, so `store.status_for` now reports them `"in_progress"` (not `"done"`)
  until the analyst opens the test and visits or skips the stiffness step. This is deliberate --
  the stiffness Shmin genuinely doesn't exist for those saves yet, so `"in_progress"` is the
  accurate status, not a bug -- and `infer_step_status` can't backfill it, since that function
  only runs when the loaded `step_status` is empty, not when it's merely missing one key.
- The G-function plot's `guard_excluded` gray preview (`model.compute_all`'s `res.guard_excluded_G`/
  `guard_excluded_p`) is unaware of `tail_guard_override`: it still previews the full raw tail past
  `guard_dt` even after an override has extended real, kept data into that same span. Cosmetic
  overlap only -- the preview and the real (ungrayed) resampled curve can occupy the same G-range --
  not a data-correctness issue, since `res.resampled`/`res.diagnostics` are correct either way.
- A second, unrelated excursion occurring AFTER the first guard fire (while `stop_at_guard=False`
  keeps resampling) is never separately detected or warned about -- only the first excursion gets
  `guard_dt`/`guarded_at` and a warning; `resample_pressure_increment` never re-arms the fire check
  once `guard_dt` is set. Accepted as a low-severity gap, not a correctness issue: the second
  excursion's own samples still correctly never get kept, by the ordinary resampling rule (a
  rising sample never satisfies `pi <= last_kept - step`), they just get no dedicated
  metadata/warning of their own.
- The apparent-ISIP tangent fit (`interpret.tangent_from_index`, ±5 samples) and the ISIP anchor
  snapping still read raw `res.bhp_all` -- neither one consults `dropout_mask`. An anchor placed
  within 5 samples of a masked dropout gets a corrupted fit.
- `resample.detect_dropouts` only masks a dip that falls all the way to `DROPOUT_FLOOR_FRAC` (10%)
  of the pre-dip level; a partial dip that doesn't reach the floor is never masked, by design (see
  "Pressure dropouts" above) -- it isn't distinguishable from a real, if abrupt, decline.
- Dropouts are only detected after shut-in (`model.compute_all` runs `detect_dropouts` on the
  post-`t_shutin_s` record only); a dropout during injection is not masked anywhere.
- A chronically flaky channel is masked one event at a time, each independently -- there is no
  single warning that says "this channel is unreliable." The event count in the "N pressure
  dropouts masked" warning is the only signal that the channel itself, not just one moment, is
  the problem.
- Detection runs on the mapped pressure channel only (`state.pressure_col`, whatever the analyst
  or the auto-suggestion chose). The auto-suggested channel can itself be wrong -- e.g. a
  secondary gauge that reads ~0 for most of the falloff -- and `detect_dropouts` has no way to
  tell that apart from a real signal; that's a separate channel-suggestion issue, not handled
  here.
- `plots._plot_dropout_markers`'s autoscale protection (see "Pressure dropouts" above) snapshots
  `ax.dataLim` before plotting the markers and restores it after -- it does not make the markers
  permanently invisible to `dataLim`, only to that one `plot()` call's own autoscale request. Any
  FUTURE call that recomputes `dataLim` from scratch (`ax.relim()`, or `ax.autoscale_view()` after
  `relim()`) would walk every artist on the Axes, including the markers, and pull them back into
  the bounding box. No current code path does this on these two renderers' Axes, but it's a latent
  trap for a future change that adds one.