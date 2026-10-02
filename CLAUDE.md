# CLAUDE.md

Guidance for working in this repository. The reasoning behind specific behaviors, measured
corpus cases, and edge-case history live in `docs/implementation-notes.md` (search it by
function name). Read the relevant section there before changing loader, resampling, tail-trim,
dropout, or unit-detection behavior: most of those rules exist because a real file broke.

## Overview

An interactive tool for interpreting a single diagnostic fracture injection test (DFIT) by the
compliance method. A Tkinter/ttk shell hosts an embedded matplotlib canvas. The interpreter opens
one data file (CSV, Fracpro `.DBS`, or an XLSX time-series workbook), maps its channels, and walks
eight workflow steps, making draggable picks on each plot. Every reported number comes from one
pure function, `model.compute_all`.

Two modes. Single-file mode opens one file directly. Folder mode ("Open Folder…") scans a root
into a queue of tests shown in a left sidebar and rolls each test's results into a per-root
`dfit_log.csv`. There is no other cross-test aggregation. Permeability is out of scope.

The eight steps (`model.STEPS`): overview → injection → isip → gfunction → tangent → loglog →
porepressure → stiffness.
- Overview: the whole dataset, unclamped, with the always-on tail-trim line. When BHP is converted
  from surface pressure it also overlays the raw surface trace (`DerivedResults.p_surface_all`).
- Injection: zoomed injection window with draggable start/shut-in lines and a te/Vinj/qmax title.
- Stiffness: semilog-y relative stiffness vs effective pressure (URTeC-2019-123 A.8/A.9). Needs
  the pore-pressure estimate, so it comes last. Porepressure and stiffness are both skipped end
  to end under PC-F.

## Commands

Dependencies are not on the system Python. Always use the project venv (Python 3.14, kept
outside the project so OneDrive does not sync it):

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe

Run the app:

    start-app.cmd                                                   # double-click, no file loaded
    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app [path/to/file|folder]

Run the tests (from the repo root):

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest

`start-app.ps1` creates the venv if missing (`py -3.14 -m venv`, falling back to
`C:\Python314\python.exe`) and runs `pip install -r requirements.txt` only when importing
numpy/pandas/matplotlib/scipy/openpyxl fails. Pinned: numpy 2.5.1, pandas 3.0.3,
matplotlib 3.11.1, scipy 1.18.0 (pinned but not imported anywhere), openpyxl 3.1.5, pytest 9.1.1.

## Architecture

The package `dfit_tool/` is layered. Lower layers never import higher ones.

    app → ui → {io_load, picks, plots, sliders, model, interpret, questionnaire, store}
    picks, plots → model, io_load, interpret
    model → interpret, resample, io_load
    resample → gfunction
    store → model, questionnaire
    io_load, questionnaire → units

- **Leaf math (numpy only).** `gfunction.py`: Nolte G-function and G-time (α=1 default, α=0.5
  option). `interpret.py`: te, apparent/effective ISIP, Shmin variants, net pressure, pore
  pressure, gradients, h-function/stiffness, and the `suggest_*` auto-pick helpers. `units.py`:
  conversion constants and header/alias unit tokens, no `dfit_tool` imports.
- **Compute core.** `resample.py`: 30-psi pressure-increment resampling, diagnostic derivatives
  (dP/dG, G·dP/dG, d²P/dG², t·dP/dt), the tail guard, and dropout detection. `model.py`:
  `PickState` (the serializable interpreter choices), `DerivedResults`, and
  `compute_all(state, td)`.
- **IO.** `io_load.py`: `load()` dispatches on extension (case-insensitive) to `load_csv`,
  `load_xlsx`, or the reverse-engineered Fracpro `.DBS` reader. CSV and XLSX share everything
  after "file → raw DataFrame" via `_finish_frame`, so every CSV time-base fix applies to XLSX.
  Also: `suggest_channels` (ranked channel-role suggestion), the hydrostatic surface→BHP
  conversion (`BHP = WHP + 0.052·mw·tvd`, valid post-shut-in), and per-channel unit detection.
  `questionnaire.py` reads fluid density and TVD from a `*questionnaire*.xlsx` next to the data.
- **Interaction (matplotlib only, no Tkinter).** `picks.py`: event controllers
  (`DragLineController`, `AnchorLineController`, `DraggablePointController`, `SpanController`,
  `ModifierSpanController`, `HoverCursorController`, `_CaptureGate` press arbiter), pure
  `commit_*` functions turning finished geometry into `PickState` changes, and per-step `seed_*`
  functions. `plots.py`: `render_*` renderers and the headless PNG export
  (`render_step_figure`/`save_all_step_pngs`, which also writes `9_summary.png` via
  `render_summary`). `sliders.py`: `PanRangeSlider`.
- **Expanded results.** `summary.py` (Tk-free, model/interpret only): `summary_sections` builds
  the grouped tables and `chart_values` the values `plots.render_summary` draws, both through
  the same not-visited gate (`summary.visited`) the sidebar uses. Computes nothing; new values
  go in `compute_all`. `ui._open_results_window` shows both in a live, non-modal Toplevel.
- **Folder-mode persistence.** `store.py` (Tk-free): `scan_root`, `TestEntry`, per-test picks
  JSON, `status_for`, and the `dfit_log.csv` read/write/upsert.
- **Shell.** `ui.py` (`DfitApp`) is the only Tkinter consumer. It wires pickers to a
  recompute-and-redraw loop and holds no interpretation logic. `app.py` launches it with an
  optional file or folder path.

### Loader time base (`io_load.py`)

This is the most fragile area. Details and the files behind each rule are in
`docs/implementation-notes.md`. The shape:

1. `parse_datetime`: two exact vectorized fast paths (24-hour, 12-hour AM/PM, plus a
   `"03-26-2018_16:44:05"` shape), each re-filtered to 1990–2100 (`_PLAUSIBLE_DT_MIN/MAX`); then
   an Excel-serial parse within `_EXCEL_SERIAL_MIN/MAX`; then a generic dateutil parse at FULL
   LENGTH (never reindexed to a subset, since pandas infers one shared format from the array).
   Before the generic parse, values that can never become a new date are blanked to NA (bare
   numbers, bare clock strings with no date via `_TIME_OF_DAY_RE`, values with no digits). The
   generic parse is skipped entirely when the fast paths resolve >= `_FAST_PATH_SKIP_FRACTION`
   (90%) of non-empty cells. tz-aware values drop their offset; mixed offsets retry with
   `utc=True`. `reject_bare_clock=False` is used only by the FIX-A companion Date+Time join.
2. Column choice: when several columns look datetime-ish, `_best_datetime_column` scores each
   on a <=5,000-row sample. A Date column pairs with its companion Time column
   (`_companion_time_col`, then `_companion_is_time_of_day` vets the values) regardless of column
   order. Elapsed-style names ("Elapsed Time", "Test Time", …) are never companions.
3. Trust check: if the datetime column parses < `_MIN_VALID_DT_FRACTION` of its own non-empty
   cells (`_non_empty_mask`), fall back in order to `_find_elapsed_column`,
   `_find_numeric_time_column` (bare `Time` with no unit hint assumes MINUTES with a load
   warning), then `_find_clock_column` (`_classify_clock_mode` resolves H:MM vs MM:SS; raises on
   ambiguity rather than guessing). If none applies, raise.
4. Cleanup: FIX D reverses newest-first logs; `_mask_isolated_timestamp_outliers` NaTs a single
   sample that disagrees with both neighbors; `_extrapolate_or_warn_edge_block` extrapolates a
   leading/trailing block of blank or error-token timestamps only when the block is contiguous,
   the good run is exactly regular, the local slope agrees, and the columns continue
   (`_columns_continue`). Otherwise it warns with a row count. Never silently drop rows.

`load_xlsx` (openpyxl, `read_only=True, data_only=True`): `_xlsx_find_header` finds a header row
(text with a datetime-ish name, followed by data-shaped rows spanning >=2 columns, skipping a few
all-text rows); a units row below it folds into names as `"Name (unit)"`; a blank header cell
takes text from the row above. Multiple qualifying sheets rank by (time base usable, real row
count via zip-level `_xlsx_zip_count_data_rows`); never trust `ws.max_row`/`<dimension>`. Reads
stop after 1000 consecutive blank rows (warning if data exists past the gap). A sheet with zero
span falls through to the next; all-zero raises. `sniff_xlsx_data(path)` is a cheap zip/XML peek
used by folder mode and `scripts/triage` to tell real time-series workbooks from the hundreds of
non-data `.xlsx` files; it never raises.

### Folder mode and persistence

- `store.scan_root` walks the whole root (`os.walk`). One `TestEntry` per stem group of data files
  per directory, keyed by a path-qualified `test_id` (`_entries_for_dir`). A `.xlsx` counts only if
  it is not an Excel lock file (`~$`), not a questionnaire, and passes `sniff_xlsx_data`.
- **test_id stability is a hard invariant**: adding a data `.xlsx` must never change an existing
  csv/dbs test_id. A same-stem xlsx joins that group; an xlsx-only collision is renamed
  `<id>/<stem>` with a warning. Enforced by `_entries_for_dir` counting only csv/dbs stem groups
  when deciding whether a directory collapses to `test_id = rel`. A loose file colliding with a
  subfolder entry is dropped for the subfolder; csv/dbs-vs-csv/dbs and xlsx-vs-xlsx collisions
  keep deeper-folder-wins. Verified across all 3269 corpus ids.
- Sources: `TestEntry.available_sources` in order CSV, DBS, XLSX. Switching source resets picks
  after a confirm (`ui._on_source_change`).
- Picks persist to `<folder>/<test_id>.dfit_picks.json`, written atomically (temp +
  `os.replace`). In single-file mode, picks save/load through a file dialog instead.
  `DerivedResults` is never serialized. `model._decode` migrates legacy saves and drops unknown
  keys so old or foreign JSON never raises. `infer_step_status` backfills `step_status` for old
  saves.
- `store.status_for`: `"done"`/`"in_progress"` derive purely from `step_status` (PC-F counts
  porepressure and stiffness as accounted for); `"skipped"` also comes from
  `state.explicit_status` (the Skip-test toggle), which wins over derivation.
- `dfit_log.csv`: `build_log_row` maps `PickState`/`DerivedResults` to `store.LOG_COLUMNS`
  (computes nothing itself); `upsert_log_row` replaces or appends by `test_id`. **`LOG_COLUMNS`
  is append-only**: new columns go at the end.
- Ending a test: Finish (on the last step) saves picks, writes a PNG of every step to
  `<stem> DFIT plots/`, and in folder mode upserts the log row and advances to the next `"new"`
  test (`ui._advance_queue`). Finish clears `explicit_status` but keeps a per-step `"skipped"`
  set by "Skip >" on the step it runs from, so such a test still reads `"skipped"`. Skip test
  sets `explicit_status = "skipped"`, saves, logs, and advances; Unskip clears it, saves, logs,
  and stays put. Single-file Finish re-saves `<stem>_picks.json`, no log write.
- `scripts/triage` builds on `store.scan_root`. A non-keeper `.xlsx` is never quarantined, and
  the decision fingerprint (`ledger.group_files_sig` over `features.sig_files_for`) uses csv/dbs
  files only unless an xlsx is itself a keeper.

## Conventions and invariants

Preserve these when changing the code.

- **`compute_all` is the single source of truth.** New reported values are computed there, never
  in the UI. `ui.py` only displays what it returns.
- **Keep `picks.py`, `plots.py`, `sliders.py` free of Tkinter.** They run headless under Agg.
- **Renderers never set view limits.** `render_*` leave Axes autoscaled and return
  `ViewDefaults(xlim, ylim, y2lim)`. `ui.py` owns per-step view state (`_views`). No
  `set_xlim`/`set_ylim` inside a renderer. `plots.apply_step_view` is the one place view
  resolution happens (returns `StepView`: the applied `ViewState` plus slider ranges); both
  `ui.refresh` and `plots.render_step_figure` call it. On every step except gfunction, the
  renderer's `ViewDefaults.ylim` is unioned into `full_y` (the slider's outer range), or
  `_make_range_slider`'s valinit clamping snaps the view off the default on first touch.
  Gfunction replaces `full_y` instead; its dP/dG slider range is the twin autoscale unioned with
  the default, then clamped to `(0, DPDG_VIEW_MAX)`.
  Rate axes (overview, injection, isip) default to 0..3x the max rate plotted on that step
  (`plots.RATE_VIEW_FACTOR`, `_rate_y2lim`) so the rate trace rides low; the default is unioned into
  `full_y2` on every non-gfunction step.
  Default y-limits round outward to tick values (`plots.nice_limits`, 1/2/2.5/5 x 10^k steps;
  `nice_log_limits` rounds the stiffness log axis to decades).
- **Hit-test through own-axes pixel transforms, never `event.inaxes`** (a `twinx` owns `inaxes`
  over the shared region). Use `_axes_contains_pixel`/`_data_from_pixel`.
- **Slider `on_changed` callbacks never call `refresh()`.** `refresh()` calls `fig.clf()`, which
  destroys the slider mid-drag. Callbacks only `set_xlim`/`set_ylim`, mutate the `ViewState`, and
  `draw_idle()`. Hold slider references on `self` (matplotlib keeps no strong ref). Slider layout
  is pixel-based and scales with `fig.dpi / 100`; it re-runs on resize via `set_position`, never
  `refresh()`.
- **Diagnostic derivatives are reported positive-up for a declining pressure** (negated).
- **Downstream of IO everything is field units** (psi/bpm/bbl). Metric input is converted at the
  boundary: `TestData.column()` multiplies the never-mutated raw column by `td.unit_factors`;
  `io_load.refresh_unit_detection` is idempotent and runs at the top of every `compute_all`. A
  real conversion is never silent (warning + `units_note` log column). No filename-based unit
  sniffing, ever.
- **Warnings are never silent about changed numbers.** An auto-trim, a masked dropout, a guard
  cutoff, or an extrapolated timestamp block always produces a warning. Warnings that explain a
  cut go at the front of `DerivedResults.warnings` (`insert(0)`). Keep each message to one short
  line (`tests/test_warning_levels.py` caps it at 90 characters). Empty trailing `.DBS` records
  are dropped without a warning.
- **Three issue levels** on `DerivedResults`: `blockers` (no pressure channel, surface pressure
  without density/TVD, a failed BHP conversion), `warnings` (changed or possibly invalid
  numbers), `notes` (informational: TVD missing so gradients are blank, stiffness decimation).
  `model.blocking_issues(state)` gates Next and Skip on Overview (`step_gate_error`); `_goto` and
  `refresh` send any other step back to Overview while a blocker holds. Overview shows every
  issue in place of the Results rows (`ui._update_issues_panel`); other steps show per-level
  counts (`ui.format_warnings_text`).

## Testing

Tests live in `tests/`. `tests/conftest.py` forces Agg before `matplotlib.pyplot` is imported, so
the suite runs without a display or Tk. Synthetic data: `tests/helpers.make_testdata` (a
DFIT-shaped `TestData`) and `helpers.injection_state`. Controllers are driven with real
synthesized `MouseEvent`s. GUI-only paths are tested by binding real `DfitApp` methods onto
duck-typed stand-ins rather than building a `tk.Tk()`.

New logic goes in a headless layer (`model`, `interpret`, `resample`, `picks`, `plots`) with a
test there, not in the Tkinter shell. Vectorized fast paths keep their loop reference
implementations (`resample._resample_loop_reference`, `resample._dropout_scan_loop`) and a fuzz
test against them.

## Domain and methodology

Compliance-method interpretation per McClure et al. (URTeC-2019-123) and the ResFrac practical
guidelines. Refs are in `Refs/` (gitignored).

Per-test deliverables (all computed in `compute_all`):

| Value | Construction |
|---|---|
| Apparent ISIP | early BHP-decline tangent extrapolated to shut-in (manual pick on isip step), or, with `state.isip_at_shutin` ("Use shut-in pressure", for tests with no water hammer), `bhp_all[shutin_idx]`; blank with a warning when that sample is NaN or dropout-masked. The tangent pick is kept in state. Method logged as `apparent_isip_method` (`tangent`/`shutin`) |
| Effective ISIP | P-vs-G straight line extrapolated to G = 0 |
| Shmin compliance | contact pressure − 75 psi (`interpret.COMPLIANCE_OFFSET_PSI`) |
| Shmin tangent | BHP at the G·dP/dG through-origin departure (closure) point. Blank, with the tangent effective ISIP, when "Tangent closure uninterpretable" (`state.tangent_uninterpretable`) is checked |
| Shmin variable | BHP at the G-time midpoint of contact and closure picks. Blank when either is unavailable (C-C/C-D/C-X, or the tangent checkbox) |
| Shmin Liberty | BHP at the anchor pick − 200 psi (`LIBERTY_OFFSET_PSI`); anchor is the min-dP/dG pick for C-A/blank, the contact pick for C-B; requires `contact_G` set; blank for C-C/C-D/C-X. Comparison only. |
| Shmin stiffness | stiffness-step pick − 75 psi. Comparison only. Blank when "No slope change apparent" (`state.stiffness_no_upturn`) is checked or under C-X |
| Shmin rapid | C-D only: apparent ISIP − 175 psi (`RAPID_CLOSURE_OFFSET_PSI`) |
| Net pressure | shared reference ISIP − that method's Shmin (compliance, tangent, variable) |
| NWB complexity | apparent ISIP − shared reference ISIP; negative reported as-is |
| Pore pressure | intercept of the late postclosure line on t^(−1/2) or t^(−1) |
| Gradients | value / `state.tvd_ft`; blank unless TVD is finite and > 0 (warns when blank) |

"Comparison only" values are shown in the panel and logged, but never feed net pressure, the
shared reference ISIP, or complexity. Shmin Liberty is not drawn on any plot; the stiffness pick
is drawn (and draggable) on the stiffness step only.

- **Shared reference ISIP**: the compliance effective ISIP, else the tangent effective ISIP, else
  none (no apparent-ISIP fallback). Source is logged as `net_pressure_isip_source`. The identity
  `Shmin + net pressure + complexity = apparent ISIP` holds for all three methods. There is no
  `net_pressure_rapid`.
- **Resampling**: after shut-in, keep a point whenever BHP has moved >= 30 psi in either
  direction from the last kept point. This replaces time-domain smoothing. The tail guard fires
  when pressure stays > 30 psi (`RISE_GUARD_PSI`) above its running minimum for >= 60 s and >= 5
  samples; `guard_dt` is pinned to the first qualifying run.
- **Tail trim** (`PickState.tail_trim_dt`, shut-in-relative seconds): seeded on Overview's first
  visit by `picks.seed_tail_trim` (sub-100-psi surface crash → trim strictly before it, reason
  `"low_pressure"`; a guard boundary sets no pick). `interpret.resolve_tail_cut_dt(trim,
  guard_dt, override)` is the single place the effective cutoff is resolved, for both masking and
  display; when the cutoff equals `guard_dt` the mask is `<`, otherwise `<=`. A drag past
  `guard_dt` sets `tail_guard_override`. `seed_tail_trim` is called from `ui._seed_step`, not
  `SEEDERS`, because it needs a recomputed `res`. Moving shut-in calls
  `picks.resync_auto_tail_trim`, which only touches `"low_pressure"` trims. Scenario changes and
  min-dP/dG corrections never touch the trim.
- **Dropouts**: `resample.detect_dropouts` masks brief near-zero gauge dips that recover (raw
  mapped channel, post-shut-in only). Consumers treat masked samples as NaN; `res.bhp_all` is
  never mutated. Overview/ISIP draw them as magenta markers excluded from autoscale.
- **No rate channel**: `interpret.suggest_injection_window_pressure` seeds the window from the
  pressure shape; te falls back to pump wall-clock duration with a warning. With rate,
  `suggest_injection_window` returns the last rate-on run whose size (volume gain, else summed
  rate) is >= 10% of the largest run's (`INJECTION_MIN_RUN_FRAC`).
- **G-function**: α = 1 default; α = 0.5 only if a test exceeds ~1 md. The dP/dG default view
  autoscales over G >= 1.0 (`Y2_SCALE_G_MIN`), capped at `DPDG_VIEW_MAX` (500).
- **Min-dP/dG pick**: seeded by `interpret.suggest_min_dpdg_index` (interior local min before the
  contact hump, `suggest_hump_index`). Dragging re-derives the contact
  (`re_derive_contact_from_min`). Shift+drag selects a window (`handle_min_dpdg_window`).

Closure scenarios (`picks.apply_closure_scenario`):

| Scenario | dP/dG shape | Stress pick |
|---|---|---|
| C-A clear | clear "S" (min then rise) | contact at min + 10%, − 75 psi |
| C-B adequate | monotonic with inflection | contact at the inflection, − 75 psi |
| C-C no-contact | monotonic, no inflection | none (no Shmin) |
| C-D rapid | monotonic, concave-up | apparent ISIP − 175 psi (`shmin_rapid`) |
| C-X uninterpretable | can't be read (bad data) | none; also blanks Shmin stiffness |

C-C, C-D, and C-X (`model.NO_CONTACT_SCENARIOS`) clear the contact pick, so none gets a
compliance Shmin or compliance effective ISIP. C-X logs `closure_quality = "uninterpretable"`.
The tangent step's independent negative finding is the `tangent_uninterpretable` checkbox
(logged in the tail column of the same name); with both set there is no shared reference ISIP,
so net pressure and complexity are blank. Panel labels get a trailing `*` when a fallback fed the value
(`"Shmin compliance*"`/its gradient when showing `shmin_rapid`; `"NWB complexity*"` when the
reference is the tangent effective ISIP). The asterisk gate must match whatever makes the value
column show a number, so an asterisk never sits next to `"-"`.

Postclosure scenarios (`picks.suggest_pp_axis` maps by `scenario[:4]`):

| Scenario | log-log signature | Pore-pressure axis |
|---|---|---|
| PC-A linear | bends to −1/2 | t^(−1/2) |
| PC-B false-radial | −1 after peak | t^(−1) |
| PC-C false radial to genuine linear | −1 then −1/2 | t^(−1/2) |
| PC-D genuine linear to genuine radial | −1/2 then −1 | either (analyst picks) |
| PC-E no trend | peak, no clear slope | t^(−1/2), low confidence |
| PC-F no peak | derivative still rising | porepressure and stiffness skipped |

PC-F: `model.skipped_steps(state)` returns `{"porepressure", "stiffness"}`. It is the only place
the rule lives; callers ask it rather than checking the scenario. `model.last_step` returns
`"loglog"`, `model.resolve_step` (used by `ui._goto`) redirects both steps to loglog, their
breadcrumbs disable, their PNGs are omitted, `compute_all` leaves pore pressure blank, and
`store.status_for` counts them as accounted for. The stiffness flag is logged blank under PC-F.

The in-app "Interpretation guide" (`ui._open_guide`) shows the ResFrac C-A…C-D and PC-A…PC-F
figures; content is in `dfit_tool/guide_content.py`, images in `dfit_tool/assets/guide/`.

## Known limitations

Accepted, not bugs to fix opportunistically. Full list with measured files in
`docs/implementation-notes.md` ("Not built / notes"). The ones most likely to matter:

- No concurrency control: two windows on one root are last-write-wins on picks and the log.
- Source switches and unit overrides persist only on the next save (navigation, Finish, Skip).
- Unsupported layouts that load timestamped rows only, with a warning: side-by-side pump+gauge
  datasets, multi-rate TDMS "Formula Server" sheets, channels zero-padded past the timestamps.
- Headerless workbooks are not detected at all. One-Time-column-per-channel workbooks load
  against one chosen time base, no warning. Merged header cells name only their anchor column
  (the rest become `"Unnamed: N"`).
- A block (not a single cell) of corrupted timestamps is not caught (Arkansas 1BH reports
  ~3,850 h).
- Bare `Time` headers are assumed minutes; a seconds file reads 60x too long (warning only).
- Very large workbooks (85–97 MB) load slowly on the Tk main thread.
- The ISIP tangent fit and anchor snapping ignore `dropout_mask`; dropouts during injection are
  not masked.
- Only the first tail-guard excursion is reported; a second one is resampled without a warning.
- 7-step-era saves read `"in_progress"` until the stiffness step is visited (intended).
- Sample data, `Refs/`, and `.superpowers/` are gitignored. Old design specs are in
  `docs/superpowers/`. The log is CSV only (parquet mirror deferred).
