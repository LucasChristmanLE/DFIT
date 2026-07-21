# DFIT Batch Interpretation Workflow — Plan

## Context

We will receive 200+ CSVs of pressure/rate/time data from diagnostic fracture
injection tests (DFITs). For each test we need to produce a consistent,
auditable interpretation and then aggregate the results across the whole set.

The interpretation follows the **compliance-method procedure** of McClure et al.
(URTeC-2019-123) and the ResFrac practical-guidelines blog. Per-test deliverables:

- **Literal ISIP** — bottomhole pressure at shut-in, from a tangent fit to the early BHP decline
  extrapolated back to the shut-in instant (FracPro-style construction).
- **Effective ISIP** — y-intercept of the pre-contact straight line on the P vs G-time plot.
- **Shmin, compliance method** — contact pressure − 75 psi.
- **Shmin, tangent method** — closure pick from the G·dP/dG straight-line construction.
- **Pore pressure** — extrapolation of the postclosure log-log trend to reciprocal-time zero.

Aggregate results (the research question):

- **Δclosure** (Shmin_compliance − Shmin_tangent) vs (Shmin_compliance − pore pressure).
- **Net-pressure distribution**, compliance vs tangent (net = reference ISIP − Shmin).
- Effective vs literal ISIP; scenario-frequency counts and **closure-quality / postclosure-trend
  frequency counts**; grouped by play / fluid / orientation.

Permeability is explicitly out of scope (would require rock/fluid inputs and fracture-geometry
assumptions we won't have).

The tool must let one interpreter work through files interactively over many sessions,
**track which CSVs are done, store every pick permanently, and resume where we left off.**

Because we don't yet have the real data (details come at the meeting — see Open Questions),
this plan defines the scientific workflow, the data model, and the module architecture.
No code is written today.

---

## Two methodology points that shaped the design

**Pressure-increment resampling (the "30 psi" step).** Not a time-domain rolling mean.
After shut-in, pressure declines monotonically; we walk the 1-second samples and emit a new
(time, pressure) row only when pressure has dropped ≥30 psi below the last kept row. Rows end up
evenly spaced in *pressure*, unevenly in *time* (dense early, sparse late). This collapses ~1M
rows to a few hundred and makes the numerical derivatives (dP/dG, t·dP/dt) stable. It *replaces*
rolling-window smoothing. Guard the late-time tail: resample against the running cumulative
pressure drop and stop where the log-log derivative steepens beyond −1 (non-monotonic data).

**Impulse classification only picks the pore-pressure axis.** The slope class does not feed a
formula. −1/2 slope (linear) → extrapolate P vs t^(−1/2) to the intercept. −1 slope (radial,
genuine or false) → extrapolate P vs t^(−1). False radial is unusable for permeability but valid
for pore pressure. No established trend → low-confidence extrapolation; derivative never peaks →
no pore-pressure estimate.

---

## Per-file interactive workflow

Each file steps through the following. Every pick is stored so the file can be re-rendered and
edited later. Steps that don't apply (per scenario) are skipped and recorded as skipped.

1. **Load & map channels.** Accept either a **CSV** or a Fracpro **`.DBS`** (auto-detected by
   extension); both normalize to the same canonical structure (time, pressure channel(s), rate).
   Detect columns; prompt for rate and volume channels with suggestions; standardize to a canonical
   pressure channel. Normalize units. If the file is WHP, convert to BHP with fluid density (incl.
   dissolved solids) and gauge/perf TVD. **`.DBS` files carry surface pressure only** (no stored
   BHP channel — Fracpro derives BHP in the compressed `.INP`, which we do not read), so the WHP→BHP
   conversion is required for DBS-sourced tests. **Density and perf/gauge TVD are entered manually
   in the pick panel** per test; they are stored in the pick JSON and master log.
2. **Pick injection start & shut-in** on a Cartesian P & rate vs time plot. Ignore pretest
   pressure-up/bleed-off cycles; for step-down tests pick the *final* sudden pressure drop.
   Compute injection duration **te = total volume / max sustained rate** (not literal duration).
   The start pick has a defined numerical role: **Vinj and max sustained rate are computed over the
   `[start, shut-in]` window**, so a start placed after pretest cycles / wellbore-fill keeps that
   pre-injection volume out of te. The start instant is *not* used as a duration.
   Record the shut-in time as a persistent vertical marker carried into the ISIP and G-function plots.
3. **Literal ISIP (tangent construction).** Replicates the FracPro pick, done here *before* the
   G-function so it is available as a reference for effective ISIP and the rapid-closure band. On a
   Cartesian BHP-vs-time plot zoomed to the first minutes after shut-in:
   - A **vertical line marks the selected shut-in** (from step 2).
   - The interpreter **places a tangent line on the linear BHP decline** just after the water-hammer
     settles, drawn as a finite segment with a **short vertical tick through its anchor point** (the
     crosshair in the reference screenshot). The segment extends as a dashed line to the shut-in
     vertical. This is a manual pick — no auto-fit.
   - **Literal ISIP = the tangent value at the shut-in vertical** (the decline extrapolated back to
     the instant of shut-in), read off where the dashed extension crosses the vertical marker.
   - **Interaction:** dragging the anchor slides the tangent *along* the BHP curve; grabbing the
     **line body** pans it without changing slope; grabbing a **segment end** rotates it to adjust
     slope. Every drag re-reads the intercept live. Anchor and slope are stored in the pick JSON so
     the construction re-renders exactly.
4. **Resample** the shut-in transient at 30 psi increments; compute G-function (α=1 default;
   α=0.5 only if >1 md), dP/dG, G·dP/dG, and log-log dP and t·dP/dt vs actual shut-in time.
5. **G-function plot** (P & dP/dG vs G-time; dP/dG y-axis manually clipped so early spikes go off
   scale and the contact deflection is visible). Classify the **closure scenario** and pick
   accordingly (table below). Produce contact pressure → Shmin_compliance, and effective ISIP.
6. **Tangent method** — on the G·dP/dG plot, fit the straight line through the origin/early data;
   closure = departure point. Record Shmin_tangent and tangent Gc.
   **Auto-pick:** on entry the tool fits a line through the origin to the early G·dP/dG data (the
   near-linear segment before it rolls over) and flags the departure point where the curve leaves
   that line by more than a set tolerance — that becomes the suggested closure. The interpreter
   fine-tunes by dragging the fit window (which endpoints define the through-origin line) and/or the
   departure marker; both the suggested and final picks are stored in the pick JSON. If the early
   data has no clean straight segment, the auto-pick is flagged low-confidence and left for a manual
   pick.
7. **Log-log plot** — select the late-time window; fit average slope; classify the **postclosure
   scenario** (table below).
8. **Pore pressure** — on the axis chosen by the postclosure scenario (t^(−1/2) or t^(−1)), fit the
   late-time line and read the intercept. Record axis and confidence.
9. **Save** the master-log row + per-file pick JSON + annotated plot PNGs; mark status done/skipped.

### Closure scenarios (drives Shmin + effective ISIP)
Each scenario code also records a plain **`closure_quality`** label for grouping and QC.

| Scenario | `closure_quality` | dP/dG shape | Stress pick | Effective ISIP |
|---|---|---|---|---|
| C-A clear contact | clear | clear "S" (min then rise) | contact at min+10%, −75 psi | line from min-dP/dG point |
| C-B adequate | adequate | monotonic w/ inflection | contact just after inflection, −75 psi | line from inflection |
| C-C no contact | no-contact | monotonic, no inflection | **none** (no Shmin, no net pressure) | none |
| C-D rapid closure | rapid | monotonic, concave-up, no tortuosity (vertical/microfrac) | literal ISIP − 100–250 psi | ISIP; net pressure not estimable |

### Postclosure scenarios (drives pore-pressure axis)
Each scenario code also records a plain **`postclosure_trend`** label for grouping and QC.

| Scenario | `postclosure_trend` | log-log signature | Pore-pressure axis |
|---|---|---|---|
| PC-A linear | linear | bends to −1/2 | t^(−1/2) |
| PC-B false radial | false-radial | −1 after peak (gas/high-GOR) | t^(−1) |
| PC-C false radial → linear | mixed | −1 then −1/2 | t^(−1/2) (later segment) |
| PC-D linear → radial | mixed | −1/2 then −1 | either |
| PC-E peak, no trend | none | peak but no clear slope | t^(−1/2) from last point (low confidence) |
| PC-F no peak | none | derivative still rising | **none** |

---

## Data model & persistence

**Master log** — one row per test (`dfit_log.csv`, plus a parquet mirror):

`file`, `test_id`, `status` (new/in_progress/done/skipped), `interpreter`, `review_date`,
`orientation`, `fluid_type`, `play`, `pressure_source` (BHP/WHP), `tvd`, `fluid_density`,
`t_start_inj`, `t_shutin`, `te`, `Vinj`, `max_rate`,
`literal_ISIP`, `effective_ISIP`,
`closure_scenario`, `closure_quality` (clear/adequate/no-contact/rapid), `contact_pressure`, `Shmin_compliance`,
`Shmin_tangent`, `tangent_Gc`,
`postclosure_scenario`, `postclosure_trend` (linear/false-radial/radial/mixed/none), `pore_pressure`, `pp_axis`, `pp_confidence`,
`net_pressure_compliance`, `net_pressure_tangent`, `delta_closure`, `notes`.

**Per-file pick state** — `picks/<test_id>.json`: raw click coordinates, chosen windows, axis
scales, α form, unit/conversion settings. Makes every interpretation reproducible and re-editable.

**Annotated plot exports** — every interpretation writes a fixed set of marked-up PNGs to
`plots/<test_id>/` (regenerated from the pick JSON, so they always match the stored values):

- `01_overview.png` — P & rate vs time with injection-start and shut-in markers, te annotation.
- `01b_isip.png` — BHP vs time zoomed to post-shut-in, the ISIP tangent segment and its dashed
  extension to the shut-in vertical, and the literal-ISIP value at the intercept.
- `02_gfunction.png` — P & dP/dG vs G-time; contact point marked, min-dP/dG point marked, the
  effective-ISIP extrapolation line drawn to the G=0 intercept, closure scenario labeled.
- `03_tangent.png` — G·dP/dG vs G-time with the tangent construction line and the closure
  departure point; Shmin_tangent annotated.
- `04_loglog.png` — dP and t·dP/dt vs shut-in time; selected window and fitted slope (−1/2 or −1)
  drawn, postclosure scenario labeled.
- `05_porepressure.png` — P vs t^(−1/2) or t^(−1) with the fitted line extended to the
  reciprocal-time-zero intercept; pore pressure annotated.

Each PNG carries a title block with test_id, interpreter, date, and the numeric picks it depicts,
so a plot is self-documenting when pulled into a report or shared for QC.

**Resume logic** — the batch runner reads the master log, lists files by status, and opens the
next `new` file (or a chosen `in_progress`/`done` file to revise). Nothing is recomputed blindly;
a reopened file reloads its JSON and re-renders exactly.

---

## UI (Tkinter shell)

A single desktop window: a Tkinter/`ttk` shell with the matplotlib canvas embedded via
`FigureCanvasTkAgg`. Four regions:

- **Left — file queue** (`ttk.Treeview`): every CSV with a status color (done / in-progress / new /
  skipped) and a progress counter (e.g. 47/210). Click any row to open that file; this is the
  resume view, driven directly by the master log.
- **Center — plot canvas**: the matplotlib figure for the current step, carrying the interactive
  picking (tangent drag, contact click, window select).
- **Right — pick panel** (`ttk` form): live-updating computed values for the current step, scenario
  radio buttons / dropdowns (where `closure_quality` and `postclosure_trend` are set), and a notes
  box. Editing here and dragging on the canvas stay in sync.
- **Bottom — step bar**: the 9 workflow steps as a breadcrumb showing position, with Back / Next /
  Skip. Skipped steps are recorded as skipped.

Save writes the master-log row + pick JSON + PNGs and advances to the next `new` file.

**Toolkit is swappable.** All picking logic lives in `picks.py` on the matplotlib canvas and does
not depend on Tkinter. The shell only wraps the queue, form, and navigation, so it can be ported to
PySide6 later (if a more polished/native UI is wanted) without touching the picking code. If that
port happens, use PySide6 (LGPL), not PyQt (GPL/commercial), for the Liberty-internal license.

---

## Module architecture (Python, Tkinter shell + matplotlib interactive picking)

- `io_load.py` — loader for CSV **and** Fracpro `.DBS` (format detected by extension; DBS reader
  parses the channel-name table + float32 arrays), channel detection + prompt, unit normalization,
  WHP→BHP conversion. Conversion is hydrostatic only — **BHP = WHP + ρ·g·TVD, no friction term** —
  which is valid because the entire interpretation is post-shut-in (flow → 0 ⇒ friction → 0).
- `resample.py` — pressure-increment resampling; derivative calc (dP/dG, t·dP/dt, dP), tail guard.
- `gfunction.py` — G/g functions (α=1 default, α=0.5 option), dP/dG, G·dP/dG.
- `picks.py` — matplotlib event-based pickers: injection/shut-in, ISIP tangent (manual: draggable
  anchor, pannable body, rotatable ends), G-window & contact, closure tangent line (auto-suggested
  through-origin fit + departure point, user-adjustable), log-log window; each returns coordinates
  and writes to the pick JSON.
- `interpret.py` — literal & effective ISIP, Shmin (both methods), net pressure, pore pressure,
  scenario branching from the tables above.
- `store.py` — master-log read/write, per-file JSON, status + resume.
- `ui.py` — Tkinter/`ttk` shell: file-queue Treeview, embedded matplotlib canvas, pick-panel form,
  step-bar navigation; hosts the `picks.py` canvas and binds panel edits ↔ canvas drags.
- `batch.py` — app entry point; builds the queue from the master log via `store.py`, launches the
  `ui.py` shell, and picks up where we left off.
- `report.py` — aggregate correlation plots (Δclosure, net-pressure distributions, ISIP compare,
  scenario counts, closure-quality / postclosure-trend counts) and summary tables, grouped by
  play/fluid/orientation.

**Net-pressure convention:** compute net pressure from **effective ISIP − Shmin** (paper
convention; tangent's lower Shmin yields higher net pressure — the effect we want to characterize).
Also store literal ISIP − Shmin for comparison.

---

## Open questions (to resolve, not blocking the skeleton)

- **Pore pressure "compliance vs tangent."** Methodology yields one pore pressure per test.
  Decide whether the deliverable truly wants two estimates, or (default) one Pp per test with
  compliance/tangent applying only to Shmin and net pressure.
- Whether the 75 psi offset, 10% rise, and 100–250 psi rapid-closure band should be tunable.

## Questions to ask at the meeting

Data & format:
- **Delivery format across the set: CSV or Fracpro `.DBS`?** The two example wells arrived as `.DBS`
  (+ `.INP`, questionnaire/survey `.xlsx`, analysis PDF); confirm whether the full 200 are CSV,
  DBS, or mixed. The loader handles both; this sets which path is exercised at scale.
- Is the pressure channel **BHP or WHP**? Both example `.DBS` files carry **surface pressure only**,
  so we compute BHP ourselves. If WHP: fluid type/brine density (dissolved-solids ppm), and
  gauge/perforation **TVD** per test (entered manually in the UI).
- Are channel names, **units** (psi/kPa, bpm, bbl), sample rate, and time format/timezone
  **consistent across all 200 files**, or vendor-dependent?
- Is **rate/volume** data present for all, some, or none? If missing, is **total injected volume**
  recorded in metadata (needed for te = Vinj / max rate)?
- Do files contain **multiple cycles / step-down** tests we must split?

Per-test metadata (needed for BHP conversion, scenario expectation, and grouping):
- **Well orientation** (vertical/horizontal/deviated) — drives rapid-closure vs tortuosity.
- **Reservoir fluid type** (gas / oil / high-GOR) — gas → false radial expected.
- **Play/formation** — grouping variable for the aggregate correlations.

Interpretation conventions:
- Preferred **literal ISIP** definition. Default here is the FracPro-style tangent: fit the early
  BHP decline and extrapolate back to shut-in. Confirm this, or whether they want instantaneous
  post-water-hammer or a fixed time offset instead.
- **Example data file ahead of the meeting** — one representative CSV so the loader, shut-in pick,
  and ISIP tangent can be validated against real column names, units, sample rate, and BHP/WHP
  structure before the full 200-file set arrives.
- Confirm **G-function α=1** for all (shale), α=0.5 only if any test is >1 md.
- Which **reference ISIP for net pressure** — effective (recommended) or literal?
- Any tests expected to be **microfrac / very low rate** (rapid-closure regime)?

Scope & output:
- Confirm **no permeability** deliverable.
- Expected artifacts: master results CSV, per-test annotated PNGs, aggregate correlation plots —
  confirm and note any required report format.
- Single interpreter, or do they want two-interpreter QC on a subset?

---

## Verification (once implemented)

- **Reproduce the paper's worked example.** Run the Utica Point Pleasant 'A' case (or a synthetic
  matching it) and confirm effective ISIP ≈ 9800/8330-class values, contact-based Shmin, and the
  tangent estimate ~400 psi lower, per URTeC-2019-123 §3.1.
- **Resampling check.** Confirm a ~1M-row transient reduces to a few hundred rows spaced ~30 psi
  apart, monotonic, with stable derivatives; confirm the tail guard trims non-monotonic late data.
- **Round-trip persistence.** Interpret a file, close, reopen from the log → identical re-render
  and values from the JSON; confirm status/resume across a simulated multi-session run.
- **Scenario coverage.** Feed one example of each closure (C-A…C-D) and postclosure (PC-A…PC-F)
  case; confirm the correct branch fires and skipped steps are recorded as skipped.
- **Aggregate report.** With a handful of completed tests, confirm the Δclosure and net-pressure
  distribution plots and summary tables populate and group correctly.
