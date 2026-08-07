# No-rate fallback: overview vlines + injection-time fallback

Date: 2026-08-07
Status: approved (autonomous cycle; TODO items are prescriptive)

## Problem

Two paired CLAUDE.md TODOs:

1. When no rate channel is auto-detected, the start/shut-in vlines never appear on the
   overview step. Root cause: `picks.seed_overview` early-returns when `state.rate_col` is
   empty, and `plots.render_overview` only creates the draggable vline artists when
   `start_idx`/`shutin_idx` are set. With no rate there is nothing to drag, so the analyst
   can never establish an injection window.
2. Datasets without rate need an injection-time fallback. `model.compute_all` gates the whole
   injection-window block -- including `t_shutin_s`, not just te -- on
   `res.rate_all is not None` (model.py:332). Without rate, te is never set and every
   downstream step (ISIP, G-function, log-log, pore pressure) is dead. Per the TODO, the
   fallback te is the true elapsed time between the user-marked start and shut-in.

## Design

### 1. Pressure-based injection-window suggestion (`interpret.py`)

New pure function `suggest_injection_window_pressure(p: np.ndarray) -> tuple[int, int]`:

- `shutin` = index of the global max of `p` (nan-aware). In a DFIT the pressure maximum sits
  at or immediately before shut-in.
- `start` = onset of the rise into that maximum: threshold = baseline + 10% of
  (p[shutin] − baseline), where baseline = nan-min of `p[:shutin+1]`. Take the **last
  upcross** of the threshold at or before `shutin` (tolerates breakdown pulses and step-rate
  cycles earlier in the record); fall back to the first sample at/above the threshold.
- Degenerate inputs (flat record, `shutin == 0`, `start >= shutin`) fall back to positional
  defaults: `start = int(0.02*(n-1))`, `shutin = int(0.25*(n-1))`, coerced so
  `start < shutin`. The function raises `ValueError` only when fewer than 2 finite samples
  exist; otherwise it always returns a valid `start < shutin` pair. It is only a default --
  the interpreter drags the lines to the true window.

### 2. Seeder fallback (`picks.seed_overview`)

Unchanged early-return if picks already exist. Then:

- If `state.rate_col` is set, try `suggest_injection_window` as today.
- If there is no rate column, **or** the rate suggestion raises `ValueError` (e.g. an
  all-zero rate channel), fall back to
  `suggest_injection_window_pressure(td.column(state.pressure_col))` -- guarded on
  `state.pressure_col` being set and swallowing `ValueError` like today (no pressure channel
  -> no seed; `compute_all` already warns about that separately).

The renderer needs no change for the vlines: once the picks are seeded the existing
`gid="start"`/`gid="shutin"` artists appear and the existing `DragLineController` wiring in
`ui._attach_controllers` works as-is. The default-view xlim block already handles the
both-picks-set case.

### 3. te fallback (`model.compute_all`)

Restructure the injection-window block:

- Outer gate becomes `start_idx is not None and shutin_idx is not None` (drop the
  `rate_all` condition). `t_shutin_s` is set there, so apparent ISIP works without rate.
- The rate-dependent pieces (qmax, Vinj, disagreement warning, effective te = Vinj/qmax)
  stay inside a nested `if res.rate_all is not None`.
- After that, if te is still unusable (`None`, non-finite, or <= 0 -- the finite check also
  catches a NaN effective te from a NaN Vinj) and `shutin > start` with a positive elapsed
  time, set `res.te_s = t[shutin] − t[start]` and append the warning
  `"te = pump duration (shut-in − start); no usable rate for Vinj/qmax"`. Appended, not
  inserted at the front -- the front slot is reserved for the tail-guard warning.

This covers both the no-rate case and the rate-present-but-degenerate case (rate never
above threshold -> qmax 0/NaN). `qmax_bpm`, `vinj*` stay `None` without rate: the side
panel formatter and `store.build_log_row` are already None-safe.

### 4. Overview title crash fix (`plots.render_overview`)

The title currently formats `res.vinj` and `res.qmax_bpm` unconditionally whenever
`res.te_s` is truthy (plots.py:147-148) -- a fallback te with no rate raises `TypeError`.
Build the title from conditional pieces: always show `te=... min` when te exists; append
`Vinj=...` and `qmax=...` only when each is not None.

## Out of scope (YAGNI, per the TODO wording "simply")

- Vinj from a cumulative-volume channel when no rate channel exists.
- Effective te from a manually entered qmax without rate.
- Any change to the C-D/scenario logic, resampling, or the log schema (te already logs).

## Testing (headless layers only)

- `interpret.suggest_injection_window_pressure`: DFIT-shaped synthetic pressure (rise to a
  max, long decline) puts shut-in at the max and start near the rise onset; earlier
  breakdown pulse does not capture the start (last-upcross rule); flat/degenerate input
  returns the positional fallback; <2 finite samples raises.
- `picks.seed_overview`: no `rate_col` -> picks seeded from pressure; rate present but all
  zeros -> pressure fallback; existing picks never clobbered; no pressure col -> no seed,
  no crash.
- `model.compute_all`: start+shutin set, no rate -> `t_shutin_s` set, `te_s` = elapsed
  seconds, fallback warning present, `qmax_bpm`/`vinj` None; with a good rate channel ->
  behavior identical to today (effective te, no fallback warning).
- `plots.render_overview`: renders without error when `te_s` is set but `vinj`/`qmax_bpm`
  are None; title contains `te=` and not `Vinj=`.
