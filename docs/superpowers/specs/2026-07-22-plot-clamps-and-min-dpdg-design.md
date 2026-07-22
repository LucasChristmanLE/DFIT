# Plot clamps + min-dP/dG pick + derived effective-ISIP tangent

Date: 2026-07-22. Source: user feedback on the running app.

Four changes:

1. Overview: clamp the plot's x extent to last nonzero rate + 15 min.
2. G-function: cap the default dP/dG (twin-axis) y-max at 500.
3. G-function: add the missing tool that finds/marks the relative minimum of dP/dG.
4. G-function: the effective-ISIP line is no longer user-editable. It is the tangent to the
   P-vs-G curve at the min-dP/dG point (plan.md scenario table: "line from min-dP/dG point").

## 1. Overview x-max clamp

`plots.render_overview` currently plots the full record; the falloff tail (weeks) dominates the
autoscaled extent and the x-slider's full range. Following the `render_isip` precedent (clamp the
*plotted data*, which clamps autoscale and slider range together):

- If `res.rate_all` is not None and any `rate > 0`: let `t_end_h = t_h[last index with rate > 0]
  + 0.25`. Mask every plotted trace (pressure and rate) to `t_h <= t_end_h` before decimation.
- If there is no rate channel or the rate is all zero: no clamp (current behavior).
- The start/shut-in `axvline`s keep indexing the full `t_h` (both lie inside the clamp).
- The existing default-view `xlim` logic stays, but its upper bound is additionally capped at
  `t_end_h`.

## 2. dP/dG default y-max cap

In `render_gfunction`, the twin-axis default is `y2lim = (0, max(hi * 1.5, 1.0))` with `hi` the
95th-percentile of finite dP/dG. Cap it: `y2lim = (0, min(max(hi * 1.5, 1.0), 500.0))`. Default
view only; the y2 slider's full range still comes from the autoscaled extent.

## 3. Relative-min-of-dP/dG pick

New pick on the G-function step, following the app's seed-then-drag idiom:

- **Model**: `PickState.min_dpdg_G: Optional[float] = None` (replaces `eff_isip_line`, see §4).
- **Finder**: rework `interpret.suggest_min_dpdg_index(G, dPdG, g_min=1.0)` to find a *relative*
  minimum: among interior indices with `G >= g_min` and finite values where
  `dPdG[i] < dPdG[i-1] and dPdG[i] <= dPdG[i+1]`, return the one with the smallest dP/dG value.
  Fallback when no interior local min exists (monotonic decline, C-C no-contact shape): the
  current masked argmin. Same signature.
- **Seeding**: `seed_gfunction` seeds `min_dpdg_G = G[suggest_min_dpdg_index(G, dPdG)]` (replaces
  the `eff_isip_line` seeding; the contact seed at the dP/dG hump is unchanged). Early-return
  guard becomes `min_dpdg_G is not None and contact_G is not None`.
- **Render**: `render_gfunction` draws a marker on the twin axis at
  `(min_dpdg_G, interp(min_dpdg_G, G, dPdG))`: `marker="v"`, `color="tab:red"`, `ms=8`,
  `gid="min_dpdg_point"`, `label="min dP/dG"`.
- **UI**: a `DraggablePointController` on the twin axis with curve `(G, dPdG)` and gid
  `"min_dpdg_point"`, committing via new `picks.commit_min_dpdg_point(state, x)` →
  `state.min_dpdg_G = float(x)` then refresh. Shares the step's `_CaptureGate` with the contact
  controller. Hint text: "Drag the min-dP/dG marker (the effective-ISIP tangent follows it) or
  the contact marker."

## 4. Effective-ISIP tangent: derived, not editable

The effective-ISIP line stops being a stored, draggable pick and becomes a value derived from the
min-dP/dG point.

- **Model**: remove `PickState.eff_isip_line`. Add `DerivedResults.eff_isip_line:
  Optional[TangentPick] = None` (derived; not serialized). In `compute_all`, when
  `state.min_dpdg_G` is set and diagnostics/resampled exist: `idx = nearest(G, min_dpdg_G)`,
  tangent from a local fit with `half=4` (same math the old "anchor" commit used) →
  `res.eff_isip_line`, and `res.effective_isip = interpret.effective_isip(anchor, slope)`.
- **Helper placement**: move the tangent-at-index math (`_local_slope` + `_tangent_from_index`)
  from picks.py into interpret.py as `local_slope` / `tangent_from_index`; picks.py imports them
  (model.py cannot import picks.py — picks imports model).
- **Load migration**: `model._decode` maps an old save's `eff_isip_line.anchor_x` to
  `min_dpdg_G` when `min_dpdg_G` is absent (the old anchor sat on the P-vs-G curve at the same
  G). Drop `eff_isip_line` from the TangentPick-decode loop.
- **`infer_step_status`**: gfunction is "done" when `min_dpdg_G is not None or contact_G is not
  None`.
- **Render**: `render_gfunction` draws the construction from `res.eff_isip_line` (same
  `_draw_tangent_construction` call, same gids `eff_isip_segment`/`tick`/`extension`, same
  green). Condition: `res.eff_isip_line is not None and res.effective_isip is not None`.
- **UI**: remove the `AnchorLineController` for the effective-ISIP line from the gfunction step.
  No controller ever matches the eff_isip gids again.
- **picks.py**: delete `commit_eff_isip_line` (no callers remain).

## Out of scope

- The contact-point pick and its seeding (still the dP/dG hump) are unchanged.
- The tangent-method step, log-log, and pore-pressure steps are untouched.

## Tests

Update: `test_anchor_line_controller.py`, `test_commit_functions.py`, `test_seed_steps.py`,
`test_step_status.py`, `test_render_constructions.py`, `test_view_state.py`, `test_plot_colors.py`,
`test_drag_controller.py` — wherever they build or assert on `eff_isip_line` as a PickState field
or wire its controller.

New coverage:

- Overview clamp: plotted x data ends at last nonzero rate + 15 min; no clamp when rate is None.
- y2 default cap: spiky dP/dG data → y2lim[1] == 500; small data (95th pct * 1.5 < 500) keeps the
  smaller default.
- `suggest_min_dpdg_index`: picks an interior relative min over a smaller global endpoint min;
  falls back to masked argmin on monotonic data.
- `commit_min_dpdg_point` + `compute_all`: setting `min_dpdg_G` yields `res.eff_isip_line`
  anchored at the nearest sample and a finite `res.effective_isip`.
- Migration: JSON containing an old `eff_isip_line` dict loads with `min_dpdg_G ==` its
  `anchor_x` and no error.
- Render: gfunction axes contain the `min_dpdg_point` gid on the twin axis and the eff_isip
  construction gids with no controller attached to them.
