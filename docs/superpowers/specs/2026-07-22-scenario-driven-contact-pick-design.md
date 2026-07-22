# Scenario-driven contact pick on the G-function step

Date: 2026-07-22
Status: approved for planning

## Goal

Make the closure-scenario selection (C-A..C-D) on the G-function step drive the contact-point
pick, per URTeC-2019-123 and the scenario table in `plan.md`:

1. **C-A clear**: contact = first point right of the min-dP/dG pick where dP/dG has risen 10%
   above the min value.
2. **C-B adequate**: contact = the inflection point of the (monotonically declining) dP/dG curve.
3. Show the contact as a **dotted vertical line** on the G-function plot.
4. Add a toggle to overlay a **d²P/dG²** curve (to help eyeball the inflection).
5. **C-C no-contact / C-D rapid**: selecting them clears the contact pick (no Shmin(compl),
   no effective ISIP), per the plan.md table. C-D's literal-ISIP-minus-offset stress estimate is
   out of scope here.

## Background

`ui.py` already has the closure-scenario combobox (shown only on the "gfunction" step);
`_on_scenario` currently just copies the combobox value into `state.closure_scenario` and
refreshes. `seed_gfunction` (picks.py) seeds `min_dpdg_G` at the relative min of dP/dG
(`interpret.suggest_min_dpdg_index`) and `contact_G` at the global dP/dG argmax, scenario-blind.
`model.compute_all` derives contact pressure, Shmin(compliance), and the effective-ISIP tangent
from `contact_G`. The contact marker and min-dP/dG marker are draggable
(`DraggablePointController`).

Paper basis (Refs/Urtec123.md): "The contact pressure occurs when dP/dG begins to increase from
its minimum value. A rule of thumb is to pick the contact pressure when there has been a 10%
increase from the minimum." The C-B inflection rule comes from the plan.md scenario table
("monotonic w/ inflection -> contact just after inflection").

## Decisions

- The C-A rule is anchored at the **user's min-dP/dG pick** (`state.min_dpdg_G`), not a
  re-detected min, so dragging the min marker and re-selecting the scenario re-derives contact
  from the user's anchor. If `min_dpdg_G` is unset, suggest it first (existing suggester).
- Scenario selection **re-suggests `contact_G` once per selection** (an explicit user action, so
  it overwrites any previous contact pick). Afterward the marker remains draggable and no refresh
  ever overrides a drag.
- "Inflection" for C-B = the **flattest point of the declining dP/dG**: the local maximum of
  d(dP/dG)/dG over the G >= 1 region (same early-spike mask convention as
  `suggest_min_dpdg_index`). With dP/dG plotted positive-up and declining, its slope is <= 0;
  the max (closest to zero) is where the decline stalls -- the concavity transition.
- The dotted vline is drawn whenever `contact_G` is set, regardless of scenario. The existing
  black square marker stays (it is the drag handle); the vline is display-only.
- The d²P/dG² toggle is a checkbox in the closure-scenario panel frame (so it appears only on the
  gfunction step). The curve draws on the dP/dG twin axis. The flag persists in `PickState`
  (`show_d2pdg2: bool = False`) and resets on file load like the other scenario widgets.

## Changes

### 1. `resample.py` -- second derivative

Add `d2PdG2: np.ndarray` to `Diagnostics`: `np.gradient(dPdG, G)` (the slope of the positive-up
dP/dG curve). Computed in `diagnostics()` alongside the existing curves.

### 2. `interpret.py` -- suggestion functions

- `suggest_contact_clear_index(G, dPdG, min_idx, rise_frac=0.10) -> Optional[int]`:
  threshold = `dPdG[min_idx] * (1 + rise_frac)`; walk right from `min_idx`; return the first
  index where `dPdG >= threshold` (finite values only). Return `None` if the curve never rises
  10% (the shape is not C-A).
- `suggest_contact_inflection_index(G, dPdG, g_min=1.0) -> Optional[int]`:
  `d2 = np.gradient(dPdG, G)`; mask to `G >= g_min` and finite; return the index of the local
  maximum of `d2` (interior local max with the largest value, mirroring the
  `suggest_min_dpdg_index` structure). Return `None` when no interior local max exists
  (fewer than 3 usable points or strictly monotonic d2).

### 3. `picks.py` -- scenario application

`apply_closure_scenario(state, res) -> Optional[str]`: pure function called when the closure
scenario is (re)selected. Reads `state.closure_scenario`; returns an optional user-facing hint
string (shown in `hint_lbl`), and mutates picks:

- `C-A clear`: ensure `min_dpdg_G` (suggest if None); map it to the nearest diagnostics index;
  run `suggest_contact_clear_index`; set `contact_G` to the found G, or leave `contact_G`
  unchanged and return a hint ("dP/dG never rises 10% above the min -- not a clear contact")
  when `None`.
- `C-B adequate`: run `suggest_contact_inflection_index`; set `contact_G`, or hint that no
  inflection was found.
- `C-C no-contact` / `C-D rapid`: set `contact_G = None` (Shmin(compl)/effective ISIP become
  None automatically in `compute_all`).
- Empty scenario: no-op.
- Degrades to a no-op (never a crash) when `res.diagnostics` is None.

`seed_gfunction` keeps seeding `min_dpdg_G`; its scenario-blind `contact_G` argmax seed stays
(first visit, before any scenario is chosen).

### 4. `plots.py` -- `render_gfunction`

- When `state.contact_G` is not None: `ax.axvline(state.contact_G, color="black", ls=":",
  lw=1.2, label="contact G")` (display-only, no gid -- the square marker remains the drag
  handle).
- When `state.show_d2pdg2`: plot `dg.d2PdG2` on the twin axis (`ax2`), `color="tab:purple"`,
  `lw=0.9`, `label="d2P/dG2"`. The existing y2 default view clamp (95th-percentile of dP/dG,
  cap 500) is unchanged; the y2 slider reaches the rest.

### 5. `model.py` -- persistence

`PickState.show_d2pdg2: bool = False`. Rides through `to_json`/`from_json` via the existing
field filtering; no migration needed.

### 6. `ui.py` -- wiring

- Add a `ttk.Checkbutton` "show d²P/dG²" to `frm_cscen` bound to a `BooleanVar`; its command
  syncs `state.show_d2pdg2` and refreshes.
- `_on_scenario`: when the **closure** scenario combobox value changed from
  `state.closure_scenario`, call `picks.apply_closure_scenario(state, res)` after syncing state,
  show any returned hint in `hint_lbl`, then refresh. Postclosure/pp-axis handling unchanged.
- `_load` resets the new checkbox var; `_update_panel`'s widget-sync (line ~720) also syncs it
  from state on picks load.

## Out of scope

- C-D's literal-ISIP - 100-250 psi stress estimate and the `closure_quality` label plumbing.
- Any change to the tangent-method step or postclosure scenarios.
- Making the dotted vline draggable (the square marker is the handle).

## Testing / verification

- Unit: `suggest_contact_clear_index` on a synthetic S-curve (min then rise) finds the first
  +10% crossing; returns None on a monotonic decline.
- Unit: `suggest_contact_inflection_index` on a synthetic decline with a flattening finds the
  flattest point; returns None on a pure exponential-style decline.
- Unit: `apply_closure_scenario` per scenario: C-A sets contact right of min; C-B sets contact
  at the inflection; C-C/C-D clear it; empty/missing diagnostics no-op.
- Render (Agg): with `contact_G` set, `render_gfunction` draws a dotted axvline; with
  `show_d2pdg2`, the twin axis has a d²P/dG² line.
- Round-trip: `show_d2pdg2` survives `to_json`/`from_json`; old saves (missing key) default
  False.
- Manual: load the PDC Argentine CSV, walk to the G-function step, select C-A and C-B, confirm
  the vline lands per rule and the d² toggle draws.
