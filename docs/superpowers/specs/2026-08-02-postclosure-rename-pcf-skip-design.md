# Postclosure scenario renames + PC-F skips pore pressure

Date: 2026-08-02

## Goal

1. Rename the postclosure scenario labels to the descriptive ResFrac guide titles
   (currently PC-C/PC-D are both "mixed" and PC-E/PC-F are both "none").
2. When PC-F is selected, skip the pore-pressure step entirely: no pore-pressure
   calculation, the Finish button appears on the log-log step, the pore-pressure
   breadcrumb is unreachable, and no pore-pressure PNG is exported.

## Renames

In `ui.py:POSTCLOSURE_SCENARIOS` (PC-A and PC-B unchanged):

| Old | New |
|---|---|
| `PC-C mixed` | `PC-C false radial to genuine linear` |
| `PC-D mixed` | `PC-D genuine linear to genuine radial` |
| `PC-E none`  | `PC-E no trend` |
| `PC-F none`  | `PC-F no peak` |

All scenario-driven logic (`picks.suggest_pp_axis`, `ui._PC_HINTS`) keys off the
`scenario[:4]` prefix, so behavior is label-independent. The full label is what
`PickState.postclosure_scenario` stores and what the log-log plot title shows.

**Save migration:** old picks JSON stores the full old label. `model._decode` gets an
exact-string normalization map (the four old labels → new labels) so a loaded save
matches the new combobox values. Unknown strings pass through untouched (the existing
"old or foreign JSON never raises" contract).

## PC-F skip

New helper in `model.py`:

```python
def porepressure_skipped(state: PickState) -> bool:
    """PC-F (no peak): the derivative never peaks, so no postclosure line exists and
    the pore-pressure step is skipped entirely."""
    return state.postclosure_scenario.startswith("PC-F")
```

`model` is already imported by `ui` and `plots`, so layering holds.

- **`model.compute_all`:** guard the pore-pressure block (`if state.pp_window and
  res.diagnostics is not None`) with `and not porepressure_skipped(state)`.
  `res.pore_pressure` stays `None` even if a stale `pp_window` pick exists.
- **`ui._advance`:** dispatch to `_finish()` when `self.step` equals the effective last
  step — a small helper `_last_step()` returning `"loglog"` when skipped else
  `STEPS[-1][0]`.
- **`ui._update_stepbar`:** the Next button becomes bold "Finish" on `_last_step()`;
  additionally the porepressure breadcrumb is force-disabled whenever skipped, even if
  that step was visited earlier in the session.
- **`ui._goto`:** redirect a `"porepressure"` destination to `"loglog"` when skipped.
  This centrally covers the Skip button on log-log, resume-on-load
  (`first_not_visited_step`), and any programmatic jump.
- **`ui._on_scenario`:** if the postclosure scenario just changed to PC-F while the
  current step is `"porepressure"`, navigate to `"loglog"` instead of refreshing in
  place (the scenario combobox is visible on both steps).
- **`plots.save_all_step_pngs`:** skip the `"porepressure"` render when
  `porepressure_skipped(state)`. Keep the existing enumerate numbering so the other
  filenames stay stable; `6_porepressure.png` is simply absent.
- **`ui._PC_HINTS["PC-F"]`:** reword to say the pore-pressure step is skipped and
  Finish is available on the log-log step.

Not changed: `_finish` already works from any step (it marks the current step done,
re-saves the JSON, and exports); `step_gate_error` stays as-is (reaching Finish on
log-log requires a scenario to be selected, and PC-F is one); the results panel shows
"-" for pore pressure because `compute_all` returns `None`.

## Docs

Update the postclosure table and the scenario-linkage paragraph in `CLAUDE.md`
(new labels, PC-F skip behavior).

## Tests

- Update old labels in `tests/test_pp_axis_suggest.py` (prefix-keyed, so semantics
  unchanged).
- `compute_all`: PC-F + a valid `pp_window` → `pore_pressure is None`; a non-PC-F
  scenario with the same window still produces a value.
- `model._decode`: each old label normalizes to its new label; an unrecognized label
  passes through.
- `save_all_step_pngs`: with PC-F, five PNGs and no `*porepressure*` file; without,
  six.
- Navigation (duck-typed stand-in pattern from `test_step_gate.py`): `_advance` on
  `"loglog"` with PC-F calls `_finish` and never `_goto`; with PC-A it advances to
  porepressure; `_goto("porepressure")` under PC-F lands on `"loglog"`.
