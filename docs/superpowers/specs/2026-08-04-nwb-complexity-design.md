# Near-wellbore complexity

Design for CLAUDE.md TODO #5.

## Goal

Report one new per-test value, near-wellbore complexity, defined as the apparent ISIP minus
the shared reference effective ISIP. It closes the additive identity

    Shmin + net pressure + complexity = apparent ISIP

Physically it is the near-wellbore friction and tortuosity that is present in the
early-decline extrapolation (the apparent ISIP) but has dissipated by the time the P-vs-G
straight line is fit (the effective ISIP).

## Reference ISIP

Complexity subtracts the same shared reference ISIP that TODO #4 established for net
pressure: the compliance effective ISIP, falling back to the tangent effective ISIP, else
undefined. There is one complexity number per test, not one per method.

That choice makes the identity close exactly for all three methods, because all three net
pressures already subtract their own Shmin from that same reference:

    net_method = ref - Shmin_method
    complexity = apparent - ref
    => Shmin_method + net_method + complexity = apparent    for method in {compliance, tangent, variable}

## Behavior

- Blank (`None`) when either the apparent ISIP or the shared reference ISIP is missing.
- Negative values are reported as-is: no warning, no clamp to zero. A negative complexity
  means the apparent ISIP came out below the P-vs-G extrapolation, which indicates a bad
  ISIP tangent pick, but the tool reports the arithmetic and leaves the judgment to the
  analyst. Clamping was rejected because it breaks the identity above.
- C-C and C-D clear the contact pick, so neither gets a compliance effective ISIP -- but the
  tangent effective ISIP still exists once the closure pick is made, so the reference falls
  back to tangent and complexity IS reported for both. `shmin_rapid` never feeds the shared
  reference, so for C-D the reported complexity is referenced to the tangent effective ISIP
  and composes with `shmin_tangent`, not with `shmin_rapid`; there is no `net_pressure_rapid`,
  so for C-D the complexity participates in no reported identity.

## Changes

### `interpret.py`

New function beside `net_pressure`:

```python
def near_wellbore_complexity(apparent_isip: float, reference_isip: float) -> float:
    """Near-wellbore complexity = apparent ISIP - reference (effective) ISIP: the
    near-wellbore friction/tortuosity that is present in the early-decline
    extrapolation but has dissipated by the time the P-vs-G line is fit. Closes
    the identity Shmin + net pressure + complexity = apparent ISIP."""
    return apparent_isip - reference_isip
```

### `model.py`

New field on `DerivedResults`, immediately after `net_pressure_isip_source`:

```python
near_wellbore_complexity: Optional[float] = None
```

`_resolve_net_pressures` already resolves the shared reference ISIP, so it sets complexity
too — one function owns the reference and everything derived from it. Its name is kept
(three tests in `tests/test_net_pressure_reference.py` call it directly); its docstring is
updated to say it resolves the reference and every value derived from it, not just the net
pressures.

```python
if ref is not None:
    if res.apparent_isip is not None:
        res.near_wellbore_complexity = interpret.near_wellbore_complexity(res.apparent_isip, ref)
    # existing per-method net pressures follow, unchanged
```

No change to `compute_all` beyond what `_resolve_net_pressures` already does. `compute_all`
sets `res.apparent_isip` before it calls the helper, so ordering is already correct.

### `ui.py`

- `PANEL_FIELDS` gains `"NWB complexity"` at row 6, directly under
  `"eff ISIP (compliance)"`, so the subtraction reads straight down the panel. The panel is
  now 19 rows; the row-count comment above the list updates.
- `FIELD_STEP["NWB complexity"] = "gfunction"`. The value needs both the isip and gfunction
  picks; gfunction is the later of the two, the same precedent `net (compliance)` follows.
- `_update_panel` adds `"NWB complexity": s(r.near_wellbore_complexity)`.

### `store.py`

- `near_wellbore_complexity` appended to the end of `LOG_COLUMNS`. Appending keeps existing
  `dfit_log.csv` files loadable, since `load_log` already backfills missing columns.
- `build_log_row` maps `res.near_wellbore_complexity` to that column. It computes nothing
  itself, per the existing contract.

## Out of scope

- No plot annotation or plot-title text for complexity.
- No interpretation-guide content.
- No per-method complexity columns (`complexity_tangent`, `complexity_variable`). TODO #4
  consolidated on a single reference ISIP; this follows it.

## Tests

New `tests/test_nwb_complexity.py`, driving `model._resolve_net_pressures` directly the way
`tests/test_net_pressure_reference.py` does:

1. Compliance reference present → complexity = apparent − compliance eff ISIP.
2. Compliance absent, tangent present → complexity = apparent − tangent eff ISIP.
3. Neither eff ISIP present → complexity stays `None`.
4. Reference present but `apparent_isip is None` → complexity stays `None`.
5. Apparent ISIP below the reference → the negative value is returned unchanged, and
   `res.warnings` stays empty (the helper appends nothing).

Plus an identity test through real `compute_all` output on synthetic data from
`tests/helpers.make_testdata`: for each method with a non-`None` Shmin, assert
`shmin_method + net_method + complexity == apparent_isip` to floating-point tolerance.

`tests/test_step_status.py::test_field_step_mapping_matches_spec` asserts against a hardcoded
`FIELD_STEP` dict, so it gains the new key. `tests/test_store.py` reads `store.LOG_COLUMNS`
dynamically everywhere and so needs no edits, but gains two tests: the new column is last in
`LOG_COLUMNS` and `build_log_row` maps it, and `load_log` backfills it for a `dfit_log.csv`
written before this change.
