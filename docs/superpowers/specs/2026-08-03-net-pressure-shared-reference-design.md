# TODO #4 — Slim eff-ISIP sidebar, shared net-pressure reference

Date: 2026-08-03

## Goal

Two changes to how effective ISIP and net pressure are presented and computed:

1. Remove the tangent- and variable-method effective ISIP rows from the in-app sidebar.
   Keep them in the `dfit_log.csv` master log for reference.
2. Change every net-pressure calculation to reference a single shared effective ISIP
   (nominally the compliance method's), instead of each method referencing its own.

## Current behavior

- **Sidebar** (`ui.PANEL_FIELDS`): three effective-ISIP rows shown — `eff ISIP (compliance)`,
  `eff ISIP (tangent)`, `eff ISIP (variable)`. Each has a `FIELD_STEP` owner and a `vals`
  entry in `ui._update_panel`.
- **Net pressure** (`model.compute_all`, ~lines 357-370): each method references its *own*
  effective ISIP, falling back to apparent ISIP per method when that method's effective ISIP
  is unavailable. So `ref_compliance = effective_isip_compliance or apparent_isip`, etc.
- **CSV log** (`store.LOG_COLUMNS`, `store.build_log_row`): already carries
  `effective_ISIP` (compliance), `effective_ISIP_tangent`, `effective_ISIP_variable`,
  `net_pressure_compliance`, `net_pressure_tangent`, `net_pressure_variable`.

## Target behavior

### 1. Sidebar — drop two eff-ISIP rows

- Remove `"eff ISIP (tangent)"` and `"eff ISIP (variable)"` from `ui.PANEL_FIELDS`,
  from `ui.FIELD_STEP`, and from the `vals` dict built in `ui._update_panel` (leaving them in
  `vals` while absent from `value_lbls` would `KeyError` in the update loop).
- Keep `"eff ISIP (compliance)"`.
- The three `net (compliance)` / `net (tangent)` / `net (variable)` rows stay in the sidebar
  unchanged.

### 2. Net pressure — single shared reference with fallback chain

In `model.compute_all`, replace the three per-method `ref_*` locals with one shared reference
ISIP resolved by this chain:

1. `res.effective_isip_compliance` if not `None`
2. else `res.effective_isip_tangent` if not `None`
3. else `None`

No apparent-ISIP fallback (previous behavior of falling back to apparent ISIP is removed).

All three net pressures subtract their own Shmin from that shared reference, keeping the
existing per-method Shmin guards:

- `net_pressure_compliance = net_pressure(ref, shmin_compliance)` when `ref` and
  `shmin_compliance` are both not `None`
- `net_pressure_tangent = net_pressure(ref, shmin_tangent)` when `ref` and `shmin_tangent`
  are both not `None`
- `net_pressure_variable = net_pressure(ref, shmin_variable)` when `ref` and `shmin_variable`
  are both not `None`

Each net pressure stays `None` when `ref` is `None` or its own Shmin is `None`.

Record which source fed the reference in a new `DerivedResults` field:

```python
net_pressure_isip_source: Optional[str] = None
```

Set to `"compliance"`, `"tangent"`, or `""` when `ref` is `None`. `DerivedResults` is never
serialized, so no migration concern.

Note: in scenarios where the contact pick is cleared (C-C, C-D), `shmin_compliance` and
`effective_isip_compliance` are set together in the same block, so both are `None` there.
`net_pressure_compliance` and (via the contact-dependent guard) `net_pressure_variable`
therefore stay `None` in those scenarios regardless of the shared reference; only
`net_pressure_tangent` reports, referencing the tangent eff ISIP.

### 3. CSV log — keep per-method eff ISIP, add source column

- `effective_ISIP_tangent` and `effective_ISIP_variable` columns stay as-is.
- Add `"net_pressure_isip_source"` to `store.LOG_COLUMNS` and map it in `store.build_log_row`
  from `res.net_pressure_isip_source`. Appended to the end of `LOG_COLUMNS` (the schema's
  documented extension convention).

### 4. Docs

Update the CLAUDE.md domain-section net-pressure paragraph: net pressure = shared reference
ISIP − Shmin, where the shared reference is compliance eff ISIP, falling back to tangent eff
ISIP, else undefined (no apparent-ISIP fallback). Note the new `net_pressure_isip_source`
log column.

## Testing

Headless tests in `tests/` against `model.compute_all` and `store.build_log_row`:

- Compliance eff ISIP present → `net_pressure_isip_source == "compliance"`; all applicable
  net pressures reference the compliance eff ISIP.
- Compliance cleared but tangent eff ISIP present → source `"tangent"`; `net_pressure_tangent`
  computed off tangent eff ISIP; `net_pressure_compliance` and `net_pressure_variable` `None`.
- Both cleared → source `""`; all three net pressures `None`.
- `build_log_row` emits the `net_pressure_isip_source` column with the expected value.
- Sidebar assertion: `"eff ISIP (tangent)"` and `"eff ISIP (variable)"` are absent from
  `ui.PANEL_FIELDS` and `ui.FIELD_STEP`; `"eff ISIP (compliance)"` remains.

## Out of scope

- TODO #5 (near-wellbore complexity). Not touched here.
- Any change to the `net (tangent)` / `net (variable)` sidebar rows or CSV columns beyond the
  reference-ISIP change.
