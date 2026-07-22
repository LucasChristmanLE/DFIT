# Scenario-Driven Contact Pick Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The closure-scenario selection on the G-function step drives the contact-point pick (C-A: +10% above min dP/dG; C-B: dP/dG inflection; C-C/C-D: clear it), with a dotted vertical contact line and an optional d²P/dG² overlay.

**Architecture:** Pure suggestion math goes in `interpret.py`; the second-derivative curve is added to `resample.Diagnostics`; a pure `apply_closure_scenario` in `picks.py` maps a scenario string to pick mutations; `plots.render_gfunction` gains the vline + overlay; `ui.py` wires the combobox change and a new checkbox. Spec: `docs/superpowers/specs/2026-07-22-scenario-driven-contact-pick-design.md`.

**Tech Stack:** Python 3.13/3.14, numpy, matplotlib (Agg in tests), Tkinter shell, pytest.

## Global Constraints

- All derivative curves follow the repo's positive-up convention: `dPdG = -np.gradient(p, G)`; `d2PdG2 = np.gradient(dPdG, G)` (slope of the positive-up curve).
- Scenario strings are the combobox values from `ui.CLOSURE_SCENARIOS`: `"C-A clear"`, `"C-B adequate"`, `"C-C no-contact"`, `"C-D rapid"`; match by `startswith("C-A")` etc.
- Suggestion functions return `Optional[int]` indices into the diagnostics arrays; `None` means "shape doesn't fit the rule" and must never raise.
- Tests run headless: `python -m pytest tests/ -q` from the repo root (conftest forces Agg).
- Comment style: module/function docstrings explaining constraints, matching the existing files.

---

### Task 1: `Diagnostics.d2PdG2`

**Files:**
- Modify: `dfit_tool/resample.py` (dataclass ~line 83, `diagnostics()` ~line 95)
- Test: `tests/test_scenario_contact.py` (create)

**Interfaces:**
- Produces: `resample.Diagnostics.d2PdG2: np.ndarray` — same length as `G`, equal to `np.gradient(dPdG, G)`. Tasks 4 and 5 consume it.

- [x] **Step 1: Write the failing test**

Create `tests/test_scenario_contact.py`:

```python
"""Scenario-driven contact pick: d2P/dG2 curve, suggestion rules, scenario application,
and the G-function render additions (dotted contact vline, d2 overlay)."""

import numpy as np

from dfit_tool import resample


def test_diagnostics_has_d2pdg2():
    dt = np.linspace(0.0, 3600.0, 200)
    p = 5000.0 - 1500.0 * (1.0 - np.exp(-dt / 900.0))
    rs = resample.resample_pressure_increment(dt, p, step=5.0)
    dg = resample.diagnostics(rs, te=300.0)
    assert dg.d2PdG2.shape == dg.G.shape
    np.testing.assert_allclose(dg.d2PdG2, np.gradient(dg.dPdG, dg.G))
```

- [x] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_contact.py::test_diagnostics_has_d2pdg2 -v`
Expected: FAIL — `TypeError: Diagnostics.__init__() ... 'd2PdG2'` missing / `AttributeError: d2PdG2`

- [x] **Step 3: Implement**

In `dfit_tool/resample.py`, add to the `Diagnostics` dataclass after `GdPdG`:

```python
    d2PdG2: np.ndarray    # slope of the (positive-up) dP/dG curve: np.gradient(dPdG, G)
```

In `diagnostics()`, after `GdPdG = G * dPdG` add:

```python
    d2PdG2 = np.gradient(dPdG, G) if len(G) > 1 else np.zeros_like(G)
```

and pass `d2PdG2=d2PdG2` in the returned `Diagnostics(...)`.

- [x] **Step 4: Run the full suite**

Run: `python -m pytest tests/ -q`
Expected: all PASS (nothing else constructs `Diagnostics` positionally past `GdPdG`; the only
direct constructor call is in `diagnostics()` itself — verify with a grep for `Diagnostics(`).

> Deviation: the grep found a second direct `Diagnostics(...)` call in
> `tests/test_view_state.py::test_gfunction_y2lim_default_capped_at_500_for_spiky_dpdg`, which
> broke with `TypeError: missing 1 required positional argument: 'd2PdG2'`. Added
> `d2PdG2=np.gradient(dPdG, G)` to that call (minimal fix) so the full suite passes.

- [x] **Step 5: Commit**

```bash
git add dfit_tool/resample.py tests/test_scenario_contact.py
git commit -m "Add d2P/dG2 to falloff diagnostics"
```

---

### Task 2: Suggestion rules in `interpret.py`

**Files:**
- Modify: `dfit_tool/interpret.py` (add after `suggest_min_dpdg_index`, ~line 233)
- Test: `tests/test_scenario_contact.py`

**Interfaces:**
- Consumes: nothing new (numpy only).
- Produces:
  - `interpret.suggest_contact_clear_index(dPdG: np.ndarray, min_idx: int, rise_frac: float = 0.10) -> Optional[int]`
  - `interpret.suggest_contact_inflection_index(G: np.ndarray, dPdG: np.ndarray, g_min: float = 1.0) -> Optional[int]`
  - Task 3 consumes both.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenario_contact.py`:

```python
from dfit_tool import interpret


def _s_curve():
    """C-A shape: dP/dG dips to a min at G=5 then rises (parabola)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 50.0 + (G - 5.0) ** 2
    return G, dPdG


def _monotonic_decline():
    """C-C shape: dP/dG only ever falls."""
    G = np.linspace(0.0, 12.0, 241)
    return G, 100.0 * np.exp(-G / 3.0)


def _decline_with_inflection():
    """C-B shape: monotonic decline, steep -> flat (at G=6) -> steep.
    slope(G) = -(1 + (G-6)^2), so d2 = gradient(dPdG, G) has its interior max at G=6."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 6.0) ** 3 / 3.0)
    return G, dPdG


def test_clear_rule_finds_first_10pct_rise():
    G, dPdG = _s_curve()
    min_idx = int(np.argmin(dPdG))
    idx = interpret.suggest_contact_clear_index(dPdG, min_idx)
    assert idx is not None and idx > min_idx
    # threshold: 10% above the min value; the found sample is the FIRST at/above it
    thr = dPdG[min_idx] * 1.10
    assert dPdG[idx] >= thr
    assert np.all(dPdG[min_idx + 1:idx] < thr)
    # analytic crossing: 50*(G-5)^2 rise of 5 -> G = 5 + sqrt(5)
    assert abs(G[idx] - (5.0 + np.sqrt(5.0))) < 0.1


def test_clear_rule_none_on_monotonic_decline():
    G, dPdG = _monotonic_decline()
    min_idx = int(np.argmin(dPdG))  # the last sample
    assert interpret.suggest_contact_clear_index(dPdG, min_idx) is None


def test_inflection_rule_finds_flattening():
    G, dPdG = _decline_with_inflection()
    assert np.all(np.diff(dPdG) < 0)  # sanity: monotone decline
    idx = interpret.suggest_contact_inflection_index(G, dPdG)
    assert idx is not None
    assert abs(G[idx] - 6.0) < 0.2


def test_inflection_rule_none_without_inflection():
    G, dPdG = _monotonic_decline()
    assert interpret.suggest_contact_inflection_index(G, dPdG) is None


def test_inflection_rule_respects_g_min():
    """A flattening before g_min is masked out (early water-hammer region)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 0.5) ** 3 / 3.0)  # flattening at G=0.5 < g_min
    assert interpret.suggest_contact_inflection_index(G, dPdG, g_min=1.0) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scenario_contact.py -v -k "clear_rule or inflection_rule"`
Expected: FAIL with `AttributeError: ... has no attribute 'suggest_contact_clear_index'`

- [ ] **Step 3: Implement**

In `dfit_tool/interpret.py`, after `suggest_min_dpdg_index`:

```python
def suggest_contact_clear_index(
    dPdG: np.ndarray, min_idx: int, rise_frac: float = 0.10
) -> Optional[int]:
    """C-A "clear" contact rule (URTeC-2019-123 3.1.2): the contact is the first sample right
    of the min-dP/dG pick where dP/dG has risen ``rise_frac`` (10%) above the min value.
    Returns None when the curve never rises that much -- the shape is not a clear contact."""
    y = np.asarray(dPdG, dtype=float)
    if min_idx < 0 or min_idx >= len(y) or not np.isfinite(y[min_idx]):
        return None
    threshold = y[min_idx] * (1.0 + rise_frac)
    for i in range(min_idx + 1, len(y)):
        if np.isfinite(y[i]) and y[i] >= threshold:
            return int(i)
    return None


def suggest_contact_inflection_index(
    G: np.ndarray, dPdG: np.ndarray, g_min: float = 1.0
) -> Optional[int]:
    """C-B "adequate" contact rule: the inflection of a monotonically declining dP/dG -- the
    flattest point of the decline, i.e. the interior local maximum of d(dP/dG)/dG over
    G >= ``g_min`` (masking the early water-hammer region, mirroring
    ``suggest_min_dpdg_index``). Returns None when no interior local max exists (a shape with
    no flattening, e.g. a pure exponential-style decline)."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    if len(y) < 3:
        return None
    d2 = np.gradient(y, G)
    mask = G >= g_min
    if not mask.any():
        mask = np.ones_like(G, dtype=bool)
    interior = np.zeros(len(d2), dtype=bool)
    finite3 = np.isfinite(d2[:-2]) & np.isfinite(d2[1:-1]) & np.isfinite(d2[2:])
    local_max = (d2[1:-1] > d2[:-2]) & (d2[1:-1] >= d2[2:])
    interior[1:-1] = finite3 & local_max & mask[1:-1]
    if interior.any():
        candidates = np.where(interior)[0]
        return int(candidates[np.argmax(d2[candidates])])
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scenario_contact.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/interpret.py tests/test_scenario_contact.py
git commit -m "Contact suggestion rules: C-A +10% above min, C-B dP/dG inflection"
```

---

### Task 3: `apply_closure_scenario` in `picks.py`

**Files:**
- Modify: `dfit_tool/picks.py` (add after `commit_closure_point`, ~line 695; it already imports `interpret`, `numpy`, `PickState`, `DerivedResults`)
- Test: `tests/test_scenario_contact.py`

**Interfaces:**
- Consumes: `interpret.suggest_min_dpdg_index(G, dPdG)`, `interpret.suggest_contact_clear_index(dPdG, min_idx)`, `interpret.suggest_contact_inflection_index(G, dPdG)`; `picks._nearest(arr, value)`.
- Produces: `picks.apply_closure_scenario(state: PickState, res: DerivedResults) -> Optional[str]` — mutates `state.min_dpdg_G`/`state.contact_G`; the return value is a user-facing hint (or None). Task 5 (ui) consumes it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenario_contact.py`:

```python
from dfit_tool import picks
from dfit_tool.model import DerivedResults, PickState
from dfit_tool.resample import Diagnostics


def _res_with(G, dPdG):
    z = np.zeros_like(G)
    dg = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                     t=G, p=z, dp=z, tdpdt=z)
    return DerivedResults(diagnostics=dg)


def test_scenario_ca_sets_contact_right_of_min():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", contact_G=99.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert state.min_dpdg_G is not None and abs(state.min_dpdg_G - 5.0) < 0.2
    assert state.contact_G is not None and state.contact_G > state.min_dpdg_G
    assert abs(state.contact_G - (5.0 + np.sqrt(5.0))) < 0.1


def test_scenario_ca_uses_existing_min_pick():
    """The rule anchors at the user's (possibly dragged) min pick, not a re-detected min."""
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=6.0)
    picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert state.min_dpdg_G == 6.0  # untouched
    # threshold from the value AT the pick: (50 + 1) * 1.1 = 56.1 -> (G-5)^2 >= 6.1
    assert abs(state.contact_G - (5.0 + np.sqrt(6.1))) < 0.1


def test_scenario_ca_hints_when_no_rise():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-A clear", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0  # left unchanged


def test_scenario_cb_sets_contact_at_inflection():
    G, dPdG = _decline_with_inflection()
    state = PickState(closure_scenario="C-B adequate")
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert abs(state.contact_G - 6.0) < 0.2


def test_scenario_cb_hints_when_no_inflection():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-B adequate", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0


def test_scenario_cc_cd_clear_contact():
    G, dPdG = _s_curve()
    for scen in ("C-C no-contact", "C-D rapid"):
        state = PickState(closure_scenario=scen, contact_G=7.0, min_dpdg_G=5.0)
        assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
        assert state.contact_G is None
        assert state.min_dpdg_G == 5.0  # the diagnostic pick survives


def test_scenario_noop_cases():
    G, dPdG = _s_curve()
    # empty scenario
    state = PickState(closure_scenario="", contact_G=7.0)
    assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
    assert state.contact_G == 7.0
    # missing diagnostics
    state = PickState(closure_scenario="C-A clear", contact_G=7.0)
    assert picks.apply_closure_scenario(state, DerivedResults()) is None
    assert state.contact_G == 7.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scenario_contact.py -v -k scenario`
Expected: FAIL with `AttributeError: ... 'apply_closure_scenario'`

- [ ] **Step 3: Implement**

In `dfit_tool/picks.py`, after `commit_closure_point`:

```python
def apply_closure_scenario(state: PickState, res: DerivedResults) -> Optional[str]:
    """Re-suggest the contact pick from the just-selected closure scenario (an explicit user
    action, so it may overwrite a previous contact pick). Pure state mutation -- no matplotlib.

    Rules (URTeC-2019-123 / plan.md scenario table):
      - C-A clear: contact = first sample right of the min-dP/dG pick where dP/dG >= 110% of
        the value at that pick. Anchors at ``state.min_dpdg_G`` (suggesting it first if unset)
        so a dragged min pick drives the rule.
      - C-B adequate: contact = the dP/dG inflection (flattest point of the decline).
      - C-C no-contact / C-D rapid: no contact pick -> Shmin(compliance) and the effective
        ISIP become None downstream (model.compute_all).

    Returns a user-facing hint string when the rule finds nothing (picks left unchanged),
    else None. Degrades to a no-op when diagnostics aren't ready.
    """
    scen = state.closure_scenario
    if not scen:
        return None
    if scen.startswith(("C-C", "C-D")):
        state.contact_G = None
        return None
    dg = res.diagnostics
    if dg is None or len(dg.G) < 3:
        return None
    if scen.startswith("C-A"):
        if state.min_dpdg_G is None:
            idx = interpret.suggest_min_dpdg_index(dg.G, dg.dPdG)
            state.min_dpdg_G = float(dg.G[idx])
        min_idx = _nearest(dg.G, state.min_dpdg_G)
        idx = interpret.suggest_contact_clear_index(dg.dPdG, min_idx)
        if idx is None:
            return ("dP/dG never rises 10% above the min -- not a clear contact "
                    "(consider C-B or C-C).")
        state.contact_G = float(dg.G[idx])
        return None
    if scen.startswith("C-B"):
        idx = interpret.suggest_contact_inflection_index(dg.G, dg.dPdG)
        if idx is None:
            return "No inflection found on dP/dG -- drag the contact marker manually."
        state.contact_G = float(dg.G[idx])
        return None
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scenario_contact.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/picks.py tests/test_scenario_contact.py
git commit -m "apply_closure_scenario: scenario selection drives the contact pick"
```

---

### Task 4: `PickState.show_d2pdg2` + G-function render additions

**Files:**
- Modify: `dfit_tool/model.py` (PickState, ~line 60)
- Modify: `dfit_tool/plots.py` (`render_gfunction`, ~lines 183-233)
- Test: `tests/test_scenario_contact.py`

**Interfaces:**
- Consumes: `Diagnostics.d2PdG2` (Task 1).
- Produces: `PickState.show_d2pdg2: bool = False`; `render_gfunction` draws an `axvline` gid `"contact_vline"` when `state.contact_G` is set, and a twin-axis line gid `"d2pdg2_curve"` when `state.show_d2pdg2`. Task 5 (ui) consumes the flag.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scenario_contact.py`:

```python
import matplotlib.pyplot as plt

from dfit_tool import plots
from dfit_tool.model import PickState, compute_all  # PickState already imported by Task 3's
                                                    # block; keep one import if merging
from tests.helpers import make_testdata, overview_state


def _gfunction_fixture(**state_kw):
    td = make_testdata()
    state = overview_state(td)
    for k, v in state_kw.items():
        setattr(state, k, v)
    res = compute_all(state, td)
    assert res.diagnostics is not None
    fig, ax = plt.subplots()
    plots.render_gfunction(ax, td, state, res)
    return fig


def _gids(fig):
    return {ln.get_gid() for a in fig.axes for ln in a.get_lines() if ln.get_gid()}


def test_render_gfunction_contact_vline():
    fig = _gfunction_fixture(contact_G=1.5)
    assert "contact_vline" in _gids(fig)
    plt.close(fig)
    fig = _gfunction_fixture()  # no contact pick -> no vline
    assert "contact_vline" not in _gids(fig)
    plt.close(fig)


def test_render_gfunction_d2_toggle():
    fig = _gfunction_fixture(show_d2pdg2=True)
    assert "d2pdg2_curve" in _gids(fig)
    plt.close(fig)
    fig = _gfunction_fixture(show_d2pdg2=False)
    assert "d2pdg2_curve" not in _gids(fig)
    plt.close(fig)


def test_show_d2pdg2_round_trips(tmp_path):
    p = tmp_path / "picks.json"
    state = PickState(show_d2pdg2=True)
    state.to_json(str(p))
    assert PickState.from_json(str(p)).show_d2pdg2 is True
    # an old save without the key defaults False
    state2 = PickState()
    state2.to_json(str(p))
    assert PickState.from_json(str(p)).show_d2pdg2 is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scenario_contact.py -v -k "render_gfunction or round_trips"`
Expected: FAIL — `TypeError: PickState.__init__() got an unexpected keyword argument 'show_d2pdg2'`

- [ ] **Step 3: Implement**

`dfit_tool/model.py`, in `PickState` right after `closure_scenario`:

```python
    show_d2pdg2: bool = False  # overlay d2P/dG2 on the G-function step (helps spot the C-B inflection)
```

`dfit_tool/plots.py`, in `render_gfunction`. After the `ax2` twin-axis block (right after the
`y2lim` computation), add the overlay:

```python
    if state.show_d2pdg2:
        ax2.plot(dg.G, dg.d2PdG2, color="tab:purple", lw=0.9, label="d2P/dG2",
                 gid="d2pdg2_curve")
```

After the existing `contact_point` marker block, add the display-only vline (no drag gid
handler — the square marker remains the drag handle):

```python
    if state.contact_G is not None:
        ax.axvline(state.contact_G, color="black", ls=":", lw=1.2, gid="contact_vline")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/ -q`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/model.py dfit_tool/plots.py tests/test_scenario_contact.py
git commit -m "G-function: dotted contact vline + optional d2P/dG2 overlay"
```

---

### Task 5: UI wiring (combobox change + d² checkbox)

No automated test — `DfitApp` needs a live Tk root, which the suite never creates. Keep the
change mechanical and verify with the full suite + a syntax/import check.

**Files:**
- Modify: `dfit_tool/ui.py`:
  - `_build_body` closure-scenario frame (~line 242-248): add the checkbox
  - `_load` (~line 316-318): reset the checkbox var
  - `_on_scenario` (~line 339-343): change detection + apply + hint
  - `_load_picks` widget-sync (~line 720): sync the checkbox var

**Interfaces:**
- Consumes: `picks.apply_closure_scenario(state, res)` (Task 3), `PickState.show_d2pdg2` (Task 4), `compute_all` (already imported in ui.py).

- [ ] **Step 1: Add the checkbox to `frm_cscen`**

In `_build_body`, after the `self.cmb_cscen.bind(...)` line:

```python
        self.var_showd2 = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.frm_cscen, text="show d²P/dG²", variable=self.var_showd2,
                        command=self._on_showd2).pack(anchor="w", pady=(4, 0))
```

- [ ] **Step 2: Add the handler + rewrite `_on_scenario`**

Replace the existing `_on_scenario` with:

```python
    def _on_scenario(self):
        cscen = self.var_cscen.get()
        cscen_changed = cscen != self.state.closure_scenario
        self.state.closure_scenario = cscen
        self.state.postclosure_scenario = self.var_pcscen.get()
        self.state.pp_axis = self.var_ppaxis.get()
        hint = None
        if cscen_changed and self.td is not None:
            # Selecting a closure scenario is an explicit request to re-derive the contact
            # pick from that scenario's rule (it may overwrite a previous pick).
            hint = picks.apply_closure_scenario(self.state, compute_all(self.state, self.td))
        self.refresh()
        if hint:
            # After refresh(): _attach_controllers just set the step's default hint text,
            # and the scenario feedback must win.
            self.hint_lbl.config(text=hint)

    def _on_showd2(self):
        self.state.show_d2pdg2 = self.var_showd2.get()
        self.refresh()
```

(`compute_all` is already imported at the top of ui.py; verify, and add to the existing
`from .model import ...` import if not.)

- [ ] **Step 3: Reset on file load**

In `_load`, next to `self.var_cscen.set("")`:

```python
        self.var_showd2.set(False)
```

- [ ] **Step 4: Sync on picks load**

In `_load_picks`, next to `self.var_cscen.set(self.state.closure_scenario)`:

```python
        self.var_showd2.set(self.state.show_d2pdg2)
```

- [ ] **Step 5: Verify**

Run: `python -m pytest tests/ -q` — all PASS.
Run: `python -c "import dfit_tool.ui"` — imports cleanly.

- [ ] **Step 6: Commit**

```bash
git add dfit_tool/ui.py
git commit -m "UI: scenario selection applies contact rule; d2P/dG2 checkbox"
```

---

## Manual verification (after all tasks)

Launch via `start-app.cmd`, load
`2019.02.12_PDC_Argentine State 7170 4U B4H_Final Data/2019.02.12_PDC_Argentine State 7170 4U B4H_Final Data.csv`,
walk Overview → ISIP → G-function:

1. Select "C-A clear": contact square + dotted vline jump to the first point where dP/dG is 10%
   above the min-dP/dG pick's value; Shmin(compl)/effective ISIP update.
2. Drag the min-dP/dG marker, re-select C-A (via another scenario and back): contact re-derives
   from the dragged min.
3. Select "C-B adequate": contact lands at the dP/dG flattening; toggle "show d²P/dG²" and
   confirm a purple curve appears on the right axis and the pick sits near its local max.
4. Select "C-C no-contact": contact marker/vline disappear; Shmin(compl), effective ISIP, and
   net (compliance) read "-".
5. Save picks, reload them: scenario + checkbox state restore.
