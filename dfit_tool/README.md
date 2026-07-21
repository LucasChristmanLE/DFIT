# DFIT Interpretation Tool — first interactive build

Runs the DFIT compliance-method interpretation end-to-end on a single CSV, interactively.
Scope and methodology: see [../plan.md](../plan.md) and the approved build plan.

## Interpreter

Dependencies are not available on the system Python. Use the project venv:

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe

(Python 3.14; numpy, pandas, matplotlib, scipy, openpyxl. venv kept outside OneDrive on purpose —
OneDrive should not sync thousands of venv files.)

## Run

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app

Then open a CSV (e.g. the Argentine test file) and step through the workflow.

## Modules

- `io_load.py`  — CSV load, datetime parse (incl. Excel-serial fallback), channel mapping, surface→BHP.
- `gfunction.py`— G/g functions, G-time (α=1 default, α=0.5 option).
- `resample.py` — 30-psi pressure-increment resampling, derivatives, tail guard.
- `interpret.py`— te, ISIP (literal/effective), Shmin (compliance/tangent), net pressure, pore pressure.
- `picks.py`    — matplotlib interactive pickers (no Tkinter dependency).
- `ui.py`       — Tkinter/ttk shell hosting the canvas + pick panel + step bar.
- `app.py`      — entry point.

Not in this build: master results log, cross-test aggregation, batch queue/resume, PNG export,
`.DBS` loader (CSV only), permeability.
