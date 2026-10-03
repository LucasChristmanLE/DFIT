# DFIT tool: implementation notes

Detailed design history, measured corpus cases, and edge-case rationale moved out of CLAUDE.md on 2026-10-01 to keep the per-session context small. CLAUDE.md holds the architecture and invariants; this file holds the reasoning behind specific behaviors. Search it by function or file name.


Guidance for working in this repository.

## Overview

An interactive tool for interpreting a single diagnostic fracture injection test (DFIT)
by the compliance method. A Tkinter/ttk shell hosts an embedded matplotlib canvas. The
interpreter opens one data file (CSV, Fracpro `.DBS`, or an XLSX time-series workbook), maps
its channels, and walks eight workflow steps, making draggable picks on each plot. Every
reported number is derived from one pure function, `model.compute_all`.

Interpretation is one well at a time. Two ways to get there: single-file mode (open one CSV,
`.DBS`, or `.xlsx` directly, today's original flow) or folder mode ("Open Folder…"), which
scans a root folder into a queue of tests, shows that queue in a left sidebar, and rolls every
test's results up into a per-root `dfit_log.csv` master log. The sidebar and the log exist only
in folder mode; single-file mode is otherwise unchanged. There is still no cross-test
aggregation beyond that one log (no charts, no rollup stats). Permeability is out of scope.

The eight steps (`model.STEPS`): overview → injection → isip → gfunction → tangent → loglog →
porepressure → stiffness. Overview shows the entire dataset, unclamped, and hosts the always-on
tail-trim line; its y-axis is labeled "pressure (psi)", and when BHP is converted from surface
pressure it also overlays the raw surface trace (`DerivedResults.p_surface_all`, set by
`compute_all`) as a thin red line on the same axis; Injection is the zoomed injection-window view with the draggable start/shut-in
lines and the te/Vinj/qmax title; stiffness is a semilog-y relative-stiffness-vs-effective-
pressure plot (URTeC-2019-123 A.8/A.9) that needs the pore-pressure estimate, so it comes last
and is skipped end to end under PC-F exactly like porepressure (`model.skipped_steps`).

## Commands

Dependencies are not on the system Python. Always use the project venv:

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe

Python 3.14. The venv is kept outside the project on purpose so OneDrive does not sync
thousands of package files.

Run the app:

    start-app.cmd                                                   # double-click, no file loaded
    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app [path/to/file|folder]

Run the tests (from the repo root):

    C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest

`start-app.ps1` self-provisions: if the venv python is missing it creates the venv
(`py -3.14 -m venv`, falling back to `C:\Python314\python.exe`), then probes
`import numpy, pandas, matplotlib, scipy, openpyxl` and runs `pip install -r requirements.txt`
only when that probe fails. `start-app.cmd` runs the `.ps1` with
`-NoProfile -ExecutionPolicy Bypass` and keeps the window open on error.

Pinned deps (`requirements.txt`): numpy 2.5.1, pandas 3.0.3, matplotlib 3.11.1,
scipy 1.18.0, openpyxl 3.1.5, pytest 9.1.1.

## Architecture

The package `dfit_tool/` is layered. Lower layers never import higher ones.

- **Leaf math (numpy only).** `gfunction.py` is the Nolte G-function and G-time (α=1
  default, α=0.5 option). `interpret.py` is the interpretation math: te, apparent/effective
  ISIP, Shmin, net pressure, pore pressure, plus the `suggest_*` auto-pick helpers. These
  take arrays and pick parameters and return numbers. `units.py` is the unit-conversion leaf
  below both `io_load` and `questionnaire`: conversion constants, factor tables, and
  header/alias token lookups for pressure/rate/volume — no `dfit_tool` imports of its own.
- **Compute core (pure python/numpy).** `resample.py` does the 30-psi pressure-increment
  resampling, the diagnostic derivatives (dP/dG, G·dP/dG, d²P/dG², t·dP/dt), and the
  tail guard. `model.py` holds `PickState` (the serializable set of interpreter choices),
  `DerivedResults`, and `compute_all(state, td)`. `compute_all` is the single source of
  truth: it produces every reported value and every array the plots need.
- **IO.** `io_load.py` loads CSV, an XLSX time-series workbook, and the reverse-engineered
  Fracpro `.DBS` binary format (`load()` dispatches on extension, case-insensitively). CSV and
  XLSX share everything downstream of "turn the file into a raw DataFrame": `load_csv` is
  `_read_csv_frame` (the `pd.read_csv` call, including the preamble-skiprows retry) followed by
  `_finish_frame`, and `load_xlsx` builds its own DataFrame (below) then calls that same
  `_finish_frame` -- so every CSV-side fix described in this bullet (datetime-column choice, the
  FIX-A companion join, FIX B's elapsed/clock fallbacks, FIX D's reversal, the outlier guard, the
  edge-NaT-block extrapolation) applies to an XLSX load unchanged. `parse_datetime` tries two
  exact, vectorized fast paths
  (24-hour, then 12-hour AM/PM), each immediately re-filtered to the plausible 1990-2100 window
  (`_PLAUSIBLE_DT_MIN/MAX`) too -- an exact-format strptime match has no range check of its own
  (a 4-digit year field accepts ANY 4 digits), so a corrupted-but-well-shaped date, e.g. a
  typo'd year, otherwise sails through untouched by any guard at all. Measured case: Civitas
  Allred's "6 - 21011210.DTF.csv" has several implausible years (1941, 4221, 7127) that match
  the exact format perfectly and previously poisoned the reported span by billions of seconds;
  guarded (combined with the isolated-outlier guard further down, which cleans up the one
  remaining corrupted-but-plausible-range cell), it reports its real ~370.66h. The same class of
  bug, with a whole BLOCK of consecutive corrupted samples rather than a single one, still
  affects at least one other file (Arkansas 1BH's own DTF export) -- see "Not built / notes".
  Then a bare Excel-serial parse -- accepted only
  within the same plausible range (`_EXCEL_SERIAL_MIN/MAX`; a small bare number like an
  elapsed-minutes value is otherwise misread as an implausible 1900-ish date) -- and then a
  generic dateutil parse over whatever's still unparsed. That last, generic parse runs at FULL
  LENGTH (never reindexed to just the still-unparsed rows): pandas' own format-inference reads a
  shared strptime format from the array's first non-null value and applies it fast to the rest,
  so feeding it a smaller, already-more-homogeneous-garbage subset can change what gets guessed
  and therefore change a row's OWN result even though dateutil parses each row independently once
  the guess is fixed (measured case: Strathcona's `100-09-14-062-04W6-rt.csv` has a row whose
  Date cell lost its leading day digits, joined with a real Time into `"/10/2022 22:11:25"`,
  sitting among thousands of well-formed `"31/10/2022 22:11:25"`-shaped neighbors -- parsed WITH
  those neighbors, pandas guesses their shared dayfirst format and the corrupted row simply fails
  to match it (correct NaT); parsed alone, dateutil's own single-value guess reads it as a
  plausible-but-wrong `2022-10-01`). What runs BEFORE this call, not after, is blanking (to NA,
  never removing -- length must stay full) every value that could never legitimately become a
  NEW date through it anyway: a bare number `pd.to_numeric` accepts (that has exactly one
  sanctioned route to a date -- the Excel-serial fallback above -- so one that fallback already
  rejected can't sneak through dateutil's own date-from-a-bare-number guesses instead); a bare
  time-of-day/clock string with no date part at all -- not just `"9:22:20 AM"`, but also an
  Excel-mangled elapsed `MM:SS.f` reading like `"09:07.0"` (`_TIME_OF_DAY_RE`, its trailing
  fractional-seconds group written outside the optional seconds-colon group so it matches with or
  without one, and its seconds group itself `\d{1,2}` so a non-zero-padded second like `"15:10:9"`
  still counts) -- since dateutil silently defaults the missing date to TODAY, producing a
  "valid" timestamp that is really just an artifact of when the code happened to run; and a value
  with no digit in it at all, which cannot be a date under any interpretation and, left in, can
  make pandas fail to guess ANY shared format for the WHOLE array (not just fail to match one),
  falling back to a slow per-element dateutil pass over every non-null value. Leaving
  already-resolved rows unmasked (not blanked) is what preserves the Strathcona good-neighbor
  context above -- blanking them would reproduce the same "malformed row parsed with no
  context" problem a reindexed subset caused; a null entry costs nothing (skipped immediately,
  no format-guess or per-row parse charged against it), but an UNMASKED already-resolved row
  does not, and this generic parse is not even attempted at all once the two fast paths already
  own enough of the column (see the `_FAST_PATH_SKIP_FRACTION` gate just below) -- masking alone
  is not sufficient by itself to keep every shape fast; the gate is what actually prevents the
  measured worst case. Even with every value masked correctly, this generic, no-`format=` parse
  is only ever attempted at all when the two fast paths above have resolved LESS than
  `_FAST_PATH_SKIP_FRACTION` (90%) of the column's own non-empty cells -- skipped entirely
  otherwise, because pandas' format-INFERENCE can fail to ever recognize some layouts, no matter
  how many clean examples the column has. Measured case: WPX Energy's "Emma Owner DFIT
  RawData.csv", 1.73M rows, all but one exactly 12-hour-AM/PM-formatted (the exception a
  `"(date time)"` units row) -- fast path 2 resolves every real row already, but this generic
  parse (masking notwithstanding) still burns ~55s running the slow per-element path over the
  other 1,729,999 already-resolved rows, because pandas' auto-format-guess never learns the
  AM/PM layout at all, REGARDLESS of how clean the data is. Once the fast paths already own
  >=90% of a column, the remainder is -- by construction -- units/header lines or corrupt
  cells, not a second, rescuably-different valid format, so the trade is: leave them NaT (they
  already are, from the fast paths' own failure) rather than pay for a whole-column dateutil
  pass just to confirm that. Horsetail 07E-0636's own 631k-row Job Time column (a literal
  `"(date time)"` units-declaration row, misread as data, at the very start of the column, see
  FIX B below) is now resolved by this SAME skip gate (its fast-path-resolved fraction is
  ~99.9998%, comfortably past it) rather than needing the masking to keep it fast -- the masking
  still matters for a column BELOW the 90% threshold that also happens to have a no-digit row
  mixed into its genuinely-unresolved remainder. The bare-clock guard is skippable
  (`parse_datetime(..., reject_bare_clock=False)`) for exactly one caller: FIX A's companion-Time
  join below, where a blank Date field beside a real Time value intentionally joins to a
  date-less string and is meant to parse as a same-day guess (DEFECT 1a) rather than fail. A tz-
  aware result (an explicit UTC/offset marker, e.g. `"2019-08-29 14:13:35Z"` or `"...-07:00"`) has
  its offset dropped via `tz_localize(None)` rather than converted in place -- DFIT elapsed-time
  math only ever needs local wall-clock time, and pandas 3.0's unit-strict cast to
  `datetime64[us]` raises outright on a tz-aware source instead of silently converting it -- EXCEPT
  when the array mixes tz-aware values with DIFFERENT UTC offsets (e.g. `"-05:00"` and `"-06:00"`
  either side of a DST transition in the same SCADA export), where plain `pd.to_datetime` raises
  `"Mixed timezones detected"` outright, even with `errors="coerce"`, since it can't represent
  non-uniform offsets in one array without a common timezone; caught and retried with `utc=True`
  (a common UTC reference instant, offset then dropped the same way) on that specific error only,
  which is also what makes the resulting elapsed-time span correct ACROSS the transition -- a
  uniform, unchanging offset cancels out in any later subtraction either way, but a transition's
  real ~1-hour jump only shows up correctly once every value shares one reference instant. Both
  the mixed-offset retry and the tz-drop have to survive a candidate that ultimately LOSES the
  multi-candidate comparison below (every candidate gets scored, including ones this call
  rejects) without crashing the whole load, or scoring, just for being evaluated. Anything
  dateutil returns outside the same plausible 1990-2100 window is rejected too
  (`_PLAUSIBLE_DT_MIN/MAX` -- a genuinely corrupted cell, e.g. a truncated date or a typo'd year,
  that dateutil still "successfully" guesses a year-1 or year-227 timestamp for rather than
  failing outright). When more than one column name looks datetime-ish, `load_csv` picks by
  whichever scores highest on a bounded, evenly-spaced sample (capped at 5,000 non-null rows, so
  scoring several candidates in a multi-million-row file stays cheap --
  `_best_datetime_column`/`_score_datetime_candidate`): the default is always scored too, even
  when it isn't itself a candidate, and a column that is *entirely* bare time-of-day values never
  wins outright even before the per-value guard above runs (`_looks_like_time_of_day_only`,
  scored 0.0). Pairing a date-like-but-not-time-like column with its bare-`"Time"` companion (FIX
  A) is independent of which one comes first in the file: even when there's only ONE real
  candidate left after `_datetime_column_candidates` reserves the companion for its date column,
  `load_csv` still switches to it if `suggest_channels`' own order-dependent first-match had
  picked the companion instead (measured case: a bare `"Time"` column sitting before its real
  `"Date"` pair in column order) -- switching used to happen only when there was more than one
  candidate to SCORE, which silently kept the wrong single-candidate pick.

  When the datetime column parses too little of the file to trust
  (`_MIN_VALID_DT_FRACTION`, fraction computed over the column's own non-empty cells, via
  `_non_empty_mask` -- not every row in the file, so blank trailing padding rows -- some exports
  pad past the last logged sample with fully empty rows -- don't drag the fraction down despite
  every real row parsing fine), `load_csv` tries three fallbacks in order, in each case
  converting to elapsed seconds from the first valid sample (rebased so the source column need
  not itself start at 0) and synthesizing a `"DateTime"` column the same way regardless of which
  fallback supplied it: `_find_elapsed_column` for a `delta`/`elapsed`-named column with a
  recognized parenthesized unit suffix; the broader `_find_numeric_time_column` for any other
  datetime-name-matching column, trying `dt_col` itself first (the common case: the very column
  `load_csv` just failed to parse as a datetime, e.g. a bare Fracpro ASCII `Time`) and only then
  any other candidate -- unit resolved in order: the column's own header suffix -> a
  units-declaration data row (e.g. a literal `"(min)"` cell right after the header, misread as
  an ordinary data row) -> a unit token in the column's name (`Time_sec`, `Minutes`) -> else,
  for a bare `Time` header with no other hint at all, an assumed-MINUTES guess (the plain
  Fracpro ASCII convention) recorded as a `TestData.load_warnings` entry, since it's a guess,
  not a read; and, only once that broader numeric path also finds nothing, `_find_clock_column`.

  `_non_empty_mask` treats a blank cell as empty wherever it sits (leading, trailing, or
  scattered), but an Excel literal error-formula sentinel (`_EMPTY_CELL_TOKENS`: `"#REF!"`,
  `"#N/A"`, `"#VALUE!"`, etc.) only when it sits inside a CONTIGUOUS run at the very start or
  end of the column (`_edge_run_mask`) -- a SCATTERED error cell mixed into otherwise-real data
  still counts as a non-empty, invalid cell, closing what would otherwise be a hole: exempting
  every error token unconditionally would let a column that's mostly (even ~99.9%) `"#N/A"`
  pass the trust-fraction check outright. Whenever such an edge run (blank OR error-token) still
  carries real data in some OTHER column -- real rows that just happen to have no usable
  timestamp, not genuine padding -- `_extrapolate_or_warn_edge_block` runs (regardless of
  whether the fraction check above even fired) right before `load_csv` returns. "Carries real
  data" itself means at least one other column holds a finite NUMERIC value
  (`_numeric_data_bearing_mask`, `pd.to_numeric(..., errors="coerce")`), not merely a non-null
  cell -- a units-declaration row right after the header (e.g. Ballard's "Dilts 31-24 TH DFIT
  CSV Data.csv", row 2: `"(min) ,(date time), (psi), (bpm), (psi), (bbls)"`) or a blank-cells-
  plus-bare-unit-string row (e.g. Black Hills' "Cope 107 -108 16HS BHP Fracpro.CSV":
  `"                 ,               psi "`) is non-null text in every column but none of it
  numeric, so it is correctly treated as no-data padding rather than either warned about as a
  data-bearing row with no usable timestamp or extrapolated a fabricated timestamp one median
  step before the first real sample -- a plain `notna()` check (an earlier version of this
  feature) got both of those wrong. Subject to two further restrictions that can each leave
  some of the block unresolved:

  - Contiguity: only rows in an UNBROKEN data-bearing run starting right at the anchor (the last
    valid timestamp, for a trailing block; the first, for a leading one) are reachable at all --
    any row with no finite numeric value in another column (fully empty, or text only, e.g. a
    status note) breaks it, and real data sitting past
    that break stays out of reach even though it's real. Measured case: Great Western's
    "Seltzer Pump - 036HN.csv" -- the row immediately after its last valid timestamp is already
    fully empty, so the file's own real trailing 60 rows (much further out, with real Pressure/
    Rate data) are never reachable and get `"60 data-bearing rows had no usable timestamp"`
    instead; the file's reported span goes back to its true ~0.283h (just the trustworthy
    prefix), not the ~1.79h a blanket "extrapolate the whole block" reading gave in an earlier
    version of this feature.
  - Readable cells: within whatever IS reachable, only a row whose own datetime cell (and
    companion Time cell, for a FIX-A join) is genuinely blank or an error token
    (`_blank_or_error_mask`) gets a manufactured value -- a cell with real, readable text that
    simply failed to PARSE (e.g. an implausible-year date the plausible-range guard below
    correctly rejects) is a real reading, not a gap, and is never overwritten. Measured case:
    Extraction Oil & Gas's "Wake 33-20-13-...PRESSURE.csv" opens with 63 rows dated `"7/25/1987"`
    (a stuck default clock before the gauge was synced) ahead of its real 1990-2100-range data
    -- readable, not blank, so none of them get a value extrapolated backward from the first
    good sample; they're reported as `"63 data-bearing rows had no usable timestamp"` and the
    file's span comes from the good data alone (~1,199.7h).

  If the surviving good run's own sample interval is regular, whatever's both reachable and
  blank/error-token gets extrapolated at the median step (`_regular_step`), but only when ALL of
  these hold; otherwise the block stays NaT:
  - Exactly regular: >=99% of consecutive steps within 1% of the median. There is no drift or
    jitter tolerance: millisecond stamps wandering 0.87-1.0 s around a 1 s cadence are NOT
    extrapolated (an earlier drift-based branch fabricated time on blocks that were not a
    continuation, below).
  - Local slope (`_local_step_agrees`): the median step over the last 200 steps of the good run
    (first 200 for a leading block) is within 1% of the whole-run median, so a rate change
    confined to the end nearest the block is not extrapolated at the whole-run rate.
  - Column continuity (`_columns_continue`): the block must be a continuation of the same
    dataset. (a) The set of numeric-data-bearing columns (>=90% finite numeric over the window)
    in the block must equal the set in the adjacent window of the good run (the last 100 rows
    before a trailing block, the first 100 after a leading one). (b) No column may hold one
    constant value across the whole block while it varied in that window; constant padding is
    not data. A failing block gets `"N data-bearing rows were not extrapolated: their columns do
    not continue the timestamped record (<reason>)." Measured cases where the record's columns
    did not continue (all wrongly extrapolated at 053f4d9): one channel padded with a constant
    past the end of the timestamps, doubling the span (Norfolk 11-1H TDMS x3, 9-1H, 8-1H1,
    10-1H1; Bridger 10-14H2 and 44-14HS TDMS; Reno 11-10PH Lime Data); side-by-side pump and
    gauge datasets with separate timestamp columns, where the pump clock was extrapolated over
    the gauge rows (ND State 10TFH/1TFH, SM Cactus); and multi-rate TDMS "Formula Server" sheets
    (Emerald Excalibur 6-25-36H and 7-25-36H, WPX Edward Flies Away and Helena Ruth Grant), where
    the pressure column has ~5x as many rows as the timestamp column. The jittery-stamp files in
    that list (most of them) are now stopped by the exact-regularity test first and report
    `"N data-bearing rows had no usable timestamp"`; the exactly-regular Emerald/WPX ones are
    stopped by the column gate. Real spans after the fix: Norfolk 11-1H 2.78 h, Bridger 10-14H2
    0.168 h. Known cost: a block whose columns genuinely freeze at shut-in (rate and total volume
    constant after the pump stops, while they varied in the last 100 rows) also fails (b), e.g.
    the final 60 s of Civitas Bijou 1B and Titan 04 pump logs are left NaT. Crestone Peak's
    block passes (Pressure and Temp both continue; only Date Time dies), as do three 60-row
    pump-log tails whose constant columns were already constant in the window.
  The message reads (`"N rows past <time> had no timestamp (<bad value>); timestamps extrapolated at
  <step> s spacing."`, N counting only the extrapolated rows); an irregular interval, or a block
  with nothing reachable/blank-or-error at all, instead leaves everything unresolved (`"N
  data-bearing rows had no usable timestamp"`, N here counting the block's data-bearing rows,
  e.g. 60 of Seltzer's 5,419-row block) --
  never silently dropped either way, and both warnings can appear together when extrapolation
  resolves only part of the block. Measured case for a clean, fully-extrapolated block: Crestone
  Peak's "21011234 raw data.csv" -- 233,470 genuinely valid, contiguous 1 Hz "Date Time" rows,
  then ~713,000 trailing `"#REF!"` rows whose Pressure/Temp columns keep reading real, continuing
  values, ALL of them reachable and all of them the error token itself (a real, good record
  whose reported span goes from ~64.85h, the trustworthy prefix alone, to the full ~262.88h once
  the trailing block is extrapolated in full).

  `_find_clock_column` looks for a column whose non-empty values are PREDOMINANTLY (same
  `_MIN_VALID_DT_FRACTION`) bare clock/time-of-day strings and nothing else, the file's only
  time base being a wall clock (or, per `_classify_clock_mode`, an elapsed duration written in
  the same shape) with no date column anywhere -- a plain instrument/pump log, not a Fracpro
  export. The SAME `"H:MM[:SS]"` shape is genuinely ambiguous between those two readings, and
  `_classify_clock_mode` resolves it before any value is converted: any value carrying AM/PM
  settles `"clock"` outright (only a real hour takes one); otherwise a 3-field `"H:MM:SS"`
  value with any hour exceeding 23 is `"elapsed_hms"` (an elapsed DURATION -- h*3600+m*60+s,
  hour unbounded, e.g. `"39:59:00"` forty minutes before hour 40 -- a naive wall-clock reading
  would misread this as stuck at hour 23, truncating a genuine 40h record to ~24h); a 2-field,
  no-AM/PM value runs through a fixed sequence of checks, in this order ((1) and (2) are proofs,
  (3) is a heuristic): (1) the first field EVER exceeding 23 is checked first and
  settles `"elapsed_mmss"` (m*60+s, first field unbounded) outright, regardless of what the
  other checks below would have said -- too high to be even a 24-hour hour, e.g. a 40-minute
  `"00:00"`..`"39:59"` MM:SS record a naive H:MM reading would misread as a bogus ~40-HOUR span;
  (2) failing that, a midnight WRAP -- a backward step from a first field >=23 down to a first
  field of 0 -- settles `"clock"`: elapsed minutes never reset to 0 after climbing past 23, they
  just keep counting, so a genuine drop back to 0 right at the top of the hour range is proof of
  a real hour rolling over; (3) failing that, the column's own resolution vs. its sample rate --
  `_median_run_length`, the median length of each run of consecutive identical (first, second)
  pairs -- settles `"clock"` when it's >=2 (the same reading repeats across multiple rows, i.e.
  the field changes slower than the sample rate, consistent with a real clock sampled faster
  than once a minute). Rule (3) cannot separate that from an MM:SS log sampled faster than 1 Hz
  (a 2 Hz 20-minute MM:SS log also repeats each value and reads as ~20 h), so when (3) decides,
  the clock warning adds "inferred from repeated values only (low confidence)"; (4) otherwise
  `"ambiguous"`. An ambiguous candidate doesn't fail the
  whole search by itself -- `_find_clock_column` still tries any remaining datetime-name-
  matching candidate before giving up -- but if NOTHING resolves cleanly, it raises
  (`"Column 'Time': H:MM vs MM:SS ambiguous (...)"`, no override mentioned, since none exists)
  naming the first ambiguous candidate, rather than guessing either way. This decision rule has
  one known, accepted gap: a plain, non-wrapping, non-repeating no-AM/PM H:MM record sampled
  once per real minute (e.g. 13:00-18:00, one row a minute -- run length 1, no wrap, first field
  never exceeds 23) has no proof either way under rules (1)-(3) and lands on `"ambiguous"`,
  even though a human reader would call it an obvious clock, and so does a 1 Hz MM:SS log under
  24 minutes (e.g. `"00:00"`..`"19:59"`), which never proves (1); this is the boundary the rule
  as specified draws, not something patched around. No corpus file reaches rules (1)-(3): all 14
  clock-fallback files are 3-field. Every one of these readings still excludes
  Dressler's own ambiguous
  `"MM:SS.f"` shape (exactly two colon-separated fields plus a fraction, no seconds group) --
  Crescent Point's "...1secdata.csv" must keep failing here, not land on a wrong reading through
  any of them. Only the `"clock"` reading gets midnight-rollover unwrapping and reversal
  correction (the two elapsed-duration readings are already monotonic by construction, so
  neither applies): if MOST of the column's consecutive steps run backward, the row order is
  reversed first (mirroring FIX D above, but for this fallback column specifically -- FIX D
  itself only ever looks at the PRIMARY datetime column, which this fallback runs only after
  that one has already failed); `_unwrap_midnight_rollovers` then adds 24h for every backward
  step of more than 1h between consecutive finite samples (1h, not a full half-day, so a
  genuine but sparsely-sampled overnight gap -- e.g. evening readings, then the next morning's,
  nothing logged in between -- still unwraps correctly; a small backward step from ordinary
  out-of-order noise stays well under it and is left alone). The returned warning says so:
  `"Time column has clock times only (no date); elapsed time computed from the first sample,
  midnight rollovers unwrapped."` (with `" Row order was reversed (newest-first log)."`
  appended when that fired), or, for the two elapsed-duration readings, that the column holds
  durations past its usual wrap point with no unwrap applied.

  After `_parse_dt()` returns (both on the first pass and after any FIX D reversal),
  `_mask_isolated_timestamp_outliers` runs unconditionally: a single valid sample whose elapsed
  time differs from BOTH its immediate valid neighbors by more than `max(1h, 100x the median
  step)`, while those two neighbors are themselves consistent with EACH OTHER within that same
  threshold, is a corrupted timestamp -- not a genuine sudden jump -- and is set to NaT, with a
  warning giving the count. This is what tells a single bad cell apart from a REAL
  discontinuity: a genuine gap changes the level and stays there, so the sample after it lands
  close to the sample that caused the jump, not back near where the record started. Measured
  case: Strathcona's `100-09-14-062-04W6-rt.csv`, one row whose Date cell reads `"8/10/2022"`
  where every neighbor reads `"28/10/2022"` (a dropped leading digit) -- unmasked, that one
  sample's own elapsed time sits around -480h (~20 days before its neighbors, since the file's
  `t=0` reference is early in the record) and the very next sample jumps back to rejoin them,
  turning the reported OVERALL span (`max - min`) into a bogus ~571h (real max ~91h, minus the
  outlier's own ~-480h minimum) instead of the record's real ~99h.

  If NONE of the three FIX-B fallbacks finds anything, `load_csv` raises -- unconditionally,
  regardless of how many "valid" timestamps the untrustworthy column happens to have in
  absolute terms, since a low, isolated minority is not a working time base. Measured case:
  Liberty's Anderson `"DFIT-FINAL.csv"` -- an undetected preamble collapses its real Date + Real
  Time columns to anonymous `"Unnamed: N"` ones, landing `dt_col` on a plain sample-INDEX column
  instead (`"1"`, `"2"`, `"3"`, ...); roughly 1 in 11 of those integers happens to fall inside
  the plausible Excel-serial range and gets misread as a real (but wrong) date, reaching a
  nonzero valid COUNT -- 40,178 of 431,349 rows -- that a "some rows parsed" check alone would
  have accepted; the FRACTION (~9%) is what correctly flags it as untrustworthy instead,
  matching Crescent Point's Dressler file (a single genuine date surviving among thousands of
  rejected bare-clock rows) under the exact same rule -- there is no more special-cased ">=2
  valid" exemption from it.
  `suggest_channels` suggests channel roles (the pressure pick is ranked, not first-match: surface/WHP-named beats
  treating-named beats generic beats demoted aux/pump/annulus/max/avg-type channels, though a
  BHP-named channel still wins outright; when given a loaded `TestData.column` callable, it also
  filters to "live" candidates first, since a dead/backside gauge or a locked-constant channel
  can otherwise outrank a real signal by name alone). `io_load.py` also converts surface pressure
  to BHP hydrostatically
  (`BHP = WHP + 0.052·mw·tvd`, valid post-shut-in where flow → 0), and detects/converts
  per-channel units via `units.py` (see Unit detection and conversion below). `questionnaire.py`
  parses a `*questionnaire*.xlsx` next to the data file for fluid density and TVD, also using
  `units.py` for meter/kg-m3 conversions.

  **XLSX time-series workbooks.** `load_xlsx` opens with openpyxl (`read_only=True,
  data_only=True`); chart sheets are never iterated (`Workbook.worksheets` already excludes
  them, so there's nothing to skip explicitly). Each remaining worksheet is peeked (its first
  ~60 rows, plus a couple more to look past any leading label/units rows) for a header row
  (`_xlsx_find_header`): the first row with >=2 non-empty string cells, at least one matching the
  same datetime/date/time/timestamp name needles `suggest_channels` uses, no real number or
  datetime/date/time value anywhere in the row ITSELF, and real, DATA-SHAPED numeric or datetime
  content in the next couple of rows below it (`_xlsx_data_window`/`_xlsx_looks_like_data_rows`):
  pooled across those rows, at least half the non-empty cells must be numeric/datetime AND that
  data must span >= 2 distinct columns -- a plain "any cell numeric" rule (the original one) lets
  a key/value preamble line ("Label:", one number or date next to it) through, since a pooled
  50%-numeric key/value pair can still land at exactly 50% while never spanning more than one
  "value" column; requiring >=2 columns is what tells a genuine multi-channel data row apart from
  that shape even when the fraction alone would pass. The lookahead window itself skips past up
  to a few consecutive PURELY textual rows first (every non-blank cell a string, none a number or
  date) -- a units-declaration row is always one of these, but so is any other all-text row a
  real file inserts between the header and its data (a descriptive label line, or a second units
  row using a unit word this tool doesn't otherwise recognize, e.g. a bare "%") -- since real DFIT
  data never legitimately starts with a run of cells that are ALL text, even a key/value
  preamble's own one real value stops the skip immediately. A blank header cell that has real text
  directly ABOVE it is filled from that row (`_xlsx_merge_two_row_header`) before anything else
  runs, which is what recovers a genuine two-row header (channel names split across the row above
  "Date, Time, ..., FluidIdx, StageIdx") without ever overwriting a header cell that already has
  its own text. A units row directly below the header (`_xlsx_row_is_units_row` -- every non-blank
  cell short and non-numeric: a parenthesized token, a recognized unit word like "psi"/"bpm"/
  "gal"/"deg F"/"minutes", or a bare date/time-format token like "MM/DD/YY"/"HH:MM:SS") is folded
  into the header names as `"Name (unit)"` (`_xlsx_fold_unit`, stripping any parens the header
  already had) before being dropped, so the existing header-suffix unit detection (`_unit_of`)
  sees it exactly as it would from a CSV -- e.g. a bare "Time" header over a "(hh:mm:ss)" units
  cell folds to `"Time (hh:mm:ss)"`; a blank header cell with only a units-row label (Southern
  Ute's unnamed "DeltaP" column) takes that label as its name outright, unparenthesized. When more
  than one worksheet qualifies they are ranked by (time base usable, real data rows). "Usable":
  the first 2000 data rows of the sheet, run through `_finish_frame`, give a non-zero elapsed
  span (`_xlsx_sheet_time_base_usable`). Measured: Strathcona's `33P12 26th dta-Graph.xlsx` has an
  "Original Data" sheet (52,444 rows) whose Date/Time column is midnight-only with no separate
  time column, which beat the real "Cleaned Data" sheet (52,442 rows) on row count alone and
  loaded as 0 h; ranking by usability first restores 14.567 h. The ranked sheets are then tried
  in order: one whose full load raises or whose span is zero/undefined falls through to the
  next; if all fall through, a zero span raises `ValueError("... no sheet with a usable time
  base")` (this is also what makes the WPX procedure workbooks, which used to load as fake 0 h
  records, raise), otherwise the top sheet's own error is re-raised. XLSX only: CSV zero-span
  behavior is unchanged. The row count is the REAL count, taken at zip level
  (only when >1 usable sheet needs ordering; `_xlsx_zip_count_data_rows`: a streaming scan of the
  sheet XML in 4 MB blocks: gap-free, fully populated blocks are counted with C-level byte operations,
  any block with a gap/blank/self-closing/inline-string row (and the header rows) takes an exact
  per-row regex path; ~0.9 s vs 3.4 s for ElementTree `iterparse` on a 733k-row sheet, which is kept
  as `_xlsx_zip_count_data_rows_et` for a part with no plain `<row>` tags, and verified equal on every
  sheet of the xlsx corpus; a row counts when it holds a `<v>` or inline `<t>` with text, skipped row numbers count as blank, 1000 consecutive blanks stop it;
  `_xlsx_count_data_rows` is the openpyxl fallback with the same rule) -- fully parsing every
  qualifying sheet through openpyxl just to count it roughly doubled single-sheet load time. Never
  `ws.max_row`/
  `ws.dimensions`, which a worksheet's declared `<dimension>` can badly over-report (a sheet with
  669 real rows reporting `max_row == 1,047,735`, an inflated "used range" from stray formatting
  far past the last real row) or under-report (a missing `<dimension>` falls back to whatever
  openpyxl defaults to, often 1, which would otherwise rank a real data sheet below a notes sheet).
  The data read applies the same 1000-consecutive-blank stop and drops that trailing run, so a
  styled cell near row 1,048,000 no longer pads the frame with ~1M empty rows (a real data gap of
  1000+ blank rows inside a sheet truncates the read there). When that stop fires,
  `_xlsx_gap_stop_warning` scans the rest of the sheet at zip level (no stop) and, if any later
  row holds a value, adds a load warning: read stopped at a gap of >=1000 blank rows, N later rows
  were not read. No corpus file is affected.
  Data cells are converted so `_finish_frame` sees exactly what a CSV would give it
  (`_xlsx_cell_to_value`): a `datetime` cell becomes a `_PRIMARY_DT_FORMAT` string, or, when it
  carries a nonzero microsecond, the fractional-seconds fast-path format instead (so a
  millisecond-stamped gauge workbook isn't truncated to whole seconds); a `datetime.time` cell
  becomes an `"HH:MM:SS"` string (likewise `"HH:MM:SS.%f"` under a nonzero microsecond); and --
  the one XLSX-specific wrinkle -- a datetime CELL that is exactly midnight, in a column that also
  has a separate companion time column (the same pairing `suggest_channels`/`_companion_time_col`
  would find, checked against the final folded/deduped column names before any row is converted),
  is formatted date-only -- this is a per-CELL check, not a whole-column one: a non-midnight cell
  in that same column still gets the full datetime format, exactly like a CSV's own split Date +
  Time columns and so exercising the same FIX-A join either way. `parse_datetime` also carries a
  fast path for `"03-26-2018_16:44:05"`-shaped stamps (dashes, an underscore separator, no AM/PM --
  a real Caprito Petroleum corpus shape), shared with the CSV loader since both funnel through
  this one function. Trailing fully-empty rows and columns are dropped (`_xlsx_trim_trailing_empty`,
  which does not know about merged cells: a merged header cell's non-anchor columns still read as
  blank and become `"Unnamed: N"`); duplicate or blank header cells are made unique the way
  `pd.read_csv` does (`_xlsx_dedupe_headers`).

  `_companion_time_col`'s own rule (shared with the CSV loader) additionally excludes a multi-token "...Time"
  name that reads as elapsed/relative rather than a wall-clock ("Elapsed Time", "Test Time", "Delta
  Time", "Duration Time" are never eligible at all, regardless of column order) and prefers a name
  that reads as a genuine time-of-day ("Real Time", "Clock Time") over any other multi-token
  "...Time" column when more than one is otherwise eligible -- an exact bare `"Time"` column still
  always wins outright, in column order, unaffected by either rule. A name match alone is still not
  enough: `_finish_frame` (shared by both loaders) also vets the companion's own sampled VALUES
  (`_companion_is_time_of_day`, which strips each sampled value before the anchored match, so
  `"Date, Time"` CSVs with a space after the comma still join) and refuses to join it unless they
  are themselves predominantly bare time-of-day, not a second full datetime -- e.g. a "Job Time" column holding
  `"04/10/2016 12:00:00"`-shaped values beside a bare "Date" column is a real, independent
  datetime candidate in its own right, never a time-of-day pair to join with "Date".

  `sniff_xlsx_data(path) -> bool` is the cheap version of the same header rule, used by folder mode
  and `scripts/triage` to tell an actual DFIT time-series workbook apart from the hundreds of
  non-data `.xlsx` files (summaries, casing tallies, completion calcs, pump schedules, production
  tallies) a real corpus root also contains: it stops at the first qualifying sheet and never
  raises -- a corrupt or password-protected workbook just reads as "not data". It is implemented
  as a direct zip/XML peek (`_xlsx_worksheet_parts`, `_xlsx_zip_peek_rows`,
  `_xlsx_zip_shared_strings`), not by opening the workbook with openpyxl: only the first ~63 rows
  of each worksheet part's raw XML are ever stream-parsed (`iterparse`, not a full-DOM read), and
  `sharedStrings.xml` is read only up to the largest string index any peeked cell actually
  references. `_xlsx_worksheet_parts` resolves the real worksheet zip members via
  `xl/workbook.xml`'s own `<sheets>` list and `xl/_rels/workbook.xml.rels` (falling back to the
  plain `xl/worksheets/sheetN.xml` naming convention if those parts are missing or malformed),
  which is what excludes chart/dialog/macro sheets without ever opening them; each raw cell's
  type (`t="s"` shared-string, `"str"`/`"inlineStr"` inline string, `"b"` boolean, `"n"`/untyped
  number -- a date/time cell is ALSO just a plain number in the raw XML, with no style lookup
  needed since the header rule treats a number and a date identically) is converted to the same
  shape openpyxl's `values_only` rows would give `_xlsx_find_header`. Both cell gaps (a blank cell
  is usually omitted from a row's XML, not written out empty) AND ROW gaps (a fully-blank row is
  routinely omitted from the XML too) are padded back in from each cell's/row's own `r` attribute
  -- the row-gap padding matters just as much as the column one: without it, a real row's text is
  unchanged but its INDEX silently shifts earlier than its true worksheet row number by however
  many blank rows preceded it, which can land `_xlsx_find_header`'s scan on (or reject) entirely
  the wrong row purely by coincidence of how many blank rows happened to sit above it. Measured:
  ~0.18s -> ~32s scanning a real corpus folder once the openpyxl-based sniff was gated into
  `store._group_data_files`, ~10s for the same ~640-file check with this zip-peek version -- most
  of the openpyxl cost was its own per-file open, which pays for the whole worksheet part's XML
  structure regardless of how many rows are asked for. Checked against openpyxl's own
  `_xlsx_find_header`-based sniff on the full 643-file non-questionnaire corpus: every
  disagreement traced to a genuine correction from the header-tightening rule above (a completion/
  survey/summary workbook the old unbounded-lookahead rule let through, or a real Date/Time/Rate/
  Pressure job-log/pump-data sheet the old rule's narrower, non-skipping lookahead was too short to
  reach past an intervening units row), none a regression.
- **Interaction (matplotlib only, no Tkinter).** `picks.py` has the event controllers
  (`DragLineController`, `AnchorLineController`, `DraggablePointController`,
  `SpanController`, `ModifierSpanController`, `HoverCursorController`, and the
  `_CaptureGate` press arbiter), the
  pure `commit_*` functions that translate finished geometry into `PickState` changes, and
  the per-step `seed_*` functions. `plots.py` has the `render_*` renderers. `sliders.py`
  has `PanRangeSlider`.
- **Folder-mode persistence.** `store.py` is Tk-free, like `model.py`, so it is unit-testable
  headless. `scan_root` walks the ENTIRE opened root, any depth (`os.walk`, not depth-1): one
  `TestEntry` per stem group of data files in each directory it visits, keyed by a path-qualified
  `test_id` (`_entries_for_dir`) -- one `TestEntry` per immediate subdirectory holding data files
  (folder layout), plus one per loose data file or same-stem csv+dbs+xlsx group directly in the
  root (flat layout); a loose-file entry whose test_id collides with a subfolder entry is dropped
  in favor of the subfolder (with a warning attached) rather than crashing the queue on a
  duplicate iid. `TestEntry` gains an `xlsx_path` alongside `csv_path`/`dbs_path`;
  `_group_data_files` groups `.csv`/`.dbs` files by stem unconditionally but gates a `.xlsx` file
  through three more checks before it counts as a data file at all -- it must not be an Excel
  lock file (its name starts with `"~$"`, checked explicitly; NOT covered by
  `is_questionnaire_filename`, which only ever returns True for a name that also contains
  "questionnaire" -- a lock file almost never does), must not be a questionnaire
  (`questionnaire.is_questionnaire_filename`), and must actually sniff as time-series data
  (`io_load.sniff_xlsx_data`, which opens and peeks the workbook) -- since the corpus has roughly
  640 non-questionnaire `.xlsx` files that are summaries/tallies/schedules, not DFIT records.
  This makes opening a folder with many candidate `.xlsx` files somewhat slower than one with
  none (measured: `aa_DJ Basin` `scan_root` 0.44 s -> about 5.5 s on first read with 113
  candidate `.xlsx` files, 2.1 s with a warm file cache; the full-corpus 643-file sniff takes
  about 20 s cold, 8.5 s warm; each file is sniffed once via the zip-peek in
  `io_load.sniff_xlsx_data` above -- openpyxl-based sniffing measured ~32s over `aa_DJ Basin`
  before that) -- an accepted cost of the gate, not a bug.

  **test_id stability.** Adding a data `.xlsx` must never change the test_id of an existing
  csv/dbs test: an `.xlsx` whose stem matches a csv/dbs group joins that group as its XLSX source
  (`_group_data_files` already groups purely by stem, unconditionally); any other `.xlsx` becomes
  its own entry. `_entries_for_dir`'s "does this directory collapse to `test_id = rel`" decision
  counts only csv/dbs stem groups when the directory has any at all -- an xlsx-only stem group
  never participates in that count and never collapses to `rel` itself while a csv/dbs group is
  also present, so a lone csv/dbs test's id can't be pushed out to `rel/<stem>` just because an
  unrelated xlsx export was discovered alongside it. A directory with no csv/dbs files at all
  (xlsx-only) keeps the original single-total-stem-group rule, unchanged. Verified against a full
  `scan_root("C:\DFIT Data")` comparison, pre- and post-xlsx-support: all 3269 pre-existing
  csv/dbs test_ids are unchanged; the xlsx-only additions are all new ids, never colliding with an
  existing one (every `test_id` within one `_entries_for_dir` call is already unique by
  construction -- keyed on a directory's own distinct stems). Across directories, `scan_root`'s
  deeper-folder-wins dedup does NOT apply to a csv/dbs entry versus an xlsx-only entry with the
  same id (e.g. loose `W.csv` plus a subfolder `W\` holding only a data xlsx): the csv/dbs entry
  keeps the id and the xlsx-only one is renamed `<id>/<stem>` (made unique with a ` (2)` suffix if
  needed) with a scan warning. csv/dbs-vs-csv/dbs and xlsx-only-vs-xlsx-only collisions keep the
  deeper-wins rule. Against the full corpus 0 of the 3269 old ids are missing or changed; 2 extra
  renamed xlsx-only ids appear (the Berry IC 11-159HC pressure-summary workbook, under both
  `Great Western` and `aa_DJ Basin`).

  `available_sources`/`data_path` add `"XLSX"` after `"CSV"`/`"DBS"`. Picks persist to a per-test
  `<folder>/<test_id>.dfit_picks.json`, written atomically (temp file + `os.replace`), same
  contract as `PickState.to_json`/`from_json`. `status_for` derives a test's queue status
  ("new"/"in_progress"/"done"/"skipped") from its saved `PickState`: `"done"` and
  `"in_progress"` are purely derived from `step_status` -- ground-truthing it, including the
  PC-F clause (`porepressure` counts as complete under PC-F even though that scenario never
  gets a `step_status` entry for it, since the step is skipped end to end; see the PC-F section
  below) -- but `"skipped"` can also come from `state.explicit_status`, the whole-test
  Skip-test button's override, which short-circuits the derivation regardless of how far the
  steps got. `load_log`/`save_log` read and atomically write the
  per-root `dfit_log.csv`; `build_log_row` maps one test's `PickState`/`DerivedResults` into a
  `LOG_COLUMNS`-shaped row (it computes nothing itself), and `upsert_log_row` replaces-or-
  appends by `test_id`.

  **`scripts/triage`'s xlsx handling.** `features.scan_folders` groups files by well root on top
  of `store.scan_root`, so it inherits the sniff-gated `.xlsx` inclusion above; its own two
  xlsx-specific rules live in `apply.py`. A non-keeper `.xlsx` is never quarantined (or moved at
  all) when `apply.plan_moves` executes a decision -- only a non-keeper csv/dbs file is, same as
  before xlsx was ever part of a scan -- since an xlsx is additive raw-export data, routinely a
  duplicate or near-duplicate of a well's own csv/dbs pair rather than a rejected candidate the
  analyst actually reviewed as such; an xlsx the analyst DID pick as a keeper is unaffected and
  still copied to the destination well folder like any other keeper. And the per-group decision
  fingerprint (`ledger.group_files_sig`, guarding against a stale decision applying to a
  since-changed file set -- see `apply._exclusion_reason`) is computed over
  `features.sig_files_for(scan.files)`: csv/dbs files only, when the group has any at all, else
  its xlsx files -- so a same-well `.xlsx` discovered by a later re-scan never itself makes an
  already-decided csv/dbs group's decision go stale. The one exception: an `.xlsx` that the
  decision itself recorded as a KEEPER is included even when the group has csv/dbs files
  (`sig_files_for(files, keeps)`; `apply._exclusion_reason` passes the decision's keeps,
  `review_app` the keeps being committed via `_record`), so replacing the kept workbook turns
  that decision stale. None of the 8 recorded decisions keeps an xlsx, so none changed.
- **Shell.** `ui.py` (`DfitApp`) is the only Tkinter consumer. It wires the per-step
  pickers to a recompute-and-redraw loop and holds no interpretation logic. `app.py` just
  launches it, accepting either a file or a folder path on the command line.

Import graph:

    app → ui → {io_load, picks, plots, sliders, model, interpret, questionnaire, store}
    picks, plots → model, io_load, interpret
    model → interpret, resample, io_load
    resample → gfunction
    store → model, questionnaire
    io_load, questionnaire → units

`picks.py`, `plots.py`, and `sliders.py` never import Tkinter, so the whole interaction
layer runs headless under the Agg backend and the shell could be ported off Tkinter without
touching them.

Persistence is per-test JSON, via `PickState.to_json`/`from_json`. In single-file mode it is
saved and loaded through a file dialog (`DfitApp._save_picks`/`_load_picks`) to a path the
analyst chooses. In folder mode there is no file dialog for picks: `ui.py` saves through
`store.save_picks_for`/`store.load_picks_for` to the fixed per-test
`<test_id>.dfit_picks.json` next to the test's data files -- on queue navigation
(`_save_current_queue_picks`), Finish, and Skip test. `DerivedResults` is never serialized
either way. `model._decode` migrates legacy saves: it maps the old `eff_isip_line` pick to
`min_dpdg_G`, rebuilds tuples and `TangentPick`, and filters unknown keys so old or foreign
JSON never raises. `step_status` (the breadcrumb history) rides along in the same JSON;
`infer_step_status` backfills it for saves made before it existed.

On the last step (`porepressure`) the stepbar's "Next >" button becomes a bolded "Finish"
button (`ui.py:_advance`/`_update_stepbar`). One click (`ui.py:_finish`) saves picks and
writes a PNG of all step plots, in their current zoom state, to a `<stem> DFIT plots/`
subfolder next to the loaded data file. In single-file mode (`current_entry` is None) that is
byte-for-byte the original behavior: picks re-save to `<stem>_picks.json`, no log write. In
folder mode, picks save through `store.save_picks_for` instead (no `<stem>_picks.json`
duplicate), Finish also upserts and writes the current test's `dfit_log.csv` row
(`ui._write_log_row`, shared with Skip test), and then advances to the next `"new"` queue
entry via `ui._advance_queue` (or reports the queue is exhausted). The PNG export itself is
headless: `plots.render_step_figure`/`save_all_step_pngs` take no
Tkinter and replicate `ui.refresh`'s view-resolution logic (including the gfunction-specific
clamps) against an offscreen `Figure`, so `ui._finish` just resolves `self._views` into the
plain tuples that function expects.

**Folder mode.** "Open Folder…" (`ui._open_folder_path`) scans the chosen root via
`store.list_tests`, populates the sidebar queue Treeview (row iid = `test_id`), and
auto-opens the first `"new"`-status test (or the first entry if none are new). There are
exactly two ways to end a test, both in the bottom stepbar: Finish (completed it) and Skip
test (park it) -- `ui._advance_queue` is their shared auto-advance tail, scanning circularly
from just after the current entry for the next `"new"`-status one and reporting when the
queue is exhausted rather than looping forever. `"done"` and `"in_progress"` are purely
derived from `step_status` (all eight steps accounted for and none skipped is `"done"`, all
accounted for with >=1 skipped is `"skipped"`, otherwise `"in_progress"`); `"done"` is never a
manual choice. Skip test (`ui._skip_test`) is the one capability that has no other
expression: a toggle button that flags the whole test `state.explicit_status = "skipped"`
regardless of how far its steps got, saves picks, writes the log row, refreshes the queue row,
and advances -- or, when the test is already flagged (the button then reads "Unskip test"),
clears the flag, saves/logs/refreshes, and stays put. `store.status_for` checks
`explicit_status` before falling back to the `step_status` derivation. Finish clears the flag
too -- completing a test un-parks it -- but it preserves a per-step `"skipped"` written by
"Skip >" on the step it is invoked from (reachable on the last step, where `next_step` clamps,
and on `loglog` under PC-F), so a test finished with a skipped step still reports `"skipped"`.
The Source dropdown
(CSV/DBS/XLSX, `TestEntry.available_sources` in that order, among whichever of the three exist)
is enabled only when a test has more than one file available; switching sources is treated
as a different data file, so it resets that test's picks after a confirm dialog
(`ui._on_source_change`).

## Conventions and invariants

Preserve these when changing the code.

- **`compute_all` is the single source of truth.** New reported values are computed there,
  not in the UI. `ui.py` only displays what `compute_all` returns.
- **Keep `picks.py`, `plots.py`, and `sliders.py` free of Tkinter.** They are the
  headless-testable layer.
- **Renderers never set view limits.** `render_*` leave the Axes autoscaled and return a
  `ViewDefaults(xlim, ylim, y2lim)`. `ui.py` owns per-step view state (`_views`) and applies
  either the stored view or the renderer default. Do not call `set_xlim`/`set_ylim` inside a
  renderer.
- **Hit-test through own-axes pixel transforms, never `event.inaxes`.** A twin axes (e.g.
  the injection's rate `twinx`) owns `inaxes` over the shared region, so an identity check
  against a specific Axes never matches. Use `_axes_contains_pixel`/`_data_from_pixel`.
- **Slider `on_changed` callbacks must never call `refresh()`.** `refresh()` calls
  `fig.clf()`, which destroys the slider mid-drag. Callbacks only `set_xlim`/`set_ylim`,
  mutate the current `ViewState`, and `draw_idle()`. A full refresh happens only on step
  change, pick commit, Apply, scenario change, or Reset view. Hold slider references on
  `self` — matplotlib keeps no strong reference and GC otherwise kills the callbacks.
  The right/bottom-margin slider layout itself is pixel-based (`sliders.right_margin_layout`/
  `bottom_margin_layout`, `ui._layout_sliders`) rather than a fixed figure fraction, since
  tick-label pixel width doesn't scale with the window; its pixel constants scale with
  `fig.dpi / 100` (Windows display scaling raises TkAgg's dpi) — it re-runs via `set_position`/`subplots_adjust` (never
  `refresh()`) on every canvas resize and once after `_build_sliders` on each refresh, and
  each vertical slider is colored to match its line via `ViewDefaults.y_color`/`y2_color`.
- **Diagnostic derivatives are reported positive-up for a declining pressure** (negated),
  matching how G-function and log-log plots are conventionally drawn.

## Testing

Tests live in `tests/` (~22 files). `tests/conftest.py` forces the Agg backend before
`matplotlib.pyplot` is imported, so the suite runs with no display or Tk. Synthetic data
comes from `tests/helpers.make_testdata` (a DFIT-shaped `TestData`) and
`helpers.injection_state`; tests drive controllers with real synthesized `MouseEvent`s.
GUI-only paths are exercised by binding real `DfitApp` methods onto duck-typed stand-ins
rather than constructing a real `tk.Tk()`.

New logic should land in a headless-testable layer (`model`, `interpret`, `resample`,
`picks` commits/controllers, `plots` renderers) and get a test there, rather than inside the
Tkinter shell.

## Domain and methodology

Compliance-method interpretation per McClure et al. (URTeC-2019-123) and the ResFrac
practical guidelines. Refs are in `Refs/` (gitignored).

Per-test deliverables:

- **Apparent ISIP** — early BHP-decline tangent extrapolated back to the shut-in instant
  (FracPro-style construction; a manual pick on the isip step).
- **Effective ISIP** — the P-vs-G straight line extrapolated to G = 0.
- **Shmin, compliance** — contact pressure − 75 psi (`interpret.COMPLIANCE_OFFSET_PSI`).
- **Shmin, tangent** — BHP at the G·dP/dG through-origin departure (closure) point. The
  through-origin line's seed (`interpret.suggest_closure_tangent`, `picks.seed_tangent`)
  anchors at an interior local max of dP/dG (G >= `g_min=1.0`, relative prominence >=
  `CLOSURE_TANGENT_MIN_PROMINENCE`) with the largest G·dP/dG, rejecting both the early
  water-hammer spike (masked below `g_min`) and a small noise-driven local max further out on
  the curve (low prominence). Falls back to a through-origin least-squares fit over the first
  third of the `G >= g_min` samples when no candidate survives (unmasked when that segment has
  fewer than 2 finite, positive-`G` samples). Closure = the last sample within 5%
  (`CLOSURE_TANGENT_TOL_FRAC`) of the line before the first departure, walking forward from the
  tangent point -- a later re-crossing (e.g. a rising tail) is never picked up. The pick is a
  plain `DraggablePointController`: only its seeded starting position comes from this rule, so
  it stays fully draggable and a reload's saved pick is never overwritten.
- **Shmin, variable** — BHP at the G-time midpoint of the contact and closure picks. This
  third "variable-compliance" method is computed by `compute_all` and reported alongside the
  other two (Shmin, closure time, and net pressure each have compliance, tangent, and
  variable rows in the panel; effective ISIP shows only the compliance row there).
- **Shmin, Liberty** — Liberty's internal variant of the compliance method, minus 200 psi
  (`interpret.LIBERTY_OFFSET_PSI`). C-A (and blank) anchors on the min-dP/dG pick
  (`state.min_dpdg_G`); C-B anchors on the contact pick instead (the inflection) — the min pick
  there is only the inflection's seed and can sit at a nearby rel-min, not the inflection
  itself. Gated on `state.contact_G` being set, so it blanks in every state that blanks the
  compliance row (including a C-A/C-B whose contact construction failed); blank for C-C/C-D.
  Shown as the "Shmin Liberty" panel row and logged to the `Shmin_liberty` column only — it is
  not drawn on any plot and feeds no other derived value (no net pressure, no shared reference
  ISIP, no complexity).
- **Shmin, stiffness** — picked pressure − 75 psi (`interpret.shmin_compliance`), the same
  offset as Shmin compliance but a different construction: the pick sits on the 8th
  ("stiffness") step's relative-stiffness-vs-effective-pressure plot (URTeC-2019-123 A.8/A.9),
  at the upturn where the fracture walls come into contact (`interpret.h_function`/
  `relative_stiffness`/`suggest_stiffness_upturn_index`, auto-seeded then draggable). Needs the
  pore-pressure estimate (the h-function's Pres term), so it is skipped end to end under PC-F
  exactly like porepressure (`model.skipped_steps`). Comparison-only, like Shmin Liberty:
  shown as the "Shmin stiffness" panel row and logged to the `Shmin_stiffness`/
  `Shmin_stiffness_gradient` columns only — never drawn into net pressure, the shared reference
  ISIP, or complexity. `h_function` is O(n²) in the point count, so above `model.
  STIFFNESS_MAX_POINTS` (2000) — reachable now that resampling keeps rises too, which can leave
  a noisy gauge's post-shut-in record at 10k-40k resampled points — the block decimates to an
  evenly spaced subset (plus the first/last sample and the min-dP/dG pick's own sample, so those
  three are never lost to rounding) before building `p_eff`/`h`/`S`, and appends a "decimated to
  limit memory" warning; `DerivedResults.stiffness_G` carries the matching G-time subset.
- **"No slope change apparent"** (`state.stiffness_no_upturn`, a side-panel checkbox visible
  only on the stiffness step) is the explicit negative finding: some tests show no abrupt
  slope change on the relative-stiffness plot, and this records that as a legitimate outcome
  rather than endorsing the auto-seeded line — same precedent as closure scenario C-C's "no
  contact → no Shmin". Checking it blanks `shmin_stiffness` (`model.compute_all` gates the
  value, not the arrays: `stiffness_p_eff`/`stiffness_S` still compute and the curve still
  renders) and suppresses the stale-pick warning for that pick (a suppressed pick reports
  nothing, so it can't be stale). The pick itself (`state.stiffness_pick_P`) is left alone —
  unchecking restores whatever was already picked, or, if none exists yet (the step's
  first-visit seeder already ran and won't re-fire), `ui._on_stiffness_no_upturn` calls
  `picks.seed_stiffness` on uncheck so the analyst isn't left with no line and no way to get
  one. `picks.seed_stiffness` itself no-ops when the flag is set, so a reloaded save with it
  doesn't get a line parked on next visit. `render_stiffness` draws no pick vline/marker while
  the flag is set and titles the plot "Relative stiffness — no slope change apparent (Shmin
  not reported)"; the curve stays in the step's PNG export either way. The checkbox itself
  never touches `step_status` — checking it and clicking Finish walks `step_status` normally,
  the same "visited this step, then advanced" path any other step takes, no pick required.
  Only `model.infer_step_status` (the legacy backfill for saves made before `step_status`
  existed) treats the flag alone as sufficient to mark the step `"done"` on reload, since a
  pre-existing save has no walk to replay. A checked box followed by navigating away *without*
  Finish therefore leaves the step merely `"visited"`, so the whole test reads `"in_progress"`
  in `store.status_for` — correct, since the analyst hasn't actually finished the test yet, and
  distinct from the per-step Skip button, which marks the whole test `"skipped"` regardless of
  how far it got. Logged to the tail-appended `stiffness_no_upturn` column (`store.LOG_COLUMNS`)
  as `True`/`False` — except once `"stiffness" in model.skipped_steps(state)` (PC-F), when
  `store.build_log_row` logs blank instead of the stored flag: switching to PC-F after checking
  the box makes the step unreachable (so the box can never be unchecked again), and the flag is
  left alone in the picks JSON on purpose (switching back off PC-F revives it) rather than
  cleared, so the log row must blank it itself, matching every other stiffness output going
  blank under PC-F.
- **Uninterpretable G-function / tangent** follow the same negative-finding pattern.
  - Closure scenario **C-X uninterpretable** (`model.NO_CONTACT_SCENARIOS` with C-C/C-D;
    `model.closure_uninterpretable`): `picks.apply_closure_scenario` clears `contact_G`, so
    compliance Shmin/effective ISIP, Liberty, and variable blank exactly as under C-C. It also
    blanks `shmin_stiffness`, because the stiffness curve is anchored on the min-dP/dG pick; the
    stiffness arrays still compute and the curve still renders, titled "G-function
    uninterpretable (Shmin not reported)", with no pick line or drag controller. The
    min-dP/dG pick and stiffness pick stay in state. `closure_quality` logs `"uninterpretable"`.
  - **`state.tangent_uninterpretable`** (checkbox on the tangent step): `compute_all` gates the
    tangent closure, tangent effective ISIP, and variable blocks on it (`tangent_ok`). The
    `closure_G`/`closure_slope` picks stay in state; unchecking restores them, or seeds them via
    `picks.seed_tangent` if none exist (`ui._on_tangent_uninterpretable`). `render_tangent`
    draws the curves but no line or marker, and the tangent step attaches no controllers.
    `infer_step_status` counts the flag as "done". Logged as the tail column
    `tangent_uninterpretable`; `tangent_Gc` logs blank while it is set, so the kept pick never
    reads as a real closure pick.
  - Both set: no shared reference ISIP, so all net pressures and complexity are blank.
- **Near-wellbore complexity** — apparent ISIP − the shared reference effective ISIP. The
  near-wellbore friction and tortuosity that is in the early-decline extrapolation but has
  dissipated by the time the P-vs-G line is fit. Shown as the "NWB complexity" panel row and
  logged to the `near_wellbore_complexity` column.
- **Pore pressure** — intercept of the late-time postclosure line on the t^(−1/2) or t^(−1)
  axis chosen by the postclosure scenario.
- **Pressure gradients** — depth-normalized (psi/ft) form of a pressure, `value / state.tvd_ft`
  (`interpret.pressure_gradient`, `model._resolve_gradients`). Three rows in the panel (apparent
  ISIP grad, Shmin compliance grad, pore pressure grad, each inline under its parent) plus five
  more logged only to `dfit_log.csv` (Shmin variable/tangent/Liberty/rapid/stiffness gradient) —
  eight columns total. `tvd_ft` must be not-None, finite, and `> 0`, or every gradient blanks to
  `None`/`"-"`; nothing upstream enforces that (`io_load.bhp_inputs_ready` accepts `tvd_ft =
  0.0`), so this guard is the only thing standing between the feature and a divide-by-zero.
  Every gradient field is strictly its own source value / TVD — no cross-field fallback in the
  model layer, so `Shmin_compliance_gradient` stays blank under C-D exactly as `Shmin_compliance`
  does, and `Shmin_rapid_gradient` is what carries that case in the CSV. The panel's rapid
  substitution (below) is a `ui.py` display choice on top of that, not a property of these
  fields. An unusable `tvd_ft` (missing, non-numeric, zero, negative, or non-finite) is never
  silent once there's something to report: `_resolve_gradients` appends a warning naming the
  problem ("TVD not set..." or "TVD `<value>` is not a positive number..."), gated on at least
  one of the eight source values being non-None so a freshly-opened test with no picks yet
  stays quiet. It deliberately coexists with the separate "Surface pressure selected but
  density/TVD not set" warning (`compute_all`) rather than being suppressed by it — that one is
  about BHP reliability, this one is about the gradients not being reported, and both land in
  the same collapsed warnings count (see the Tail trim section below for the display itself).

**Net pressure** = shared reference ISIP − Shmin. All three methods (compliance, tangent,
variable) subtract their own Shmin from one shared reference ISIP: the compliance effective
ISIP, falling back to the tangent effective ISIP, else undefined (no apparent-ISIP fallback,
so a net pressure is blank when neither effective ISIP exists or that method's Shmin is
absent). `compute_all` records the source that fed the reference in
`DerivedResults.net_pressure_isip_source` ("compliance"/"tangent"/""), logged to the
`net_pressure_isip_source` column of `dfit_log.csv`. The tangent and variable effective ISIPs
are kept in the CSV log but are no longer shown in the sidebar panel. Near-wellbore complexity
subtracts that same shared reference from the apparent ISIP
(`interpret.near_wellbore_complexity`), which closes the identity `Shmin + net pressure +
complexity = apparent ISIP` for all three methods. It is one value per test, not one per
method, set by `model._resolve_net_pressures` alongside the net pressures and guarded on the
apparent ISIP being present. A negative value is reported as-is -- no warning, no clamp --
since clamping would break the identity. C-C and C-D clear the contact pick, so neither gets
a compliance effective ISIP -- but once the closure pick is made the tangent effective ISIP
still exists, so the shared reference falls back to tangent and complexity IS reported for
both. `shmin_rapid` never feeds the shared reference, so for C-D the reported complexity is
referenced to the tangent effective ISIP and composes with `shmin_tangent`, not with
`shmin_rapid` -- there is no `net_pressure_rapid`, so for C-D the complexity participates in
no reported identity.

**Resampling.** After shut-in, keep one (time, pressure) point each time BHP has moved >= 30 psi
in EITHER direction from the last kept point -- a decline of >= 30 psi is kept exactly as
before, and so, now, is a rise of >= 30 psi (a water-hammer rebound, a late tail rise). This
collapses ~10⁶ raw rows to a few hundred, dense whenever the pressure is moving quickly and
sparse when it's flat, which stabilizes the numerical derivatives. It replaces time-domain
smoothing. The resampled record is no longer monotonically decreasing -- a rise shows up in
`res.resampled.p` too. A tail guard flags a default cutoff once the pressure sustains a rise
above its running minimum (non-monotonic late data) for long enough, and in enough samples, to
rule out noise -- see Tail trim below for the exact thresholds and how an explicit override can
move the cutoff past it; the guard's own detection is unaffected by the keep-rule change (it
still runs off a plain cumulative min over finite samples, independent of what gets kept).

`resample.resample_pressure_increment` is vectorized the same way `detect_dropouts` is: the
original per-sample loop is kept, unchanged, as `resample._resample_loop_reference` (a fallback
for a non-finite/non-positive `step` or a non-finite/negative `rise_tol`, and the thing
`tests/test_resample_vectorized.py` fuzzes the fast path against). Tail-guard detection vectorizes
exactly as it always did: the loop's `running_min` is provably a plain cumulative min over the
finite samples (`np.minimum.accumulate`, non-finite mapped to `+inf` so it never lowers it), and
the tail-guard runs are found the same way any run-length problem vectorizes -- label each
above-tolerance run's start position, forward-fill it across the run with
`np.maximum.accumulate`, and check the sustain conditions at every above-tolerance sample in one
pass; the earliest sample anywhere that satisfies both wins, matching the loop's fixed-once
`guard_dt`/`guarded_at`. The kept walk itself has no such shortcut any more: the old version
exploited the fact that a kept sample's value was always exactly the running min at that instant,
so it could search a monotone array with a single `np.searchsorted`; a bidirectional (either
direction) keep rule has no monotone structure to search, since a run sample can now be kept too
(it just can't lower the guard's own `running_min`). Instead, from the current kept sample the
next one is the first later sample with `abs(p - last_kept) >= step` (a NaN comparison is always
False, so a non-finite sample is skipped exactly like the loop's `continue`), found with a
doubling window (`p[c+1 : c+1+w]`, `w` starting at `max(64, 2 * previous_gap)` and doubling until
it hits or runs off the end) rather than one vectorized pass over the whole record: most gaps
between kept points are small, so the window usually hits on the first try, and the total numpy
work across all windows is O(n) amortized while the Python-level loop still runs once per OUTPUT
point (a few hundred), not once per raw row -- except on a pathological input where nearly every
sample swings >= step, which degrades toward one Python iteration per raw row.
`stop_at_guard=True` then drops any kept index at or past the run's first sample (`s_abs`) --
unlike before, a run sample CAN be kept under the new rule as the excursion climbs, so this is a
real discard, not a no-op, and it's what keeps the historic guarantee that nothing at or past
`guard_dt` is ever consumed. On a synthetic 1.5M-sample noisy decline this keeps the function well
under 1 s (previously ~0.05 s for the old, strictly-decreasing walk).

**No-rate fallback.** When a dataset has no rate channel (or a dead one that never exceeds
the detection threshold), `picks.seed_injection` seeds the start/shut-in vlines from the
pressure shape instead (`interpret.suggest_injection_window_pressure`: shut-in at the
pressure max, start at the last upcross of a 10%-of-rise threshold, non-finite samples
ignored so dropouts can't fake an upcross, positional defaults for degenerate shapes), so the
draggable lines always exist. `picks.seed_overview` delegates to the same seeder on the
Overview step's first visit, so the reference lines exist there too. `compute_all` sets
`t_shutin_s` from the picks alone and, when the effective te (Vinj/qmax) is unavailable,
falls back to te = wall-clock pump duration (shut-in − start) with an appended warning. Vinj
and qmax stay blank without rate; the Injection title shows only the pieces that exist. When a
rate channel does exist, the rate-based seed (`interpret.suggest_injection_window`) splits the
rate-on samples into contiguous runs and first drops runs whose median rate exceeds
`INJECTION_MAX_PLAUSIBLE_BPM` (150), unless every run does (then the channel is more likely in
the wrong unit). It sizes the rest all by volume gain (or all by summed rate when there is no
volume channel or any run's gain is non-finite or non-positive), drops any run smaller than
`INJECTION_MIN_RUN_FRAC` (10%) of the largest run's size, and returns the last surviving run
(start = its first sample, shut-in = one past its last). This skips both an earlier breakdown
pulse or step-rate cycle and a trailing post-shut-in rate blip, landing on the main, sustained
injection. Cream 2C-21HZ (DJ Basin, `.DBS`) is the ceiling's case: a fill/prime reads a flat
~432 on "Slurry Flow Rate" for 4.6 min at 15 psi surface, then a pressure test to ~3,900 psi;
its summed rate was 15x the real ~10 bpm injection's, so it won the seed. The rate-axis default
(`plots._rate_y2lim`) also leaves out samples above 150 bpm, unless every sample is that high, so
such a fill doesn't flatten the real rate trace; the slider's outer range is the twin's own
autoscale, so it still reaches them.

An absolute floor (last run >= N bbl, whatever the earlier runs' size) was tried and rejected
(2026-10-03). Over the 2,077 corpus files with a rate channel it moved 112-190 seeds for
N = 1-10 bbl. Many records carry a 1-40 bbl post-shut-in run (rate noise at ~0 psi during the
falloff, a small top-up), and on 12 plotted files the floor picked wrong on 6 and right on 1.
Still open: a prime at a *plausible* rate (WRP Anderson 18-3-11HC: ~90 bpm at ~0 psi before
the real injection) wins on size; pressure, not volume, separates it. A brief rate drop also
splits one injection into two runs (WRP Sharp 24-3-11HC, Continental Maryland 2-16H), and the
seed then ends at the first run's end.

**Tail trim.** The tail guard only catches a late rise, and now only a *sustained* one: it
fires when a run of samples stays continuously more than a fixed 30 psi
(`resample.RISE_GUARD_PSI`, independent of the resample step) above the running minimum for
both >= 60 s and >= 5 samples (`resample.RISE_GUARD_SUSTAIN_S`/`RISE_GUARD_SUSTAIN_SAMPLES` --
the sample-count floor matters at coarse (>= 60 s) sample spacing, where duration alone would
already be satisfied by the run's second sample), so a brief water-hammer rebound or noise
spike no longer trips it. A non-finite sample mid-run resets the run rather than being skipped
through it -- continuity can't be confirmed across a dropout, so two excursions separated by
missing data can't bridge into a false fire. `guard_dt`/`guarded_at` are pinned to the FIRST run
that satisfies both sustain conditions -- a second, later qualifying excursion further out in the
same record never overwrites them, regardless of `stop_at_guard` (below). When the guard does
fire, the excluded raw tail is drawn as a faint gray preview capped at 2x the G-range that was
actually kept BEFORE the guard fired (the last `resampled_full` sample strictly before `guard_dt`,
not the whole record -- `stop_at_guard=False`, below, can extend real kept samples well past
`guard_dt`, which would otherwise loosen this cap considerably) on the G-function plot alongside a
warning inserted at the front of `DerivedResults.warnings` (not appended), so it's the first line
if the analyst expands the list -- `ui.py`'s `warn_lbl` (under the Notes box) collapses the whole
warnings list to a one-line count by default (`"N warnings (click to expand)"`,
`ui.format_warnings_text`), expanding in place on click (`DfitApp._toggle_warnings`) and
re-collapsing on the next recompute -- rather than getting buried below an earlier-queued warning
once expanded. A
monotone crash to ~0 psi (gauge pulled, well opened) still sails through the guard untouched and
pollutes the derivatives -- the Overview step's tail-trim line is the defense against that.

The trim line is always on: there is no more "Show trim tool" toggle. `picks.seed_tail_trim`
(called explicitly from `ui._seed_step`, not a `SEEDERS` entry -- it needs a `res` recomputed
*after* the injection window is seeded) parks-and-applies a default cut on the Overview step's
first visit, via `interpret.suggest_tail_trim_dt(dt_post, p_surface_post, guard_dt)`: the
earliest of the rise-guard boundary (`res.resampled_full.guard_dt`) or the first post-shut-in
sample where surface pressure drops below 100 psi (`interpret.MIN_SURFACE_PRESSURE_PSI`; skipped
when the mapped channel is already BHP, where a sub-100-psi test is meaningless), tie going to
the rise guard. The low-pressure candidate is the collapse **onset**, not that first sub-floor
sample k: over the raw samples in `[dt[k] - lookback, dt[k])`, lookback = min(300 s
(`interpret.TAIL_ONSET_LOOKBACK_S`), 2% of dt[k] (`TAIL_ONSET_LOOKBACK_FRAC`)),
`p_ref` is the finite max and the cut is the dt of the first sample after the LAST one with
`p >= p_ref - onset_tol_psi` (`seed_tail_trim` passes `state.resample_step`, default 30). With no
window samples or none qualifying the cut stays `dt[k]`. The window bounds how much real decline
it can remove; the 2% term keeps an early crash from eating the steep early decline (a crash
within minutes of shut-in is a bad test anyway, so it simply gets almost no back-off). On a
134-test corpus sample (88 with a >= 1 h falloff and no crash in the first hour), 22 trims moved
earlier, 1-23 s except one 174 s. Arkansas 1BH MERGED.DBS: surface pressure holds 805.7 psi, collapses
802 -> 575 psi over 12 one-second samples, then 268 -> 32 -> 13 psi. Cutting at the first
sub-100 sample (774443.04 s) left the whole 805 -> 268 collapse in, and the 30-psi resampler kept
7 points on it within dG ~ 1e-3 (dP/dG 2.5e5-3.7e6), which the hump suggester took as the
contact hump. The onset cut is 774431.04 s (Gmax 126.37 -> 120.07). A rise-guard boundary sets **no pick at all** (`PickState.tail_trim_dt` stays
`None`) -- the guard boundary is already the effective cutoff by default
(`interpret.resolve_tail_cut_dt`, below, with no trim in state), so only the rendered line
position and gray-out need to reflect it, not a stored trim; a sub-100-psi crash
snaps to the last **raw** post-shut-in sample **strictly before** the cut and sets
`PickState.tail_trim_reason = "low_pressure"` (logged to `tail_trim_reason`, appended after
`tail_trim_s` in `LOG_COLUMNS`). Raw, not the last full-resample (30-psi kept) sample: on a slow
late falloff the resampler keeps nothing for hours, so the last kept point before the crash can
sit far early (AEF 05-61-34-5649B DFIT.DBS: kept points at 204.6, 251.9 and 290.96 h, crash at
290.96 h, 678 -> 661 psi over the last 39 h; the old kept-point snap trimmed at 251.9 h). The kept
points at or before the trim are identical either way, so no diagnostic changes; only the line,
gray-out, warning text and logged `tail_trim_s` move. Strictly-before matters: the crash cliff is
almost always kept *and is usually the last point kept*, so it must be excluded. The seeder BAILS
(sets no pick) rather than clamps when the trim would sit before `dt_full[2]` (fewer than 3 kept
points would remain; clamping up to `dt_full[2]` could sit at or past the crash and keep a
sub-floor sample inside a record the "Tail auto-trimmed" message claims is clean), or when
`dt_full[-1] < cut_dt` (crash beyond where resampling reached, nothing in the record to remove). Both bail cases are still covered by the separate low-surface-pressure
warning (which scans raw, not kept, samples), so neither goes silent -- setting no trim is
strictly better than setting a wrong one. Neither candidate existing (clean record) leaves the
line parked at the end of the data, nothing trimmed. The seeder is non-destructive (a pre-existing
trim, e.g. from a reloaded save, is left alone).

Dragging the line (`DragLineController`, gid `"tail_trim"`, in time-domain hours, its own private
gate since nothing else is draggable on that axes) commits `PickState.tail_trim_dt`
(shut-in-relative seconds, `None` = no trim) via `picks.commit_tail_trim`, which always resets
`tail_trim_reason` to `""` -- a manual drag or clear overrides whatever auto-attribution put the
trim where it was. `compute_all` resamples the full post-shut-in record via
`resample.resample_pressure_increment(..., stop_at_guard=False)`, so `DerivedResults.resampled_full`/
`G_full` always span the whole record regardless of whether the rise guard fired -- `stop_at_guard`
(default `True`, preserving the historic hard-stop for `tests/test_rise_guard.py`'s direct-call
tests and any other caller that still wants the old "guard ends the record" behavior) still
detects `guard_dt`/`guarded_at` from the FIRST run that satisfies both sustain conditions, pinned
there for good (see above) regardless of which mode is active. What `False` changes is what
happens at the moment of firing: instead of truncating and stopping, the run simply resets
(exactly like an ordinary run that dips back below tolerance on its own) and resampling
continues, still relative to the ORIGINAL running minimum -- never reset, never frozen at a
fresh anchor. Under the bidirectional keep rule this is no longer a no-op for the excursion
itself: the run's own samples are evaluated by the same `abs(pi - last_kept) >= step` rule as
any other sample, so the excursion's climb is captured as it happens, right through the region
the guard flagged -- this is the actual point of `stop_at_guard=False`. A later genuine decline
below the true historical minimum is still picked up completely normally on top of that. There
is no monotonicity invariant left to reason about -- kept points can go up or down -- so unlike
the old strictly-decreasing rule, a permanently elevated tail (a stuck sensor or an ongoing leak)
is no longer resampled down to nothing: its climb gets captured, and it only stops producing new
kept points once it flattens out -- though a flat jump still typically earns exactly ONE kept
point of its own (the jump into the elevated level, if it clears >= step off the last pre-guard
kept point; with the default `resample_step` equal to the guard's fixed 30-psi `rise_tol`, an
excursion large enough to fire the guard almost always clears this too), and nothing further
once it's genuinely flat -- there's no new information to resample there after that. `compute_all` then masks to the cutoff
returned by
`interpret.resolve_tail_cut_dt(state.tail_trim_dt, guard_dt, state.tail_guard_override)` before
computing diagnostics (see below) -- so the trim propagates to every downstream value (effective
ISIP, Shmin, log-log, pore pressure) with no other plumbing. Whenever `tail_trim_dt` is set, `compute_all` also emits an
explanatory warning (`insert(0)`, same front-of-stack treatment as the guard warning) so an
auto-applied trim is never silent: for `"low_pressure"`, `"Tail auto-trimmed ... surface pressure
crashes below 100 psi shortly after this point. Drag the Overview trim line to the right edge to
undo."` -- worded to point at the crash beginning just past the cut, not at the cut itself, since
by construction of the `side="left"` snap the trim dt is a sample where pressure is still ABOVE
the floor -- with an appended clause, `" Doing so also overrides the tail guard, which fires
further into the tail, extending the cutoff to the end of the record."`, whenever
`resampled_full.guard_dt` is also set for this record (a guard existing at all here necessarily
fires later than this low-pressure cut, per `suggest_tail_trim_dt`'s earliest-wins rule), so
"drag to the right edge" doesn't read as a plain undo when it also overrides a fired guard; else
`"Tail trimmed ... (N raw samples excluded)"` for a manual trim (see the override-related
exception to that message in "Overriding the guard" below). This matters
because the seeded trim usually *clears* the separate "Surface pressure fell below 100 psi"
warning below (its mask is narrowed by `tail_trim_dt`) -- without this line an auto-applied trim
would otherwise report nothing having changed.

**Resyncing shut-in moves (`picks.resync_auto_tail_trim`).** `tail_trim_dt` is shut-in-relative,
but the shut-in pick can move on the Injection step after the trim was seeded on Overview.
Dragging shut-in later by delta moves the auto trim's absolute-time cut delta later too (nothing
else re-derives it), quietly re-admitting whatever crash it was supposed to exclude while the
"Tail auto-trimmed" warning keeps claiming the crash is handled. `ui.py`'s Injection controller
wiring calls `resync_auto_tail_trim(state, td, res)` after committing a shut-in drag (never a
start-only drag: `tail_trim_dt`/`guard_dt` both live in dt-from-shut-in space, and
`resample_pressure_increment` takes only `(dt, p)` built from shut-in onward, so a start-only
change can't move either). The resync is a no-op unless `tail_trim_reason == "low_pressure"` --
a `""` reason means no trim or the analyst's own manual pick, neither of which resync ever
touches -- otherwise it clears the stale pick and re-runs `seed_tail_trim` against the new window,
which correctly sets no trim at all if the new window turns out to have no crash.

**Overriding the guard (`PickState.tail_guard_override`, `interpret.resolve_tail_cut_dt`).**
`resampled_full` spans the whole record (`stop_at_guard=False`, above) whenever the excursion's
own climb, or a genuine further decline, moves >= step off the last pre-guard kept point, but
resampled_full correctly keeps NOTHING at or past `guard_dt` only when no sample anywhere past
the guard -- not even the excursion's own first sample -- ever clears that +-step move off the
last pre-guard kept value (reachable when `resample_step` is configured larger than the
excursion's height above it), so there may be no resampled sample there to snap to at all. The override therefore lives in `ui.py`'s Overview
drag-commit closure (`_attach_controllers`), not in the resampler: when the drag target is at or
before `guard_dt` (or there's no guard), it snaps to the nearest raw post-shut-in sample among
only those strictly before `guard_dt`, floored at the third kept point (`>= 3` kept points) and
cleared when released at the last candidate. Raw, not `resampled_full.dt`: kept points can be
hours apart on a slow falloff, which left a drag only two places to land (AEF 05-61-34-5649B).
Restricted to `< guard_dt` so a drag at/before the guard can never land past it and silently set
`tail_guard_override`. When the drag target is PAST `guard_dt`, it also snaps against the RAW post-shut-in
samples (`self.td.t_s` from shut-in onward) -- resolvable unconditionally regardless of whether
the resampler found any new points past the guard, since a raw sample always exists to snap to
even when `resampled_full` doesn't. That said, "resolvable" is not "meaningful": the override
always changes the DISPLAY (the gray-out boundary, and which of the two guard-related warnings
shows, below) but only changes the actual `res.resampled`/`res.diagnostics` numbers -- and
therefore Shmin/effective ISIP/pore pressure -- when the resampler genuinely had something new to
offer past the guard. With the bidirectional keep rule this is now the common case for a real
tail rise (the excursion's own climb usually moves >= step off the last pre-guard kept point,
same as a later decline below the pre-guard historical minimum would). Only when no sample
anywhere in the overridden range clears that +-step move off the last pre-guard kept value does
dragging past the guard move the line and the gray-out while leaving every reported number
exactly as the guard-clamped default, and `compute_all` says so explicitly (see the
warning-suppression
paragraph further down). Critically, the raw-snap branch never clears the trim to `None`, even when the drag lands
on the raw record's very last sample -- it always sets an explicit `tail_trim_dt` there instead
(functionally "include everything," since nothing exists past it to exclude), because clearing to
`None` would fall back through `resolve_tail_cut_dt` to `guard_dt` again, defeating the whole
point of the override. This is the actual bug fix: previously (an earlier, reverted version of
this feature) a drag past a permanently-elevated guard had nothing to snap to, clamped to the last
resampled index, hit the "released at/past the end -> clear the trim" branch, and the line snapped
right back to `guard_dt` on the next render no matter where the analyst dropped it.

A set `tail_trim_dt` sitting past `guard_dt` is still ambiguous on its own -- it's either that
deliberate drag, or a stale auto-trim left behind by a shut-in move before
`resync_auto_tail_trim` ran (above). `picks.commit_tail_trim` takes an optional `guard_dt` param
(one existing call site omits it -- `seed_tail_trim`'s own commit, since a seeded trim is never
past the guard by construction; `ui.py`'s Overview drag handler is the other call site, and it
always passes `res.resampled_full.guard_dt`) and sets `state.tail_guard_override = tail_trim_dt
is not None and guard_dt is not None and tail_trim_dt > guard_dt` every time it runs, so a later
commit that no longer lands past the guard clears a stale `True` back to `False`.
`interpret.resolve_tail_cut_dt(tail_trim_dt, guard_dt, override)` is the single place both
`compute_all`'s masking and `render_overview`'s display cut resolve this: no trim set ->
`guard_dt` (or `None`); a trim at/before the guard (or no guard at all) -> the trim, unchanged; a
trim past the guard -> the trim, unchanged, only when `override` is `True`, else clamped back to
`guard_dt`. `store.LOG_COLUMNS`/`build_log_row` log the flag too, as `tail_guard_override`,
tail-appended at the very end of `LOG_COLUMNS` (not adjacent to `tail_trim_s`/`tail_trim_reason`,
which sit earlier in the column order, per the append-only convention).
`compute_all` resolves this cutoff once and reuses it for the masking above -- when the resolved
cutoff equals `guard_dt` exactly (no trim narrower than the guard, or an explicit trim that got
clamped back to it for sitting past the guard with no override), the mask is `rs_full.dt <
cutoff`, not `<=`: the bidirectional resampler can now keep a sample exactly at `guard_dt` (the
first rising sample of the guarded excursion, if it moved >= step off the last pre-guard kept
point), and that sample must stay excluded to preserve the guard's historic "nothing at or past
`guard_dt` is consumed" contract. Any other cutoff (an explicit trim, whether below the guard or,
with an override, past it) keeps `<=` as before. `compute_all` reuses the same resolved cutoff
for one more thing: the "Tail guard stopped resampling ... later data excluded" warning fires in
its original form only while the guard is still actually binding, i.e. the resolved cutoff is
`<= guard_dt` (the no-override default, and a stale trim that got clamped back). Once an override
lets the trim stick past `guard_dt`, `compute_all` checks whether that override actually admitted
new resampled points there (`np.any((rs_full.dt >= guard_dt) & (rs_full.dt <= cutoff))` --
INCLUSIVE of `guard_dt` itself, since the bidirectional resampler can keep a sample exactly there
that the default mask above always excludes) before deciding what to say instead of this warning
-- it is never just silently dropped:
  - **Data genuinely admitted**: the original warning is suppressed outright. With the
    bidirectional keep rule this is now the common case for a real tail rise -- the excursion's
    own first sample usually clears >= step off the last pre-guard kept point on its own (default
    `resample_step` equals the guard's fixed 30-psi `rise_tol`), let alone a continued climb or a
    later decline. The separate "Tail trimmed ... " warning (from the `tail_trim_dt is not None`
    block below) already reports the real, later cutoff, and showing both at once would
    contradict each other.
  - **Nothing admitted** (reachable only when NO sample in `[guard_dt, cutoff]` ever clears that
    +-step move off the last pre-guard kept value -- e.g. a rise whose height clears the fixed
    30-psi `rise_tol` enough to fire the guard but not `resample_step`, when the latter is
    configured larger than that height): the original warning is replaced with a distinct, honest
    one -- `"Tail-guard override to N min added no points past M min; results unchanged"`. The
    ordinary "Tail trimmed ... (0 samples excluded)" message is also
    suppressed in this specific case (it would otherwise sit right next to the honest warning and
    read as a confusing, near-contradictory pair claiming both "nothing changed" and "0 excluded"
    about the same cut) -- everything the plain message would have said is already covered by the
    honest one.

`plots.render_overview`'s `show_trim` kwarg is now `interactive` (default `True`, meaning "this is
the live canvas, not an export" rather than "the analyst toggled the tool on"); `render_step_figure`
passes `interactive=False` for `"overview"` (mirroring the `step_key == "gfunction"` special
cases), so an exported PNG never carries a line the analyst can't actually drag. The effective
display cut is `interpret.resolve_tail_cut_dt(state.tail_trim_dt, res.resampled_full.guard_dt,
state.tail_guard_override)` -- both `tail_trim_dt` and `guard_dt` live in shut-in-relative dt, so a
stale trim left behind by a shut-in move (before a resync runs, or for any other caller that builds
`res` without going through the Injection wiring) can sit PAST the guard; taking `tail_trim_dt`
alone (the old "a set pick is always <= guard_dt, no clamp needed" reasoning) would then render the
guard-excluded region as kept, the opposite of this feature's purpose, so the clamp-back-to-`guard_dt`
is required whenever `tail_guard_override` isn't set -- and skipped, on purpose, when it is, so a
deliberate drag past the guard actually sticks. The
Overview renderer grays out the raw trace past that cut (`gid="tail_excluded"`) whenever it's not
`None`, which is what makes a **guard-excluded** tail finally visible on Overview too --
previously only the G-function plot's own `guard_excluded` preview showed it there; the
G-function plot itself
still carries no trim artifacts at all. The draggable vline itself (`gid="tail_trim"`, drawn only
when `interactive`) sits at that same cut when set, else the last raw sample time. Releasing a
drag at/past the last point clears the trim, and the commit clamps to >=3 kept points -- a record
whose full resample already yields <=3 points can therefore only ever clear the trim, never set
one (a pathological saved trim hits a warning directing the analyst back to the Overview tab's
trim line, instead of a dead plot). The trim is never touched by a scenario change or by either
min-dP/dG correction (the triangle drag, the Shift+drag window) -- this replaces the old
"manual-only (no seeder) ... never auto-set" rule, which no longer holds.

`render_overview`'s `ViewDefaults.ylim` is pinned to `(0.0, p_hi + pad)` unconditionally --
including a converted-BHP record, where it squashes the trace into the top of the axes; the
y-slider and Reset view are the escape. Because that can reach below the Axes' own autoscaled
extent, both `ui.refresh` and `plots.render_step_figure` union a renderer's `ViewDefaults.ylim`
into `full_y` (the slider's outer range) for every step except gfunction, which still *replaces*
`full_y` (it must keep shielding the y-slider from the effective-ISIP tangent's dashed extension,
which can swing the Axes' own autoscale to extreme psi) -- otherwise `_make_range_slider`'s
`valinit` clamping would silently pull the view back up into the autoscaled extent on the first
slider touch, losing the 0 baseline.

Warnings: WHP below 100 psi (`interpret.MIN_SURFACE_PRESSURE_PSI`) anywhere the resampler
actually consumed post-shut-in data -- up to the rise guard's own cutoff
(`resample.Resampled.guard_dt`), unless an override is actively extending past it (see
"Overriding the guard" below, which this scan's window also respects), further narrowed by a
trim if one is set, deliberately *not*
bounded by the last resampled point kept (which can sit up to one `resample_step` above the true
minimum and so miss a crash just past it) -- flags BHP as unreliable there (only when the mapped
channel is surface pressure). The `guard_dt` bound is itself skipped when an active
`tail_guard_override` has actually extended the effective cutoff past it (reusing the same
`cutoff > guard_dt` check `compute_all` already made for the guard-warning logic above, not
re-derived) -- otherwise a sub-100-psi crash the override genuinely admits into the diagnostics
would never get its own warning just because it sits past `guard_dt`; the `tail_trim_dt` bound
still narrows the scan by the explicit trim either way. A stale-pick warning (gated on a trim actually being set) covers two
distinct failure modes: the G-function picks (contact, min-dP/dG, closure) left beyond the trim
would otherwise silently interp-clamp to the trimmed edge, while a pore-pressure window affected
by the trim gets the same warning by outcome -- a finite upper bound beyond the trimmed edge just
shrinks its fit (still returns a value), and a window with fewer than 2 surviving samples empties
it and blanks `pore_pressure` outright (an open-ended upper bound shrinks benignly with the trim
and is exempt from the shrunk check) -- so neither failure mode goes silent.

**Pressure dropouts.** A brief gauge dropout -- pressure reads near zero for a few seconds, then
returns -- otherwise poisons everything downstream of it: it pins `resample_pressure_increment`'s
`running_min` at the dip (nothing later is ever `RISE_GUARD_PSI` below that, so resampling stops
dead there) and the recovery itself fires the rise guard. `resample.detect_dropouts(dt, p)` finds
and masks these automatically, no per-test toggle, on the raw mapped pressure channel (before any
hydrostatic offset, so "near zero" means near zero on the gauge, not on a converted BHP) --
`model.compute_all` runs it once `res.t_shutin_s` is known, before the resample block, and stores
the result on `DerivedResults.dropout_mask` (bool, full length over `td` samples, `False` before
shut-in) and `DerivedResults.dropouts` (a `resample.Dropout(dt_start, dt_end, n_samples, p_min)`
per event, describing the dip run itself, not any lead-in extension). Every consumer of raw
post-shut-in pressure applies the mask by treating a masked sample as missing (NaN) --
`res.bhp_all` itself is never mutated. Onset is a sample falling to <=
`resample.DROPOUT_FLOOR_FRAC` (10%) of the most recent finite, unmasked level, gated on that level
being above `resample.DROPOUT_MIN_REF_PSI` (100 psi) -- without that gate, noise on an
already-dead channel re-triggers onset endlessly. The dip run is masked only if it recovers
(returns to within `RISE_GUARD_PSI` of the pre-dip level) before the record ends and the recovery
is "momentary": not sustained by the rise guard's own definition (`>= RISE_GUARD_SUSTAIN_S` AND
`>= RISE_GUARD_SUSTAIN_SAMPLES`) and under `resample.DROPOUT_MAX_S` (10 min) total duration -- a
crash that never recovers is the low-pressure tail trim's job, not this detector's, and a
genuinely sustained dip (a dead channel, a real quiet period) is left alone. A lead-in extension
(a slide toward zero just before onset, seen on the motivating Encore record) is walked backward
from onset and masked too, but only when the walk stops on its own -- within `RISE_GUARD_SUSTAIN_S`
of onset, at a level within `resample.DROPOUT_LEADIN_MATCH_PSI` (150 psi) of the recovery level --
so a real step change that happens to straddle a one-sample glitch (a long plateau at a level
nowhere near the recovery) masks only the glitch, not the plateau. `detect_dropouts` is inherently
sequential state (not vectorizable outright), but it spends almost all of its length outside any
dip even on a record that DOES have one, so `resample._dropout_candidates(p, floor_frac, min_ref)`
computes every possible onset position -- and the `ref` each one is checked against -- in one
vectorized pass over consecutive pairs in the finite-only subsequence of `p`, and
`resample._dropout_scan_candidates` jumps directly between them instead of stepping sample by
sample there. This is exact, not approximate: outside an active dip (the only place onset is ever
checked -- a dip's own interior samples are compared only against the frozen pre-dip `ref`, never
re-checked for a fresh onset), `ref` is always exactly the immediately preceding finite sample --
including right after a dip resolves, where it resets to the recovered sample, which is already
that relationship in the same global computation -- so every position the scan would ever need to
examine is one of these candidates, with the correct `ref` already attached. A candidate that
happens to sit inside a dip a still-earlier candidate already resolved is simply never reached: the
scan skips straight past the whole resolved range (`np.searchsorted` against the sorted candidate
list). Only the in-dip recovery scan and the lead-in walk still step sample by sample -- both are
bounded by the event (a handful to a few hundred samples), not by the whole record. On Encore's
1.2M post-shut-in samples this drops `detect_dropouts` from ~0.7 s (the original per-sample loop,
which used to run in full even once a dropout was found) to roughly the vectorized pre-check's own
cost. The original per-sample loop (`resample._dropout_scan_loop`) is kept as a reference
implementation, called only by a test that fuzzes it against the candidate-jump path over
thousands of randomized series. Consumers: the resample block
masks `res.bhp_all` to NaN before calling `resample_pressure_increment` (already handles NaN --
skipped, resets any rise run -- no resampler change needed) and reuses that same masked array for
the guard-excluded preview; the low-surface-pressure scan excludes masked samples from its window
(`m &= ~res.dropout_mask`) so a masked glitch never fires "Surface pressure fell below 100 psi";
and `picks.seed_tail_trim` masks `p_surface_post` to NaN before calling
`interpret.suggest_tail_trim_dt` (which already ignores non-finite samples) so a dropout never
triggers the `"low_pressure"` auto-trim. A firing detector is never silent: `compute_all`
`insert(0)`s one warning line -- on the motivating Encore record, `"Pressure dropout masked at 11
min (23 samples, 22 s, to 12 psi)"` -- or, for several events, `"3 pressure dropouts masked,
first at 11 min"` -- silent when there are none. Both the minute and (single-event) duration figures are
gated on their FORMATTED (rounded) value, not the raw number, so what's checked always matches
what's printed: `"at 0 min"` reads as "no time elapsed at all" and is misleading for anything in
the first ~30 s, so it becomes `"at <1 min"` instead when `f"{dt_start/60:.0f}"` rounds to `"0"`;
likewise the single-event duration clause is dropped entirely (not printed as the useless `"0
s"`) when `f"{duration:.0f}"` rounds to `"0"` -- true for an exact-zero single-sample dip
(`dt_end == dt_start`) and for a sub-0.5 s multi-sample one on a sub-1-Hz channel alike. The
single-event line also grammar-checks the common case: `"1 sample"` not `"1 samples"` for a
single-sample glitch -- e.g. `"Pressure dropout masked at <1 min (1 sample, to 15 psi)"`.
`plots._split_dropouts(p, mask)` splits a trace into `(p_clean, p_masked)` (NaN in the
other array); `render_overview`/`render_isip` plot `p_clean` as the main trace (so the line breaks
across a masked gap instead of spiking to it) and `p_masked`'s finite samples as small magenta
markers (`gid="dropout_masked"`, label "masked dropout") -- undecimated, since the handful of
masked samples in a 10^5+-point record would almost certainly fall between the main trace's
decimation stride and never get drawn. The markers are plotted with `scalex=False, scaley=False`
and the Axes' `dataLim` is snapshotted before and restored after, so a masked sample's real value
(a -9999 psi sentinel, or Encore's ~12 psi) never pulls the autoscaled view -- or anything else
that reads `ax.get_ylim()`/`dataLim` afterward, e.g. the ISIP step's default view or the Overview
y-slider's full range -- down to it; the marker is still a real plotted artist at its true (and
possibly off-screen) position, just excluded from the bounding box. Exported PNGs
(`render_step_figure`) get this automatically, since it isn't gated on `interactive`. Above
20,000 masked samples (one rise excursion can mask 38k) the markers are thinned to an even stride.

**Rise excursions.** An upward excursion that comes back down otherwise trips the tail guard and
ends resampling there. Arkansas 1BH `1BH MERGED.DBS` has a +50 psi bump at 975-1075 min after
shut-in (guard at 985 min, ~200 h lost). Akbary `Edge - Akbary DFIT Injection All Data
Combined.DBS` has a +2277 psi excursion at 954-1046 min (guard at 955 min, ~35 h lost).
`resample.detect_rise_excursions(dt, p)` uses the guard's own run definition (continuously above
`running_min + RISE_GUARD_PSI` for >= 60 s and >= 5 samples), so water hammer is never a
candidate. The return is the first later finite sample <= `base + 30`. The mask runs from one
sample after the last new running-min sample before the run (so the slow start of a ramp is
included), walked forward so it is never longer before the run than the event's own duration,
through the sample before the return. An event is masked only when all three hold:
- it starts >= `RISE_EXCURSION_MIN_START_S` (180 s);
- its duration is <= `RISE_EXCURSION_MAX_FRAC` (0.25) of the elapsed time at its start;
- the median of the first 5 finite samples from the return is >= `RISE_EXCURSION_CRASH_FRAC`
  (0.5) of the base.

Any failure stops the scan with nothing masked for it or after it, so the guard fires there
exactly as it would without the detector. Masked samples never lower the running min, so the fast
path computes the running min, runs, fire points and new-min positions once over the whole
record. Only the return search runs per event, with a doubling window. `_rise_excursion_loop` is
the reference, fuzzed against the fast path.

Corpus evidence (2026-10-02, all 3550 `C:\DFIT Data` tests, first source, auto-seeded injection):
- 2712 loaded and resampled, and the guard fired on 1116 of them.
- 956 of those had a first excursion that returns. Their duration/elapsed spreads continuously
  from 1e-4 to >500, with no clean gap.
- Contact sheets (late starts) at <= 0.25 were mostly real glitches: Arkansas, Latham, Alma West
  bumps and plateaus.
- 0.25-0.5 added mostly junk: a second pump pulse (Crestone Reserve), a dead gauge (Montney), and
  false-base events.
- Several "returns" were the gauge crashing to ~0 (Crestone Rush 4CH, Raindance, Kiwetinohk),
  hence the crash check.
- Early excursions (first ~15 min) were mostly pressure steps from a mis-picked shut-in or
  staged-down pumps, hence the start floor. With the 0.25 cap a >= 60-s run can't be masked
  before 240 s anyway, so the 180-s floor only acts if the cap is raised.
- A cluster at ~830-1030 min after shut-in recurs across operators (Akbary, Arkansas, Latham x2,
  John Peanut, Deporter, Khaki Campbell, Liberty 10TFH, Berry, SM B&D). It looks like a merge
  seam or surface operation. These spread 0.16-0.9, and the cap catches about half.

The detector inherits the guard's weakness: a single-sample downward blip pins a false running
min, after which real decline sits "above" it (Akbary ~1350 min, Black Hills Ponderosa). A manual
mask over the blip fixes it, because manual masks are NaN'd before detection.

**Manual masks.** `PickState.mask_intervals` / `keep_intervals` are absolute `td.t_s` seconds
(stable across shut-in moves), sanitized by `_decode` (non-finite, inverted or malformed entries
are dropped). `compute_all` builds `res.dropout_mask = (dropouts | rises | manual) & ~keep`, post
shut-in only.
- A keep is a true override: a kept rise is resampled, and the guard fires on it again.
- An event whose own contiguous masked run (including a dropout's lead-in) is entirely kept is
  removed from `res.dropouts` / `res.rise_excursions`, so it is not reported as masked.
- Warnings: "Manual mask: N samples in K intervals" and "Manual keep: N auto-masked samples
  restored".
- Overview gestures share one `_CaptureGate` with the tail-trim line. Shift+drag adds a mask,
  Ctrl+drag a keep (`ModifierSpanController(exclude=...)` keeps them exclusive), right-click a
  band removes it (`IntervalRemoveController`), and a Clear button empties both lists.
- `picks.commit_mask_interval` merges into one kind and subtracts the range from the other, so
  the lists never overlap.
- Log columns (appended): `dropouts_masked`, `rises_masked`, `manual_masks`, `manual_keeps`.

**G-function.** α = 1 (low-leakoff) is the default; α = 0.5 only if a test exceeds ~1 md.

The dP/dG (twin) axis has two independent limits. The **default view** autoscales to the data:
the max of dP/dG over `G >= plots.Y2_SCALE_G_MIN` (1.0, the same g_min convention
`interpret.suggest_min_dpdg_index` and the d2P/dG2 block use), +10%, floored at 1.0, capped at
`plots.DPDG_VIEW_MAX`. Masking by G is what keeps the early water-hammer spike out of it -- a
percentile over all samples was dominated by that spike, since the resampled grid is densest
across it, and the old hard 50-psi/G cap squashed any record whose real derivative ran higher.
A record entirely below G=1 falls back to all finite samples. The **slider's full range** is the
twin Axes' own autoscale unioned with that default, then hard-clamped into
`(0, plots.DPDG_VIEW_MAX)` -- both steps in `plots.apply_step_view`, which `ui.refresh` and
`plots.render_step_figure` share. The clamp is what stops the slider
traveling past 2000 no matter how extreme the raw spike is. The union is what keeps the default
view inside the travel: the raw autoscale is inflated at the top by the near-G=0 spike and
lifted off zero at the bottom by a nonzero data minimum, so the default can fall outside it in
either direction, and `_make_range_slider`'s valinit pinning would then snap the view off the
default on the first slider touch -- the same failure the `full_y` union prevents on every other
step. Note the full range is an *intersection* with `(0, DPDG_VIEW_MAX)`, not that interval itself.
`render_tangent`'s own G·dP/dG default is still the 95th-percentile rule and has no cap.

Closure scenarios drive the contact pick and effective ISIP:

| Scenario | dP/dG shape | Stress pick |
|---|---|---|
| C-A clear | clear "S" (min then rise) | contact at min + 10%, − 75 psi |
| C-B adequate | monotonic with inflection | contact at the inflection, − 75 psi |
| C-C no-contact | monotonic, no inflection | none (no Shmin) |
| C-D rapid | monotonic, concave-up, no tortuosity | apparent ISIP − 100–250 psi (methodology; see note) |

Note: in the current code (`picks.apply_closure_scenario`) C-C and C-D both clear the contact
pick, so `compute_all` produces no compliance Shmin and no effective ISIP for either. C-D
additionally reports `shmin_rapid` = apparent ISIP − 175 psi (the midpoint of the 100–250 psi
range; `interpret.RAPID_CLOSURE_OFFSET_PSI`), shown in the G-function title
(`interpret.format_shmin_rapid`, verbose form). Net pressure is deliberately not
derived from it -- there is no `net_pressure_rapid`.

**Panel asterisks.** Three result-panel rows carry a trailing `*` on the label when the number in
them came from a fallback rather than the primary construction. All three are display-only, set in
`ui._update_panel` by mutating `self.name_lbls[...]`. The actual rule is that the gate matches
whatever makes the value column render a real number instead of `"-"`, so an asterisk can never
sit next to a `"-"`: two of the three rows blank only on the same `not_visited` check the value
column uses, but `"Shmin compliance grad* (psi/ft)"` has a second blanking path the value column also
respects (see below) and so needs a second clause in its gate. All reset to plain text on the
else branch, since the label widgets persist across refreshes. None is explained in the panel
itself.

- `"Shmin compliance*"` -- `shmin_rapid` has no panel row of its own; it stands in for the
  compliance Shmin in that row when `use_rapid = shmin_compliance is None and shmin_rapid is not
  None`. The value is the short `format_shmin_rapid` form, `"9325 ±75"` -- no tilde, the ±
  half-range is what signals the approximation in the value column. The two are never both set
  (C-D clears the contact, and `compute_all` only sets `shmin_rapid` for C-D), so the fallback is
  unambiguous. The G-function title separately carries the verbose
  `Shmin(rapid)=9325 ±75 (ISIP − 100–250)` whenever this is showing.
- `"Shmin compliance grad* (psi/ft)"` -- shadows the `"Shmin compliance*"` row directly above it, same
  `use_rapid` gate, selecting `shmin_rapid_gradient` instead of `shmin_compliance_gradient` for
  the value. Unlike the parent row there is no `±75` half-range in a gradient value (not worth
  rendering), so the label asterisk is the only marker here. `shmin_rapid_gradient` itself gets
  no panel row of its own, mirroring `shmin_rapid`. This row has a second blanking path the
  parent row does not: `model._resolve_gradients`'s `tvd_ft > 0` guard, which blanks every
  gradient (including `shmin_rapid_gradient`) regardless of `use_rapid` -- reachable on a
  C-D-rapid test with no TVD entered (e.g. a `pressure_is_bhp` downhole-gauge record, where TVD
  is never needed). So this row's gate adds a third clause, `r.shmin_rapid_gradient is not
  None`, on top of `use_rapid and gf_visited` -- without it the label would show the asterisk
  over a `"-"` value in that state.
- `"NWB complexity*"` -- complexity is apparent ISIP minus the *shared reference* effective ISIP,
  and this marks the case where that reference fell back to the tangent effective ISIP:
  `use_tangent_ref = near_wellbore_complexity is not None and net_pressure_isip_source ==
  "tangent"`. The reachable condition is "no compliance effective ISIP but a tangent one exists",
  which C-C and C-D both produce (they clear the contact, so there is nothing to reference), but
  they are not the only way there -- e.g. selecting C-C and then C-A, where
  `re_derive_contact_from_min` finds nothing at >=110% of the min and leaves `contact_G` `None`,
  shows the mark under a C-A scenario. It is truthful in every such state; the gate is the
  reference, not the scenario. Note the two asterisks do not mean the same thing -- this one says
  "referenced to the fallback", not "approximate". Under C-D both show at once **once the closure
  pick exists**: without it there is no tangent effective ISIP either, so complexity is `None` and
  only the Shmin asterisk shows.

The CSV log is unaffected by either: `store.LOG_COLUMNS` keeps its own `Shmin_rapid` column, and
`net_pressure_isip_source` is already logged.

**Min-dP/dG pick.** The red triangle seeds from `interpret.suggest_min_dpdg_index`: interior
local minima of dP/dG over the whole record (no fixed G threshold), preferring candidates
before the fracture-contact hump. `interpret.suggest_hump_index` locates that hump as the
interior local max of dP/dG with the largest G·dP/dG -- the water-hammer spike near G=0 loses
on the G factor, the decaying tail has no local max -- falling back to the raw argmax on a
monotone record; the same helper seeds the blank-scenario contact placeholder. The seeder
(`picks.seed_gfunction`) hides two regions (NaN) from `suggest_min_dpdg_index`,
`suggest_hump_index` and `is_clear_closure`; the interpret suggesters themselves still search the
whole curve, and manual corrections are untouched:
- **G < `interpret.SEED_MIN_G` (1.0).** On 1-s data with a fast early decline the 30-psi
  resampler keeps nearly every raw sample, and the pressure *rate* wobbles second to second.
  1BH MERGED.DBS: BHP 10,690 -> 6,950 psi in the first 209 s, monotone (not water hammer), dP/dG
  swinging +-10x below G ~ 0.5 and +-20% near G = 1. Its envelope looked like a hump at G ~ 0.3
  (G*dP/dG 3,400 vs the real bump's 450), so the min seeded at G = 0.074 and passed the C-A gate.
  An auto-seed below G = 1 is never right in practice, so the floor is a flat rule. It is skipped
  when fewer than 6 samples sit at G >= 1 (a record that never reaches G ~ 1 is a te/window
  problem; those keep the old whole-curve seed). Rejected alternatives: masking points whose
  central difference spans < 20 raw samples (it also masked clean 5-10 s data and lost real
  elbows; review finding), a zigzag-roughness prefix, and log-G median smoothing.
- **A trailing crash spike (`interpret.terminal_spike_start`).** Walking back from the last finite
  sample while |dP/dG| > 10x (`TERMINAL_SPIKE_FACTOR`) the median |dP/dG| over G <= 95% of the last
  G (`TERMINAL_SPIKE_REF_G_FRAC`), stepping over up to 2 small or NaN samples inside the spike
  (`TERMINAL_SPIKE_MAX_GAP`; a down-then-up reversal puts a ~0 central difference mid-spike). This catches bleed-offs the tail trim cannot: a BHP channel
  (Delphi 46702 100-09-21-059-22W5: BHP to 31 psi, ~60 crash points in the last 0.03% of G, more
  than the falloff itself, hence the G-based reference rather than a sample median) or a record
  ending mid-bleed above 100 psi (Flaherty 18-7-10NBH, Stenehjem 10H, Harlequin, Khaki Campbell
  Trap1, all previously auto C-A on the spike). A real C-A hump is a 10-100% rise, never 10x.

1BH after both: min G = 25.14, contact placeholder G = 38.0, no auto C-A (the rise is 1.8%). On
the 88 usable corpus tests above: seeds at G < 0.1 went 38 -> 10 and below G = 1 57 -> 15 (all 15
on records whose curve ends at G <= 1.16), auto C-A 43 -> 30, no contact on a terminal spike.
`seed_tangent` is not masked: on 1BH its seed lands at G = 1.90, probably the same noise. A record with
no interior local min at all (C-C shape) falls back to the G>=1-masked global min. Two
corrections exist: dragging the triangle commits on release and re-derives the contact via
`re_derive_contact_from_min` (C-A: +10% rule; C-B: g_min-masked inflection nearest the drag);
Shift+drag selects a window instead (`picks.ModifierSpanController`, sharing the step's
capture gate and registered ahead of the point controllers so a Shift-press wins the gesture).
The window commit, `picks.handle_min_dpdg_window`, sets both picks itself: C-A puts the
triangle at the window's dP/dG min and the contact by the +10% rule from there; C-B puts the
triangle *and* the contact at the inflection found inside the window
(`suggest_contact_inflection_index(g_range=...)`, no g_min mask, so a below-G=1 inflection
stays where the analyst put the window). The span is a live gesture only -- never persisted,
never rendered in exports. When C-A is active and dP/dG never reaches 110% of the picked min
(no clear contact), `render_gfunction` draws a dashed hline at that threshold on the dP/dG
axis (`gid="clear_threshold_line"`, skipped when the threshold is non-finite) with a
"consider C-B" label; it appears in exported PNGs whenever the condition holds.

Postclosure scenarios drive the pore-pressure axis:

| Scenario | log-log signature | Pore-pressure axis |
|---|---|---|
| PC-A linear | bends to −1/2 | t^(−1/2) |
| PC-B false-radial | −1 after peak | t^(−1) |
| PC-C false radial to genuine linear | −1 then −1/2 | t^(−1/2) |
| PC-D genuine linear to genuine radial | −1/2 then −1 | either |
| PC-E no trend | peak, no clear slope | t^(−1/2), low confidence |
| PC-F no peak | derivative still rising | pore-pressure step skipped |

This linkage is implemented: `picks.suggest_pp_axis` maps a postclosure scenario string to
`pp_axis` ("tm12"/"tm1") by its `scenario[:4]` prefix, so the mapping is label-independent, and
`ui._on_scenario` applies it whenever the postclosure-scenario combobox changes, also
re-syncing on step entry/load via `_update_panel_visibility`. The side panel's pore-pressure-
axis radios lock (disabled) whenever a scenario dictates the axis. PC-D ("either") and PC-F
("no peak") are intentionally absent from the mapping, so the radios stay enabled and the axis
is left to the analyst -- moot for PC-F, since that scenario skips the axis-picking step
entirely (below). `model._decode` normalizes the four old pre-rename labels (`"PC-C mixed"`,
`"PC-D mixed"`, `"PC-E none"`, `"PC-F none"`) found in older saved picks JSON to their current
form; unrecognized strings pass through untouched.

**Log-log window seed and PC-A auto-assign.** `picks.seed_loglog` takes its window from
`interpret.suggest_loglog_window` on t·dP/dt. The window always starts after the postclosure
peak, found by `interpret._loglog_peak`: the latest local maximum (max over ±3 samples) whose
topographic prominence is >= 0.15 decades and that has >= 5 samples after it. "Latest", not
"tallest": Argentine 7170 has an early spike at ~6 s, and Caprito 99-202H has a pre-closure
hump at ~280 s about 6x taller than its postclosure peak at ~4.8e4 s. An earlier
trough-then-argmax rule failed on Caprito because its first sample (t = 3 s) was the global
low. After the peak, every window of >= 5 samples is fit in closed form (prefix sums, O(m²)).
The widest window (in decades) with |slope + 0.5| <= 0.10, residual RMS <= 0.05 decades, and
span >= 0.3 decades wins. Otherwise the straightest section at any slope: the widest window
with RMS <= 0.05 and span >= 0.3 decades, else the lowest-RMS window. The −1/2 preference is a
first choice only; an earlier "closest to −1/2" fallback landed in the curved rollover just
after the peak whenever the real trend was some other slope. No qualifying
peak (derivative still rising) falls back to the old last-40% window. On 7170 the seed is
1.56e5–1.57e6 s, slope −0.58, which matches the analyst's saved window (1.55e5–1.67e6 s, PC-A).

The window slope is `DerivedResults.loglog_slope` (`interpret.loglog_window_slope`, both edges
inclusive). The renderer used to compute it over `[i0, i1)` and drop the right edge.
`picks.auto_assign_postclosure(state, slope)` sets `"PC-A linear"` with
`PickState.postclosure_auto = True` when the slope is within 0.10 of −1/2. It runs after every
window drag (`ui` loglog `on_span`), and after the seed only when `suggest_loglog_window`
reports a qualifying −1/2 window (its third return value); a fallback window is not
evidence of PC-A. It never touches a scenario the analyst chose; it re-sets an auto PC-A on a hit and
clears it to blank on a miss. Any postclosure-combobox selection (`ui._on_pcscen_selected`)
clears `postclosure_auto`, including re-picking the auto PC-A, which confirms it. The closure
combobox and the axis radios share `_on_scenario`, so the clear lives in the combobox handler
only. While the flag holds, the
log-log hint says the scenario was auto-set (the plot title does not); the flag is logged in the
`postclosure_auto` tail column.

**PC-F skip.** `model.skipped_steps(state)` returns `{"porepressure", "stiffness"}` whenever
`postclosure_scenario` starts with `"PC-F"`, else an empty set. It is the only place the rule
lives. Porepressure is skipped because the derivative never peaks, so no postclosure line
exists; `compute_all` leaves `pore_pressure` `None` even if a stale `pp_window` pick exists.
Stiffness is skipped because its h-function needs a pore-pressure estimate as its `Pres` term.
Consumers ask `skipped_steps` (or the helpers built on it) rather than checking the scenario:
`model.last_step` reports `"loglog"`, so `_advance`'s Next button becomes "Finish" there;
`model.resolve_step` (called by `ui._goto`) sends a skipped destination (the log-log Skip
button, resume-on-load, a breadcrumb click) to the nearest earlier active step, `"loglog"`;
`_update_stepbar` force-disables skipped breadcrumbs even if visited earlier in the session;
`plots.save_all_step_pngs` omits their PNGs (the others keep their `STEP_KEYS`-order
numbering); and `store.status_for` drops them from the steps that must be accounted for, so a
PC-F test with neither step's `step_status` entry still reports `"done"`.

PC-F is not the only way to land on an empty stiffness plot -- `plots.render_stiffness` shows
an instructional title in place of the plot (no exception, no blank axes) whenever
`res.stiffness_S` is `None`, which `model.compute_all`'s stiffness block leaves unset in any of
four gate failures: `state.min_dpdg_G` not yet picked, no pore-pressure estimate (this is where
the PC-F case actually surfaces, transitively, per `skipped_steps` above), `res.te_s`
unavailable, or fewer than 4 resampled points surviving (the tail trim, an aggressive resample
step, or a short record can all produce this). A separate guard inside the same renderer handles
a fifth, arrays-present-but-degenerate case: `stiffness_S` computed but entirely
non-positive/non-finite (no positive sample to log-scale against) also renders an instructional
title rather than an axis-only log plot.

An in-app "Interpretation guide" window (opened from the "Interpretation guide..." buttons on
the G-function step's closure-scenario panel and the log-log/pore-pressure steps'
postclosure-scenario panel) shows the ResFrac closure (C-A...C-D) and postclosure (PC-A...PC-F)
figures and explanatory text side by side with the pickers. Content lives in
`dfit_tool/guide_content.py` (headless, no Tkinter); figures are vendored PNGs under
`dfit_tool/assets/guide/`. Both buttons open the same reused `ttk.Notebook` window and just
select their tab (`ui.py:_open_guide`).

**Unit detection and conversion.** Everything downstream of the IO boundary (fixed psi
offsets, the hydrostatic constant, plot labels, panel rows, `dfit_log.csv` columns) stays
field-unit (psi/bpm/bbl); metric input is normalized at the boundary instead. Detection order
per channel (`io_load.detect_channel_unit`): a per-channel UI override (anything but `"auto"`)
→ the column header's parenthesized suffix (`_unit_of`, e.g. `"CASING Pressure (KPAg)"`) →
a kind-specific fallback. Pressure's fallback is the magnitude heuristic
(`classify_pressure_magnitude`): 99th percentile of finite samples, `< 15,000` → psi/high,
`>= 20,000` → kpa/high, the 15-20k overlap band → psi/low (no clean single-unit reading, so it
defaults to the historically-assumed unit). Rate and volume get no standalone heuristic (the
bpm/m3-min ranges overlap with no clean threshold) and instead inherit the file-level pressure
verdict -- any non-psi pressure unit means metric rate/volume too, always at low confidence
(an inference from another channel, not direct evidence) -- but a channel's own header suffix
always wins over that inheritance. Conversion is lazy: `TestData.column()` multiplies the raw,
never-mutated df column by a cached factor in `td.unit_factors` (default 1.0 = no-op for any
caller that never invokes detection); `io_load.refresh_unit_detection` rebuilds
`unit_factors`/`unit_detections`/`unit_warnings` from the raw columns and is idempotent, so
`model.compute_all` calls it at the top of every recompute (before any channel read, including
picks seeders) rather than caching a detection result across state changes. Warning policy
(`_maybe_warn`) is never-silent-on-a-real-conversion: always warn when a resolved factor isn't
1.0, also warn on a low-confidence pressure heuristic call even at factor 1.0 (the classification
itself is uncertain even though it landed on the default unit) -- silent only for the ordinary
factor-1.0 field-units load. `PickState` carries `pressure_unit`/`rate_unit`/`volume_unit`
(each `"auto"` by default; old saves lacking the keys take the dataclass default, no migration
needed). `ui._on_unit_change` mirrors `_on_source_change`: changing a channel's unit away from
its current value pops a confirm dialog (an override rescales the same column under existing
picks -- `isip_tangent` stores an absolute psi anchor, `te_s` feeds `g_time`), reverting the
combobox on decline; accept resets picks via `_reset_picks_keep_mapping` (channel mapping, unit
overrides, density/TVD, well/formation, alpha, resample step, notes, and active source carry
forward; numeric picks/step_status/tail_trim reset) and returns to Overview. With no file
loaded (`self.td is None`) it just reverts the combobox before ever reaching the dialog.
`store.LOG_COLUMNS` carries one `units_note` column (tail-appended, per the append-only
convention), mapped from `DerivedResults.unit_conversion_note` -- a compact summary of every
non-1.0 factor applied (e.g. `"pressure: kpa×0.145038 (header)"`).

`questionnaire.py` gained matching metric handling: a TVD cell is read as meters, detected any of
three ways -- an explicit meter suffix (`mTVD`/`mKB`/`mMD`/bare `m`/`metre(s)`/`meter(s)`, checked
immediately after the matched number so a bare number stays feet); the label itself spelled
`mTVD` (e.g. `"mTVD: 3368"`); or a delimiter-bounded meter token in the gap between the label and
the number (e.g. `"TVD (mKB): 3368"`, `"TVD in mKB = 3368"`) -- and is converted via
`units.M_TO_FT` *before* the `_TVD_MIN`/`_TVD_MAX` range filter -- ordering matters, since a
meters value like 3368 sits inside the feet-assumed 1000-25000 window and would otherwise be
silently accepted as feet; density gained a kg/m3 unit (`units.KGM3_TO_PPG`) alongside the
existing ppg/SG/psi-per-ft forms, and a fluid-name fallback (`"fresh water"`/`"freshwater"`/
`"water"`, exact trimmed whole-cell match) that assumes 8.34 ppg with a warning when a cell names
the fluid but gives no number at all. Multi-well workbooks
(`parse_questionnaire(path, well_hint=...)`) disambiguate a
multi-sheet workbook by matching `well_hint` against each sheet's title, then its "Well Name"
answer cell; a positive match narrows to that one sheet, but a missing hint or no match reads
the *whole* workbook (concatenating every sheet's cells) rather than guessing `sheets[0]`
alone -- narrowing only ever happens on a positive match, so a `scripts/` caller that passes no
hint at all still finds data that happens to live on a later sheet (e.g. behind a cover/
instructions sheet). `well_hint` itself comes from `ui.py`: folder mode passes the queue
entry's `test_id`, single-file mode passes the data-file stem (`ui._load_questionnaire`). The
match (`questionnaire._sheet_match_length`) is delimiter-bounded containment in both
directions -- the contained string must sit between non-alphanumeric characters or a string edge
in the container, so a short hint like "1H" can't false-match inside "21H" -- and when more than
one sheet matches, the one with the longest matched text wins, with a warning naming the chosen
sheet.

Accepted gaps, deliberately not solved: a psi-pressure file with an unlabeled metric rate/volume
is undetectable in-band (the per-channel override dropdown is the escape hatch); and there is no
filename-based unit sniffing, ever -- the corpus has a "Metric"-named DBS file that holds psi
data byte-identical to its non-metric-named sibling, so a filename heuristic would misclassify it.
In `questionnaire.py`'s TVD label matching, a digit-ending token right before an `mTVD` label
(e.g. `"Zone 2 mTVD: 3368"`) and a suffix-only TVD-first cell with no colon label (e.g.
`"3368 mTVD / mMD 6656"`) are still read as feet/MD respectively -- inherent label-vs-suffix
ambiguity, not fixed.

## Not built / notes

- Unsupported XLSX/CSV layouts, which the extrapolation gate above refuses to guess at (the
  affected files load with their timestamped rows only, plus a warning):
  - Side-by-side datasets: pump and gauge data in one sheet, each with its own timestamp columns
    (ND State 10TFH/1TFH, SM Cactus). Only the first timestamp column is used; the other
    dataset's rows are not given a time base.
  - Multi-rate TDMS exports: a "Formula Server" sheet whose pressure column has several times
    as many rows as its timestamp column (Emerald Excalibur 6-25-36H and 7-25-36H, WPX Edward Flies Away and Helena Ruth Grant; about 5x). The
    rows cannot be aligned to the timestamps without the per-channel rate, so the block is left
    NaT.
  - A channel zero-padded past the end of the timestamps (Norfolk, Bridger, Reno 11-10PH) is
    treated as padding, not data; the reported span is the timestamped record only.
- `scipy` is pinned in `requirements.txt` and probed by `start-app.ps1` but is not currently
  imported anywhere in the package.
- Sample data folders, `Refs/`, and `.superpowers/` are gitignored. Design specs and plans
  from the build are under `docs/superpowers/`.
- The master log is CSV only; a parquet mirror alongside `dfit_log.csv` is a deferred
  extension point (`store.py`'s module docstring), not yet implemented.
- No concurrency control: folder mode assumes a single interpreter working a root at a time.
  Two people (or two windows) open on the same root last-write-wins on both the picks JSON and
  `dfit_log.csv` -- there is no lock file or merge.
- Accepted known limitations in the folder-mode Source switch/resume logic: if a test's saved
  picks name a source the resume logic silently falls back to applying those picks to whatever
  source actually loads, with no warning; a source switch (`ui._on_source_change`) only
  persists to disk on the next save (queue navigation, Finish, or Skip test), not
  immediately; and using the manual "Load picks…" file-dialog button while in folder mode
  loads that JSON into the workspace but does not re-sync the Source combobox to it (the
  Skip-test button does re-sync, via `_apply_loaded_state`'s `_goto` -> `refresh` ->
  `_update_stepbar` -> `_update_skip_test_btn` chain).
- A per-channel unit override (`ui._on_unit_change`) mirrors the Source-switch limitation above:
  it only persists to disk on the next save (queue navigation, Finish, or Skip test), not
  immediately.
- A pre-`step_status` legacy picks JSON, reloaded, can silently re-run the Overview auto-trim
  seeder and change reported numbers. `infer_step_status` backfills `step_status` for saves made
  before it existed, but it omits `"overview"` from that backfill, so `"overview"` still reads
  `not_visited` on reload -- and since the tail-trim seeder is park-and-apply on that step's first
  visit (Tail trim, above), re-visiting Overview can re-seed a trim (or a different one than
  whatever was in effect when that save's `dfit_log.csv` row was written), moving Shmin/effective
  ISIP/net pressure/pore pressure with no analyst action. This is a known consequence of the
  park-and-apply design colliding with an unrelated legacy-migration gap, not something the tail
  trim feature itself can detect or guard against.
- Adding "stiffness" as an 8th workflow step is an accepted migration consequence for saves
  finished under the old 7-step workflow: those have a 7-key `step_status` with no
  `"stiffness"` entry, so `store.status_for` now reports them `"in_progress"` (not `"done"`)
  until the analyst opens the test and visits or skips the stiffness step. This is deliberate --
  the stiffness Shmin genuinely doesn't exist for those saves yet, so `"in_progress"` is the
  accurate status, not a bug -- and `infer_step_status` can't backfill it, since that function
  only runs when the loaded `step_status` is empty, not when it's merely missing one key.
- The G-function plot's `guard_excluded` gray preview (`model.compute_all`'s `res.guard_excluded_G`/
  `guard_excluded_p`) is unaware of `tail_guard_override`: it still previews the full raw tail past
  `guard_dt` even after an override has extended real, kept data into that same span. Cosmetic
  overlap only -- the preview and the real (ungrayed) resampled curve can occupy the same G-range --
  not a data-correctness issue, since `res.resampled`/`res.diagnostics` are correct either way.
- A second, unrelated excursion occurring AFTER the first guard fire (while `stop_at_guard=False`
  keeps resampling) is never separately detected or warned about -- only the first excursion gets
  `guard_dt`/`guarded_at` and a warning; `resample_pressure_increment` never re-arms the fire check
  once `guard_dt` is set. Under the bidirectional (±step) keep rule this is no longer a pure
  metadata gap: the second excursion's own samples are evaluated by the same ordinary
  `abs(pi - last_kept) >= step` rule as anything else, so a genuine second rise generally DOES get
  resampled (and can move Shmin/effective ISIP/pore pressure) with no warning pointing at it --
  only the first excursion's `guard_dt` is ever reported. Accepted as a low-severity gap, not
  fixed here.
- The apparent-ISIP tangent fit (`interpret.tangent_from_index`, ±5 samples) and the ISIP anchor
  snapping still read raw `res.bhp_all` -- neither one consults `dropout_mask`. An anchor placed
  within 5 samples of a masked dropout gets a corrupted fit.
- `resample.detect_dropouts` only masks a dip that falls all the way to `DROPOUT_FLOOR_FRAC` (10%)
  of the pre-dip level; a partial dip that doesn't reach the floor is never masked, by design (see
  "Pressure dropouts" above) -- it isn't distinguishable from a real, if abrupt, decline.
- Dropouts are only detected after shut-in (`model.compute_all` runs `detect_dropouts` on the
  post-`t_shutin_s` record only); a dropout during injection is not masked anywhere.
- A chronically flaky channel is masked one event at a time, each independently -- there is no
  single warning that says "this channel is unreliable." The event count in the "N pressure
  dropouts masked" warning is the only signal that the channel itself, not just one moment, is
  the problem.
- Detection runs on the mapped pressure channel only (`state.pressure_col`, whatever the analyst
  or the auto-suggestion chose). The auto-suggested channel can itself be wrong -- e.g. a
  secondary gauge that reads ~0 for most of the falloff -- and `detect_dropouts` has no way to
  tell that apart from a real signal; that's a separate channel-suggestion issue, not handled
  here.
- `plots._plot_dropout_markers`'s autoscale protection (see "Pressure dropouts" above) snapshots
  `ax.dataLim` before plotting the markers and restores it after -- it does not make the markers
  permanently invisible to `dataLim`, only to that one `plot()` call's own autoscale request. Any
  FUTURE call that recomputes `dataLim` from scratch (`ax.relim()`, or `ax.autoscale_view()` after
  `relim()`) would walk every artist on the Axes, including the markers, and pull them back into
  the bounding box. No current code path does this on these two renderers' Axes, but it's a latent
  trap for a future change that adds one.
- INSITE-preamble files whose real header row is misread as data (e.g. Spearhead, Powell -- see
  `_find_numeric_time_column`/`_find_clock_column`'s own "undetected preamble" cases) start their
  reported elapsed-time record a few hours early: a job-header line holding a plain date (e.g.
  `"15-Sep-2017"`, no time) sits ahead of the real first sample and, when it happens to be the
  first value `parse_datetime` accepts, becomes `t_s`'s zero point instead of the first real
  reading -- typically ~13h early on the two measured files. `_detect_header_skiprows`'s own
  preamble-skip only fires on a `ParserError`, which a uniformly-wide preamble (same field count
  as the real data) never raises, so this is a `_detect_header_skiprows` gap, not something the
  datetime parsing itself can distinguish (the date line is a perfectly genuine, correctly-parsed
  date -- just the wrong row).
- `_find_numeric_time_column`'s bare-`"Time"`-header assumed-MINUTES guess is a guess, not a read,
  and a load warning says so -- but it is still occasionally wrong on a file whose bare `"Time"`
  column is actually SECONDS (no other hint anywhere resolves the unit), reporting a duration
  60x too long. There is no way to tell the two apart from the column alone; the warning is the
  only mitigation.
- L1: `parse_datetime`'s `_FAST_PATH_SKIP_FRACTION` gate (skip the generic dateutil parse once
  the two exact fast paths already own >=90% of a column) can silently drop a genuine SECOND
  valid date format if that format's rows happen to sit in the >=90% majority resolved by the
  fast paths and the OTHER format is the minority left for Fallback 2 -- the gate has no way to
  tell "the remainder is genuinely a second format" apart from "the remainder is units/header
  lines or corrupt cells," which is exactly the ambiguity it accepts trading away for speed (see
  the Emma Owner measured case above). Synthetic-only so far, not seen forcing a wrong answer on
  any real corpus file: a file mixing two real formats where the RARER one appears first in file
  order poses no risk (still resolved, since Fallback 2 only skips based on the FRACTION, not
  position), but one where the format split happens to land close to the 90% line could go
  either way depending on which format has the numerical majority.
- Civitas Allred's `"6 - 21011210.DTF.csv"` (and its `aa_DJ Basin\Allred Fed...` copy) is the
  measured case for BOTH the fast-path plausible-year guard and the isolated-timestamp-outlier
  guard, and combining them resolves it to its real ~370.66h. A different file, `aa_DJ Basin\
  Arkansas 1BH\1BH - 21011204.DTF.csv` (same instrument export family), has the same underlying
  problem in a shape neither guard reaches: a BLOCK of ~40 consecutive samples jumps to an
  implausible-but-still-1990-2100-range date range and back (a ~3,700h round-trip), rather than
  a single isolated cell -- every sample WITHIN that block is consistent with its immediate
  neighbor (also inside the block), so `_mask_isolated_timestamp_outliers`'s "differs from BOTH
  neighbors" test never fires on any of them, only the two boundary transitions would even look
  anomalous, and neither boundary alone satisfies "differs from BOTH neighbors" (one side of
  each is consistent with its own, good-or-bad, region). The file currently reports a reasonable-
  looking but wrong ~3,850h; a genuine fix would need a block-aware version of the outlier guard,
  out of scope here.
- A headerless raw gauge workbook (its first row is already data, no header row at all) is never
  picked up by `_xlsx_find_header`, which requires a text header row before any data -- there is
  no fallback that treats row 1 itself as the data start.
- A multi-series XLSX layout with one Time column PER channel (e.g. Strathcona's `_487.xlsx`,
  where each pressure/rate/temperature series carries its own adjacent time column rather than
  sharing one) is not specifically handled: `load_xlsx` still picks exactly one datetime column
  and one optional companion for the whole sheet, the same as a CSV, so only the channel(s)
  aligned to that one chosen time base load with a meaningful time axis.
- A very large workbook (Anderson/Tank DFIT.xlsx, 85-97 MB) loads slowly, or can appear to hang,
  since `load_xlsx` still has to read every real data row of the winning sheet into memory once
  it's chosen (the zip-level row count only bounds the SHEET-CHOICE
  cost, not the eventual full read) -- and it runs on the Tkinter main thread, so the UI is
  unresponsive for the duration. Not fixed here.
- A merged header cell becomes `"Unnamed: N"` for every column except its own anchor (top-left)
  cell -- `_xlsx_trim_trailing_empty`/`_xlsx_dedupe_headers` have no merged-cell awareness, so a
  header merged across several columns effectively names only the first of them.
- A header row that is missing a label for one of its own data sub-columns can misalign every
  later column by one -- measured on Tamboran's "SS-1H DFIT Datacan ... Download.xlsx": its
  header reads `"Date", "Real", "Time", <blank>, "Pressure", "Temperature"` over data columns
  `month, day, year, time-of-day, pressure, temperature`, so the bare `"Date"` header column
  actually holds only the MONTH sub-value and the bare `"Time"` header column holds the YEAR
  sub-value -- the genuine time-of-day values sit one column further right, under a blank
  (`"Unnamed: N"`) header the loader has no name-based reason to ever look at. `load_xlsx` now
  correctly refuses to join "Date" with "Time" as a companion pair (`_companion_is_time_of_day`
  vetoes it -- "Time"'s real values are a 4-digit year, not a time-of-day), and, with no other
  named datetime candidate or elapsed/clock fallback available, raises `ValueError` rather than
  silently reporting a fake, zero-duration "load" (every one of its ~68,000 rows previously
  collapsed onto the same nonsense instant, since the old code joined "Date" with "Time" anyway
  and dateutil happened to parse the resulting garbage string into some one fixed date). This is
  a genuine header defect in the source workbook, not a defect in the loader -- there is no
  general rule that recovers a datetime from an anonymous column just because the named ones
  are wrong. The file's own same-well `.csv` sibling (a different export, correctly labeled) is
  unaffected and still loads normally.