# Handoff: `io_load.py` CSV loader fixes

Written 2026-08-05. Everything below was measured against the real corpus, not inferred. Where a
number appears, it came from running code against a named file.

## 1. The larger goal

`C:\DFIT Data` is a 29.9 GB pull of DFIT test data, 2,464 data files in 910 leaf folders. It is a
disposable cloud mirror: deleting and restructuring it is authorized.

`dfit_tool.store.scan_root` returns **2,299 queue entries** for it, but there are only **252
questionnaires** and **207 unique well names**. The cause is not the folder layout — nothing records
*which* file in a well folder is the test worth interpreting, so the scanner makes one entry per
data-file stem. One well folder becomes seven tests.

The fix is a triage tool (`scripts/dfit_triage.py` + `scripts/triage/`) that plots every file, lets a
human pick the keeper(s) per folder, then moves keepers into a `Basin/Well/` tree. That tool is
**built and reviewed** (4 Opus passes, all findings addressed). A complete scan exists at
`C:\DFIT Data\_triage`: 402 folders, 1,237 files, 1,140 rendered panels.

The user's decision was to **wait for the loader fixes below before doing the visual review**, because
a file that mis-loads shows a wrong plot and would be judged on it.

## 2. The immediate task

Four fixes to `dfit_tool/io_load.py`, the app's bottom-layer CSV loader. All four are implemented in
the working tree. **Review then found six defects in that implementation. Fixing those six is the
remaining work.**

Scope was explicitly authorized by the user, who chose "Timestamp parsing" + "Spotter preamble" when
asked whether to expand into `io_load.py` (the approved triage plan had put it out of scope).

### The four fixes as implemented

| Fix | What it does | Motivating file |
|---|---|---|
| A | Joins separate `Date` + `Time` columns; detects day-first dates | Strathcona `*-rt Aug15.csv` |
| B | Falls back to an elapsed-time column when the datetime column is unusable | `Goodnight_DFIT_data.csv` |
| C | Skips a leading preamble before the real header row | Spotter exports (48 files) |
| D | Reverses a wholly reverse-chronological export | Civitas Bijou SignalFire |

Composition order inside `load_csv`: read (C) → `suggest_channels` (A1) → build date+time string (A2)
→ parse with day-first sniff (A3) → reverse if wholly descending (D) → elapsed fallback if valid
fraction too low (B) → existing path.

## 3. State of the tree

- `dfit_tool/io_load.py` — modified, +203/−9. Not committed.
- `tests/test_csv_loader.py` — new, 30 tests. There was previously **no** test coverage of
  `load_csv` / `parse_datetime` / `suggest_channels` at all (`tests/test_dbs.py` covers only the
  binary loader).
- Suite: **612 passing, 0 failing** (582 pre-existing + 30 new).
- Nothing has been committed. Nothing under `C:\DFIT Data` has been moved or deleted;
  `dfit_triage.py apply` has never run with `--commit`.
- Unrelated pre-existing dirty files (do not touch): `CLAUDE.md`, `dfit_tool/model.py`,
  `dfit_tool/store.py`, `dfit_tool/ui.py`, several `tests/test_*`, `dfit_log.csv`, `scripts/`.

## 4. Commands

Dependencies are **not** on the system Python. Always:

```
C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe
```

Tests from the repo root:

```
C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest -q
```

Read `CLAUDE.md` at the repo root first. Hard constraints: `io_load.py` imports nothing from a higher
layer (no Tkinter, no matplotlib); `compute_all` stays the single source of truth; new logic lands in
a headless-testable layer with a test there.

## 5. Verification method that actually works

Unit tests did not catch any of the six defects. Two corpus-wide A/B comparisons did. Use both.

**(a) Whole-corpus loader A/B.** Compare `git show HEAD:dfit_tool/io_load.py` against the working
copy over all 1,145 CSVs under `C:\DFIT Data`, on every file where a fix can fire. This is what found
defects 1, 2 and 3.

**(b) Triage feature diff.** `scripts/triage/features.load_scan(r"C:\DFIT Data")` returns the cached
pre-fix measurements for 540 CSV signatures. Re-run `features.extract` on each and diff. Result of
the last run: **482 unchanged, 58 changed**. A working script is at
`<scratchpad>/corpus_diff.py` (~15 min, run it in the background). It independently confirmed the
defect-1 regression, which is why I trust it.

Method (b) only covers folders that have a questionnaire, so it **misses** the two defect-2 files.
Method (a) is the complete one.

## 6. The six defects to fix

Ranked. 1 and 2 are blockers.

### DEFECT 1 — regression: FIX A makes 7 loadable files unopenable

`load_csv` joins `date + " " + time` unconditionally once a companion `Time` column is found, with no
fallback when the joined string parses nothing.

File: `C:\DFIT Data\Lucero Energy\Tahu 2TF2H 3rd DFIT\TAHU 2TF2H DFIT pumping on Stage 1 perfs.csv`
Header: `Date,Time,Marker,Combined Flow Rate,Combined Flow Total,Max Pressure`
Row 1: `6/30/2023,15:58:17:647,,0,0,278`

`Time` is `HH:MM:SS:mmm` — a colon before the milliseconds. `"6/30/2023 15:58:17:647"` matches
neither `%m/%d/%Y %H:%M:%S`, nor the Excel-serial fallback, nor dateutil. All rows → NaT →
`valid_frac == 0.0` → no elapsed column → `ValueError("Could not parse any datetimes from column
'Date'")`. `ui._load_common` shows "Load failed" and the file cannot be opened. Before the change it
loaded 1,476 rows (degenerate all-zero `t_s`, since `Date` alone is one day).

All 7 paths:
```
Lucero Energy\Tahu 2TF2H\TAHU 2TFH DEFIT-e4cf7c9f-4a1b-454f-831c-029ada7e697a-2023-06-21-10-08.csv
Lucero Energy\Tahu 2TF2H\TAHU 3TFH DEFIT-e4cf7c9f-4a1b-454f-831c-029ada7e697a-2023-06-21-10-08.csv
Lucero Energy\Tahu 2TF2H 1st DFIT\TAHU 2TF2H DFIT pumping on Stage 1 perfs.csv
Lucero Energy\Tahu 2TF2H 1st DFIT\TAHU 2TFH DEFIT-e4cf7c9f-4a1b-454f-831c-029ada7e697a-2023-06-21-10-08.csv
Lucero Energy\Tahu 2TF2H 2nd DFIT\TAHU 2TF2H DFIT pumping on Stage 1 perfs.csv
Lucero Energy\Tahu 2TF2H 3rd DFIT\TAHU 2TF2H DFIT pumping on Stage 1 perfs.csv
Lucero Energy\Tahu 3MBH\TAHU 3MBH  DEFIT-0ac4ff04-9dff-4b8e-8210-6b58f4d6d3f3-2023-06-21-09-31.csv
```

Intended fix, both parts:
- **Never regress to unopenable.** Parse both the joined and the date-only series; keep whichever
  yields more valid values, joined winning ties. Measured basis: Strathcona joined 333,234 vs.
  date-only 231,423 (joined wins); Lucero joined 0 vs. date-only 1,476 (date-only wins).
- **Recover the real time base.** Normalize `HH:MM:SS:mmm` → `HH:MM:SS.mmm` before joining, only on
  a full-match of `\d{1,2}:\d{2}:\d{2}:\d{1,3}`, replacing only the last colon. Takes those 7 files
  from an all-zero `t_s` to a real sub-second one.

### DEFECT 2 — silent garbage: FIX C selects a *data* row as the header on 2 of 48 files

`_detect_header_skiprows` returns the first line whose field count equals the modal width, with no
check that the line is header-like. When the real header is **narrower** than the data rows (ragged
export / trailing comma), the modal width first appears on the first data row. This reaches exactly
the "parses successfully into silent garbage instead of erroring" outcome the spec forbade — just via
a narrow header rather than via the retry loop that was forbidden.

File: `C:\DFIT Data\Whiting Oil & Gas Corp\Horsetail 07E-0636\All CSV Data Whiting\Razor 12H 1316B.csv`
(547,554 data rows)
```
Time,Job Time                              <- real header, 2 fields
(min) ,(date time)                         <- units row, 2 fields
400.00000,12/16/2016 12:40:01 PM,          <- data, 3 fields (trailing comma)
```
modal = 3 → `skiprows=2` → first data row becomes the header. Measured: columns
`['400.00000', '12/16/2016 12:40:01 PM', 'Unnamed: 2']`, first data row silently dropped (547,553
kept), `suggest_channels` finds no datetime so `dt_col = '400.00000'` (elapsed **minutes**) is read
through the Excel-serial fallback into 1901 dates, and `t_s` spans 786,238,223 s = **24.9 years**.
Correct behavior for this file is to keep raising the original `ParserError`.

Second instance: `C:\DFIT Data\WPX Energy\WPX Olson 12-1H\Spreadsheets\Olson 12-1 HX Zones 8-11.csv`
— `skiprows=11` lands on an all-`0` INSITE metadata row (17 fields); the real header
`"Time","Treatment At Wellhead",...` at line 13 becomes a data row → columns `['0','0.1','0.2',...]`.

Only luck limits the damage: in both cases the garbage column names mean no pressure channel is
found, so the analyst sees "No pressure channel selected" rather than plausible wrong numbers.

Intended fix: **reject any candidate line in which a non-empty field parses as a bare number**
(`float()` on the stripped field). A header essentially never has a purely numeric field; a data row
almost always does. Return the first modal-width line with at least one non-empty field and no
numeric field; if none exists in the sampled window, return 0 so the original `ParserError`
propagates. Returning 0 is always safe — it restores pre-change behavior. Check `Pressure(psia)`,
`Row(s): 474847`, `Job ID: 9942`, `Delta(Hrs)`, `TZ` all read as non-numeric.

This may also fix the WPX file by continuing past the metadata row to the real header; either outcome
(fixed, or back to raising) is acceptable.

**Required verification:** enumerate every CSV under `C:\DFIT Data` where `pd.read_csv` raises
`ParserError` (measured: 48). Report detected `skiprows` and resulting columns for each. Confirm the
46 already-correct ones are unchanged, and that these still work:
- `Abraxas Petroleum Corp\Axas Caprito 99-202H\Caprito 99-202H_040218_1115093.csv` → 474,846 rows, `['Date/Time', 'TZ', 'Pressure(psia)']`
- `Hess\EN PERSON 11-22\EN_PERSON11-22 Casing.csv` → 222,753 rows, `['Date/Time', 'TZ', 'Pressure(psia)', 'Temperature(F)']`
- `Vesta Energy\Vesta 102_8-6-41-27-W4\13751-1115471.csv` → 24,296 rows, `['Datetime', 'TZ', 'Delta(Hrs)', 'Flow Rate(m3/min)', 'Totaliser(m3)']`
- `Bright Rock\SHULTZ FED 1114 34-73 N-DH\DT000208_EL.csv` → must still load (4-line preamble variant, recovers 733,045 rows)

### DEFECT 3 — the day-first sniff misses the direct siblings of the file it was written for

`_dayfirst_hint` requires a day-of-month > 12 somewhere in the column. A day-first record spanning
only days 1–12 of one month gives no such evidence.

File: `C:\DFIT Data\Strathcona Resources\100-01-28-061-03W6\100-01-28-061-03W6-rt.csv` (209,191 rows).
Unique `Date` values are exactly `{9/8/2022, 10/8/2022, 11/8/2022, 12/8/2022}` — a 3-day test, Aug
9–12. No first component exceeds 12 → hint False → parsed Sep 8 → Dec 8 → `t_s` spans **91.1 days**.

Measured (old span → current-code span):
```
100-01-28-061-03W6-rt Aug15.csv   91.00 d -> 6.01 d    fixed
100-01-28-061-03W6-rt.csv         91.00 d -> 91.13 d   STILL WRONG
102-02-09-062-03W6-rt Aug15.csv   91.00 d -> 6.04 d    fixed
102-02-09-062-03W6-rt.csv         91.00 d -> 91.16 d   STILL WRONG
102-02-28-061-03W6-rt Aug15.csv   91.00 d -> 6.06 d    fixed
102-02-28-061-03W6-rt.csv         91.00 d -> 91.19 d   STILL WRONG
```
The same well now reads 6 days or 91 days depending on which file in the folder is opened. That is
worse than being uniformly wrong.

A decisive signal exists: the second component is constant at `8` while the first increments
9→10→11→12. Over a record short enough to sit inside one month — which a DFIT is — the component
that **varies** is the day and the constant one is the month.

Intended fix: decide in this order, first rule to fire wins.
1. `max_first > 12` and `max_second <= 12` → **day-first** (proof: no month exceeds 12).
2. `max_second > 12` and `max_first <= 12` → **month-first** (symmetric proof).
3. Year constant, first component has exactly one distinct value, second has ≥2 → **month-first**
   (US file inside one month: `5/1`, `5/2`, `5/3`).
4. Year constant, second has exactly one distinct value, first has ≥2 → **day-first** (the
   Strathcona case).
5. Otherwise **month-first**, today's default (record crosses a month boundary; no signal).

Rules 3–4 need the year, so extend the regex to capture it, still requiring the first two components
to be 1–2 digits so a 4-digit ISO leading component never matches. If the year varies, fall to rule 5.

Expected result: the three `-rt.csv` files drop from ~91 d to ~3 d.

### DEFECT 4 — FIX B's `t_s` is not rebased to zero and carries NaN

`_find_elapsed_column` returns `vals * mult` verbatim; `load_csv` passes it through as `t_s`. But
`TestData.t_s` is documented as "elapsed seconds from first sample", which the datetime path
guarantees via `elapsed_seconds`.

Measured: `True Oil\Abra Data\spotter-14720-264041_EL.csv` → `t_s[0] = 10.00008`;
`spotter-14720-281575_EL.csv` → `t_s[0] = 1.00008`. Synthetic (`Delta(Hrs)` starting at 5.0 with a
blank cell) → `t_s = [18000.0, nan, 19800.0, 21600.0]`.

Downstream impact was checked, not assumed: `model.compute_all` works on `td.t_s - res.t_shutin_s`
(relative, so reported values are unaffected), `dt_all >= 0` drops NaN, and `picks` uses `nanargmin`.
The two observable effects are `plots.render_overview`'s "time from file start (h)" axis being offset
and the `t_start_inj` column in `store.py:358` carrying the same offset.

Intended fix: subtract the first finite value so `t_s[0] == 0.0`. Leave NaN as NaN — missing elapsed
data mirrors NaT in the datetime path — and comment why.

Verify `Goodnight_DFIT_data.csv` does not move: 1,048,575 rows, `t_s[-1]/3600 == 291.2705556`.

### DEFECT 5 — two tests claiming to pin the companion-time guard are vacuous

`tests/test_csv_loader.py` ~lines 27 and 34 are meant to pin the `"date" in name and "time" not in
name` clause in `suggest_channels`. Measured: deleting `and "time" not in lc[datetime_col]` from
`io_load.py:154` leaves **all 30 tests green**. Neither fixture has a separate column whose bare name
is `"time"`, so `time_col` stays `None` for the wrong reason.

Intended fix: add the discriminating inputs — `suggest_channels(["Date/Time", "Time", "PRESS"])`,
`["DateTime", "Time", "PRESS"]`, `["Timestamp (MST)", "Time", "PRESS"]` must all give `time is None`.
Then delete the clause, confirm a test fails, restore, confirm green.

### DEFECT 6 (low) — the header detector reads the whole file, and reader errors escape

`lines = [row for i, row in enumerate(reader) if i < 20]` filters but never stops. Measured: on a
400,003-line / 11.6 MB file all 400,003 lines are consumed to keep 20. Consequences: a 200,000-char
quoted field on line 4 raises `_csv.Error: field larger than field limit (131072)`, replacing the
informative `ParserError`; a non-UTF-8 byte past line 20 raises `UnicodeDecodeError`. No corpus file
hits either today.

Intended fix: `itertools.islice` to 20 lines; wrap in try/except for `csv.Error`,
`UnicodeDecodeError`, `OSError`, returning 0 so the original `ParserError` re-raises.

## 7. Things that are NOT bugs — do not "fix" them

Earlier in this work I framed the problem as "124 files lose rows to `pd.to_datetime` coercion,"
implying a parser overhaul. Measuring each file individually inverted that. Two of the biggest
apparent losses are legitimately garbage and are correctly discarded:

- `C:\DFIT Data\Crestone Peak\Reserve\3BH\21011234 raw data.csv` — 712,909 dropped rows are literal
  `#REF!` Excel errors from row 233,470. Header `Date Time, Pressure(psi), Temp(F)`.
- `C:\DFIT Data\Great Western\Seltzer\036HN\Seltzer Pump - 036HN.csv` — 5,419 dropped rows are blank,
  from row 1,020.

Both were verified byte-identical old vs. new under the current change. Keep them that way; they are
a useful regression canary because neither has an elapsed column, so FIX B must never fire on them.

Also out of scope: `load_dbs`'s trailing `idx == 0` padding rows. Real DBS files carry a block of
them, which resets `t_s` to 0 at the end. The triage layer already compensates via
`features.monotonic_prefix`; the app itself does not. Separately tracked — the user's `CLAUDE.md`
TODO list has "need to be able to trim tail".

## 8. Test expectations for the fixes

All fixtures synthetic under `tmp_path`. **No test may read anything under `C:\DFIT Data`.**

- D1: `HH:MM:SS:mmm` fixture loads with all rows parsed and increasing sub-second `t_s`; a
  garbage-`Time` fixture falls back to date-only and still loads; a fixture where joined beats
  date-only keeps joined. Pin the millisecond normalizer directly (rewrites `15:58:17:647`; leaves
  `15:58:17`, `15:58:17.647`, `8:23:17` alone).
- D2: the Razor shape returns 0 and `load_csv` raises; an all-numeric metadata row at modal width is
  skipped past to the real header; assert a returned header line never has a numeric first field.
- D3: table-driven over all five rules, plus end-to-end that the rule-4 fixture gives August dates
  and a ~3-day span.
- D4: elapsed starting at 5.0 h gives `t_s[0] == 0.0`; an interior blank keeps NaN and still rebases.
- D5: the three `suggest_channels` cases.
- D6: oversized quoted field past line 20 → detection returns 0, does not raise; same for a
  non-UTF-8 byte.

**Mutation-test every anti-regression guard**: break the production code, confirm a test fails,
restore exactly, re-run green. A guard that survives its mutation is pinning nothing. Six guards were
mutation-tested on the first pass and one (D5) still turned out vacuous, so verify by running, not by
reading.

## 9. After the fixes

1. `opus-reviewer` pass on the result. `io_load.py` is the core loader and 612 tests depend on it.
2. Re-run the corpus A/B. Confirm the 7 Lucero regressions are gone, the 3 Strathcona `-rt.csv`
   spans drop to ~3 d, and the two "not a bug" files stay byte-identical.
3. Bump `PNG_RENDER_VERSION` in `scripts/triage/features.py` (currently 3) and re-scan:
   `rm -rf "C:\DFIT Data\_triage"` then
   `python scripts\dfit_triage.py scan --root "C:\DFIT Data"` (~30 min).
   The bump matters because cached PNGs are keyed on content signature + version, so a stale panel
   would otherwise be reused next to fresh, disagreeing numbers.
4. Hand the review window to the user:
   `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe scripts\dfit_triage.py review --root "C:\DFIT Data"`
5. Note `scripts/triage/features.py:monotonic_prefix`'s docstring claims reverse-chronological files
   are deliberately unhandled and that zero corpus files reach that path. FIX D makes the first claim
   stale, and the second was wrong anyway (it measured negative durations, while genuinely reversed
   files were being silently truncated to `rows=1` — the 2 SignalFire files, now recovering 396,682
   rows). Update that docstring.

## 10. Process note worth carrying

Across this work, **the most serious defects were in specifications I wrote**, each correct for the
single case I had in mind and wrong against the corpus. Examples: grouping triage folders by
questionnaire directory (merged unrelated wells); treating a NaN timestamp as a record boundary;
inventing reverse-chronological detection, then deleting it on evidence that measured the wrong
thing; pinning `PNG_RENDER_VERSION` on a "~1% of files affected" premise that measured 22%.

The pattern: a rule that is obviously right for one file, applied to 1,145. The countermeasure that
actually worked is the corpus-wide A/B in section 5 — not more unit tests, and not more review of the
reasoning. Run the corpus comparison before believing any loader change.
