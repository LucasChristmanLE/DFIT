"""Pressure-increment resampling and diagnostic derivatives.

After shut-in the pressure declines monotonically. Sampling at a fixed *pressure* step (default
30 psi) instead of a fixed time step collapses ~10^5 raw rows to a few hundred that are dense early
and sparse late, which is exactly what makes the numerical derivatives (dP/dG, t*dP/dt) stable. This
replaces time-domain rolling-mean smoothing.

Sign convention for the diagnostic curves: pressure declines after shut-in, so d(BHP)/dG < 0. Every
derivative curve here is reported **positive-up for a declining pressure** (i.e. negated), matching
the way G-function and log-log diagnostic plots are conventionally drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gfunction import g_time

# Tail rise guard defaults. Fixed rather than derived from ``step`` -- a smaller resample step
# (endorsed by the reference methodology) must not silently tighten the guard and start
# discarding ordinary water-hammer ringing.
RISE_GUARD_PSI = 30.0
RISE_GUARD_SUSTAIN_S = 60.0
# On its own, RISE_GUARD_SUSTAIN_S can be satisfied by just two samples at >= 60 s spacing --
# effectively the old single-excursion behavior. Also require this many above-tolerance samples
# in the run (counting its first) before the duration check is allowed to fire.
RISE_GUARD_SUSTAIN_SAMPLES = 5


@dataclass
class Resampled:
    dt: np.ndarray   # elapsed seconds since shut-in (>= 0), pressure-increment spaced
    p: np.ndarray    # BHP at those points (psi), monotonically decreasing
    n_raw: int       # number of raw post-shut-in samples considered
    guarded_at: int | None = None  # resampled index where the tail guard stopped, if it did
    # dt of the FIRST sample of the sustained run that tripped the rise guard -- not the
    # confirming sample -- or None if the guard never fired. This is still the true boundary of
    # what the resampler actually consumed: the run itself contributes nothing to keep_dt, so up
    # to a full sustain window (minus one sample) can separate the last *kept* point (dt[-1])
    # from this boundary. Callers that need "everything the resampler looked at" (e.g. the
    # low-surface-pressure scan) should bound by guard_dt, not dt[-1].
    guard_dt: float | None = None


def resample_pressure_increment(
    dt: np.ndarray,
    p: np.ndarray,
    step: float = 30.0,
    rise_tol: float | None = None,
    sustain_s: float = RISE_GUARD_SUSTAIN_S,
    sustain_samples: int = RISE_GUARD_SUSTAIN_SAMPLES,
    stop_at_guard: bool = True,
) -> Resampled:
    """Keep a point each time BHP has dropped >= ``step`` psi below the last kept point.

    ``dt`` and ``p`` are the post-shut-in samples (dt >= 0, increasing). ``rise_tol`` (default =
    ``RISE_GUARD_PSI``, independent of ``step``) is the tail guard's tolerance above the running
    minimum. A single sample past that tolerance is not enough on its own -- gauge noise and
    water-hammer rebounds routinely poke above it for a few seconds. The guard only fires once a
    run of samples stays continuously above tolerance for >= ``sustain_s`` seconds AND contains
    >= ``sustain_samples`` such samples (counting the run's first) -- the sample-count floor
    matters at coarse (>= ``sustain_s``) sample spacing, where duration alone would already be
    satisfied by the run's second sample. Any sample at or below tolerance resets the run (and is
    itself processed normally); so does a non-finite sample -- continuity can't be confirmed
    across a dropout, so two excursions separated by missing data must not bridge into one fire.
    On fire, ``guard_dt`` is the dt of the FIRST sample of the run (not the confirming sample) and
    ``guarded_at`` is the count of points already kept at that moment -- both stay ``None`` unless
    a run actually satisfies both conditions before the record ends. Samples inside a candidate
    run are neither kept nor allowed to lower ``running_min``.

    ``stop_at_guard`` (default True) is what a fired guard does next: True stops resampling
    outright (the historic behavior -- everything past ``guard_dt`` is simply never consumed).
    False keeps going instead, exactly like an ordinary run that dips back below tolerance on
    its own: the run resets and every later sample is evaluated by the same plain
    running_min/step rule, still relative to the ORIGINAL (never reset, never frozen)
    running_min. This needs no special re-anchoring, because a rising excursion never satisfies
    the ordinary keep rule (``pi <= last_kept - step``) in the first place -- rejecting it as
    part of a guard run and rejecting it via the ordinary rule are the same outcome. If the
    excursion is later followed by a genuine further decline below the true historical minimum,
    that decline is picked up completely normally, with no monotonicity issue, since only
    genuinely new lows are ever kept. If the excursion never comes back down, no further points
    are kept past ``guard_dt`` in either mode -- mathematically honest, since there is no new
    information to resample there, not a gap to paper over with a fake point. In either mode,
    ``guard_dt``/``guarded_at`` are recorded once, from the FIRST run that satisfies both sustain
    conditions -- a later qualifying excursion (evaluated only when ``stop_at_guard=False``)
    never overwrites them. Callers that want the guard-excluded region masked out of the
    *result* (rather than never resampled at all) pass False here and mask afterward -- see
    ``model.compute_all``'s ``resampled_full`` and ``interpret.resolve_tail_cut_dt``.

    This is a vectorized replacement for a per-sample Python loop (kept, unchanged, as
    ``_resample_loop_reference`` -- see its docstring, and ``tests/test_resample_vectorized.py``
    for the fuzz test that pins agreement between the two). It's exact, not approximate, for two
    reasons. First, ``running_min`` in the loop is provably a plain cumulative min over the
    finite samples (``np.minimum.accumulate`` on ``p`` with non-finite samples mapped to
    ``+inf`` so they never lower it): the loop skips updating it on an above-tolerance ("run")
    sample, but such a sample is by construction already above ``running_min``, so it could
    never have lowered it anyway -- the skip changes nothing. Second, a run sample can never
    satisfy the keep rule (``pi <= last_kept - step``) either, since the running min always
    stays above ``last_kept - step`` (any sample that pushes it to or below that threshold is kept
    on the spot) and a run sample is above the running min -- so which samples get kept
    doesn't depend on the guard's run bookkeeping at all, only on truncation at the moment a
    guard fire actually breaks the loop. This means the kept points can be found by walking the
    cumulative-min array forward from one kept point to the next with ``np.searchsorted`` --
    since a kept sample's value is always exactly the running min at that instant (it's the same
    sample that just pushed the running min down to it), this takes as many steps as there are
    OUTPUT points (a few hundred), not one per raw row. The tail-guard runs are found the same
    way any run-length problem vectorizes: label each above-tolerance run's start position and
    forward-fill it across the run with ``np.maximum.accumulate``, then check the sustain
    conditions at every above-tolerance sample in one pass; the earliest sample anywhere that
    satisfies both is the loop's fire point, and ``stop_at_guard=True`` drops any kept index past
    it (none can fall inside the firing run itself, since a run sample is never kept). The
    vectorized path handles every input except a non-finite/non-positive ``step`` or a
    non-finite/negative ``rise_tol``, where the equivalence argument doesn't obviously extend and
    it defers to ``_resample_loop_reference`` outright.
    """
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    if rise_tol is None:
        rise_tol = RISE_GUARD_PSI
    if not (np.isfinite(step) and step > 0 and np.isfinite(rise_tol) and rise_tol >= 0):
        return _resample_loop_reference(dt, p, step, rise_tol, sustain_s, sustain_samples,
                                        stop_at_guard)

    n = len(p)
    finite = np.isfinite(p)
    finite_idx = np.flatnonzero(finite)
    if finite_idx.size == 0:
        return Resampled(dt=np.array([]), p=np.array([]), n_raw=n)

    i0 = int(finite_idx[0])
    # Cumulative min over finite samples from i0 onward -- non-finite samples are mapped to +inf
    # so they never lower it, matching the loop's plain `continue` on a non-finite sample.
    pf = np.where(finite, p, np.inf)
    cm = np.minimum.accumulate(pf[i0:])  # cm[k] <-> absolute index i0 + k
    m = n - i0

    # Tail-guard runs: above[k] (k >= 1, absolute index i0+k) mirrors the loop's
    # `pi > running_min + rise_tol` check, using cm[k - 1] as "running_min just before
    # processing this sample" (exact -- see the docstring paragraph above).
    above = np.zeros(m, dtype=bool)
    if m > 1:
        above[1:] = finite[i0 + 1:] & (p[i0 + 1:] > cm[:-1] + rise_tol)

    guard_dt: float | None = None
    guarded_at: int | None = None
    s_abs: int | None = None
    j_abs: int | None = None
    if above.any():
        starts = above.copy()
        starts[1:] &= ~above[:-1]
        idx_arr = np.arange(m)
        run_start = np.maximum.accumulate(np.where(starts, idx_arr, -1))
        rel_positions = np.flatnonzero(above)
        rs = run_start[rel_positions]
        run_len = rel_positions - rs + 1  # count including the run's first sample
        dt_seg = dt[i0:]
        # `rel_positions != rs` mirrors the loop only checking the fire condition on a run's
        # non-first sample.
        cond = ((rel_positions != rs) & (run_len >= sustain_samples)
                & (dt_seg[rel_positions] - dt_seg[rs] >= sustain_s))
        fire = np.flatnonzero(cond)
        if fire.size:
            first = fire[0]  # earliest (s, j) pair in time order -- matches "first fire overall"
            s_rel = int(rs[first])
            j_rel = int(rel_positions[first])
            s_abs = i0 + s_rel
            j_abs = i0 + j_rel
            guard_dt = float(dt_seg[s_rel])

    # Kept indices: walk the cumulative min forward from one kept point to the next. `cm` is
    # non-increasing, so searching `-cm` (non-decreasing) with searchsorted finds the first
    # position, after the current kept index, where cm has dropped to <= threshold -- which is
    # necessarily the sample that caused that drop (a run sample never lowers cm), i.e. exactly
    # the next kept sample. `step > 0` guarantees the threshold strictly decreases each time, so
    # this always makes progress.
    neg_cm = -cm
    kept_idx: list[int] = [i0]
    cur_rel = 0
    # Kept as np.float64, not a Python float, so a numpy float32 ``step`` promotes to float64 in
    # ``last_val - step`` exactly as it does in the loop (NEP 50).
    last_val = p[i0]
    while True:
        start = cur_rel + 1
        if start >= m:
            break
        pos = np.searchsorted(neg_cm[start:], -(last_val - step), side="left")
        if pos == m - start:
            break
        rel = start + pos
        abs_i = i0 + rel
        kept_idx.append(abs_i)
        last_val = p[abs_i]
        cur_rel = rel

    if s_abs is not None:
        guarded_at = sum(1 for idx in kept_idx if idx < s_abs)
    if j_abs is not None and stop_at_guard:
        # None of the dropped indices can sit inside [s, j] -- that whole span is the firing
        # run, and a run sample is never kept.
        kept_idx = [idx for idx in kept_idx if idx <= j_abs]

    kept_arr = np.array(kept_idx, dtype=int)
    return Resampled(
        dt=dt[kept_arr],
        p=p[kept_arr],
        n_raw=n,
        guarded_at=guarded_at,
        guard_dt=guard_dt,
    )


def _resample_loop_reference(
    dt: np.ndarray,
    p: np.ndarray,
    step: float,
    rise_tol: float | None,
    sustain_s: float,
    sustain_samples: int,
    stop_at_guard: bool,
) -> Resampled:
    """Reference implementation: the original, unoptimized per-sample scan -- see
    ``resample_pressure_increment`` for the algorithm. No longer called by
    ``resample_pressure_increment`` itself except as a fallback for the handful of degenerate
    parameter combinations the vectorized path's equivalence argument doesn't cover (a
    non-finite/non-positive ``step`` or a non-finite/negative ``rise_tol``); kept so tests can
    fuzz it against the fast path for agreement, the same way ``_dropout_scan_loop`` is kept
    below for ``detect_dropouts``."""
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    if rise_tol is None:
        rise_tol = RISE_GUARD_PSI

    keep_dt: list[float] = []
    keep_p: list[float] = []
    guarded_at: int | None = None
    guard_dt: float | None = None

    running_min = np.inf
    last_kept = np.inf
    run_start_dt: float | None = None
    run_start_guarded_at: int | None = None
    run_count = 0
    for i in range(len(p)):
        pi = p[i]
        if not np.isfinite(pi):
            # A dropout mid-run breaks continuity -- can't confirm the rise stayed sustained
            # across it, so reset conservatively rather than let two spikes separated by missing
            # data bridge into a false fire. Outside a run this is a no-op, same as always.
            run_start_dt = None
            run_count = 0
            continue
        # First finite point is always kept as the reference.
        if not keep_dt:
            keep_dt.append(dt[i])
            keep_p.append(pi)
            last_kept = pi
            running_min = pi
            continue
        if pi > running_min + rise_tol:
            # Above tolerance: extend the current run, or start a new one. Either way this
            # sample is excluded from keep_p/running_min -- it can't satisfy either condition,
            # the same as the ordinary keep rule would reject it anyway.
            if run_start_dt is None:
                run_start_dt = float(dt[i])
                run_start_guarded_at = len(keep_p)
                run_count = 1
            else:
                run_count += 1
                if (guard_dt is None and dt[i] - run_start_dt >= sustain_s
                        and run_count >= sustain_samples):
                    # First (and only) qualifying fire -- guard_dt is None is belt-and-suspenders
                    # against a later run re-firing.
                    guard_dt = run_start_dt
                    guarded_at = run_start_guarded_at
                    if stop_at_guard:
                        break
                    # Reset the run and keep going exactly like an ordinary run that dips back
                    # below tolerance on its own -- running_min/last_kept are left untouched
                    # (never reset, never frozen), so a genuine further decline below the true
                    # historical minimum is still picked up by the ordinary rule below, and a
                    # rise that never comes back down correctly earns no further kept points.
                    run_start_dt = None
                    run_count = 0
            continue
        # At or below tolerance: reset any in-progress run and fall through to normal
        # processing.
        run_start_dt = None
        run_count = 0
        running_min = min(running_min, pi)
        if pi <= last_kept - step:
            keep_dt.append(dt[i])
            keep_p.append(pi)
            last_kept = pi

    return Resampled(
        dt=np.array(keep_dt),
        p=np.array(keep_p),
        n_raw=int(len(p)),
        guarded_at=guarded_at,
        guard_dt=guard_dt,
    )


# Momentary near-zero pressure dropout detection (see ../CLAUDE.md's "Pressure dropouts"
# section). A brief gauge dropout -- pressure reads ~0 for a few seconds, then returns --
# otherwise pins resample_pressure_increment's running_min at the dip and fires the rise guard
# on the recovery, starving every downstream diagnostic. DROPOUT_MAX_S/DROPOUT_LEADIN_MATCH_PSI
# are guardrails found by the corpus stress test (see the plan's "Corpus evidence" section), not
# first-principles constants.
DROPOUT_FLOOR_FRAC = 0.10       # onset: sample falls to <= 10% of the pre-dip level
DROPOUT_MIN_REF_PSI = 100.0     # onset only when the pre-dip level is above this -- without this
                                # guard, noise on an already-dead channel re-triggers onset
                                # endlessly (22 million pseudo-events in the corpus scan).
DROPOUT_MAX_S = 600.0           # hard cap: a dip longer than this is never "momentary"
DROPOUT_LEADIN_MATCH_PSI = 150.0  # lead-in accepted only when the level just before it is
                                   # within this of the recovery level


@dataclass
class Dropout:
    """One masked momentary near-zero pressure dropout. ``dt_start``/``dt_end``/``n_samples``/
    ``p_min`` describe the dip run itself (the samples from onset through the last one before
    recovery) -- not the lead-in extension, if one was also masked ahead of it (see
    ``detect_dropouts``)."""
    dt_start: float
    dt_end: float
    n_samples: int
    p_min: float


def detect_dropouts(
    dt: np.ndarray,
    p: np.ndarray,
    floor_frac: float = DROPOUT_FLOOR_FRAC,
    min_ref: float = DROPOUT_MIN_REF_PSI,
    recover_tol: float = RISE_GUARD_PSI,
    sustain_s: float = RISE_GUARD_SUSTAIN_S,
    sustain_samples: int = RISE_GUARD_SUSTAIN_SAMPLES,
    max_s: float = DROPOUT_MAX_S,
    leadin_match: float = DROPOUT_LEADIN_MATCH_PSI,
) -> tuple[np.ndarray, list[Dropout]]:
    """Detect and mask momentary near-zero pressure dropouts in a post-shut-in record.

    ``dt``/``p`` are the raw post-shut-in samples (dt >= 0, increasing), on whatever channel the
    caller passes -- ``model.compute_all`` calls this on the raw mapped pressure channel, before
    any hydrostatic offset, so "near zero" means near zero on the gauge. Returns a bool mask
    aligned with ``p`` (True = masked) and the list of ``Dropout`` records found.

    Algorithm (finite samples only -- a non-finite sample is skipped, never part of a decision):
    1. ``ref`` is the most recent finite, unmasked sample value.
    2. Onset at sample i when ``ref > min_ref`` and ``p[i] <= floor_frac * ref``.
    3. The dip run is every sample from i onward that has not yet recovered, i.e.
       ``p < ref - recover_tol`` -- a single threshold off the ORIGINAL pre-dip ref throughout,
       not a separate "still near zero" band: on the motivating Encore record the recovery passes
       through an intermediate ramp sample, and a band-based run would end there without ever
       counting as recovered.
    4. Recovery is the first sample with ``p >= ref - recover_tol``. No recovery before the
       record ends -> not a dropout (a crash that never recovers is the low-pressure tail trim's
       job, not this detector's) -- scanning stops there, since there's nothing left to look at.
    5. Momentary = recovered AND NOT sustained AND duration < ``max_s``, where sustained =
       ``duration >= sustain_s AND count >= sustain_samples`` (the rise guard's own definition,
       counting the run's first sample) and duration = ``dt[last dip] - dt[onset]``. Only
       momentary dips are masked.
    6. Lead-in extension: ``R`` = median of the first 5 finite samples at/after recovery. Walk
       backward from onset-1 while ``p < R - recover_tol``, at most ``sustain_s`` before onset.
       The walked samples are masked only if the walk stopped on its own (didn't hit the time
       cap) at a sample within ``leadin_match`` of ``R`` -- otherwise only the dip itself is
       masked. This is what lets a slow pre-dip slide toward zero (not yet a dip on its own) get
       swept in, while a real step change that happens to straddle a one-sample glitch (a long
       plateau at a level nowhere near the recovery) does not.
    7. Continue scanning after recovery; ``ref`` resets from that unmasked (recovered) sample.

    Most records have no dropout at all, and even a record that does have one spends the vast
    majority of its length outside any dip -- so the scan itself jumps straight between the
    vectorized pre-check's candidate onset positions (``_dropout_candidates``) instead of
    stepping sample by sample there; see ``_dropout_scan_candidates`` for why that's exact, not
    approximate. Only the dip-interior recovery scan and lead-in walk still step sample by
    sample, since both are bounded by the event itself (a handful to a few hundred samples), not
    by the whole record.
    """
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    n = len(p)
    candidate_idx, candidate_ref = _dropout_candidates(p, floor_frac, min_ref)
    if not len(candidate_idx):
        return np.zeros(n, dtype=bool), []
    return _dropout_scan_candidates(dt, p, candidate_idx, candidate_ref, recover_tol, sustain_s,
                                    sustain_samples, max_s, leadin_match)


def _dropout_candidates(
    p: np.ndarray, floor_frac: float, min_ref: float
) -> tuple[np.ndarray, np.ndarray]:
    """Every position ``_dropout_scan_candidates`` needs to check for onset, computed in one
    vectorized pass, plus the ``ref`` each one is checked against.

    Onset is only ever checked outside an active dip run (a dip's own interior samples are
    compared only against the frozen pre-dip ``ref``, never re-checked for a fresh onset), and
    outside a dip, ``ref`` -- "the most recent finite, unmasked sample value" -- is always
    exactly the immediately preceding FINITE sample: it is set from the previous iteration's
    plain per-sample branch, or reset to the recovered sample right after a dip resolves, with
    intervening non-finite samples skipped without updating it. So every position where onset
    could possibly fire is one of the consecutive pairs in the finite-only subsequence of ``p``,
    with ``ref`` = that pair's earlier element -- this is exact (not merely a necessary
    condition), because a candidate's ``ref`` never needs to be re-derived from anything other
    than ``p`` itself: even immediately after a dip resolves, the reset ``ref`` (the recovered
    sample) IS that sample's immediately-preceding-finite-sample relationship to whatever comes
    next, so it already appears correctly in this same global computation. A candidate landing
    inside a dip that a still-earlier candidate already resolved is simply never reached by the
    scan (it jumps straight past the whole resolved range) -- see ``_dropout_scan_candidates``.

    Returns ``(candidate_idx, candidate_ref)`` in increasing order of index, both empty if there
    is no possible onset anywhere in ``p``.
    """
    finite_idx = np.flatnonzero(np.isfinite(p))
    if finite_idx.size < 2:
        return np.empty(0, dtype=finite_idx.dtype), np.empty(0, dtype=float)
    vals = p[finite_idx]
    prev, cur = vals[:-1], vals[1:]
    hit = (prev > min_ref) & (cur <= floor_frac * prev)
    return finite_idx[1:][hit], prev[hit]


def _dropout_onset_possible(p: np.ndarray, floor_frac: float, min_ref: float) -> bool:
    """Whether ``_dropout_candidates`` finds any possible onset at all -- kept as its own
    function since it reads more directly than checking ``len(...) > 0`` at call sites (and a
    test pins it directly)."""
    return len(_dropout_candidates(p, floor_frac, min_ref)[0]) > 0


def _dropout_scan_candidates(
    dt: np.ndarray,
    p: np.ndarray,
    candidate_idx: np.ndarray,
    candidate_ref: np.ndarray,
    recover_tol: float,
    sustain_s: float,
    sustain_samples: int,
    max_s: float,
    leadin_match: float,
) -> tuple[np.ndarray, list[Dropout]]:
    """The real scan: jump directly between ``candidate_idx`` positions instead of stepping
    sample by sample outside a dip -- see ``detect_dropouts``/``_dropout_candidates`` for why
    this is exact. The in-dip recovery scan and lead-in walk are untouched from the original
    per-sample loop (``_dropout_scan_loop``, kept as a reference implementation for tests):
    both are bounded by the event, not by the whole record, so there is nothing to jump between
    there.

    After a dip resolves at ``recovered_at``, every candidate with index ``<= recovered_at`` is
    skipped outright (``np.searchsorted(..., side="right")``) -- some of those candidates may sit
    inside the dip's own interior with a ``ref`` attributed by the global pass above that would
    be wrong if it were ever used (a dip's true, frozen ``ref`` is the PRE-dip level, not each
    interior sample's own predecessor) -- but it never is: the scan only ever reads a candidate's
    ``ref`` once, at the moment it becomes the current onset, and any candidate inside an
    already-resolved dip is never visited at all.
    """
    n = len(p)
    mask = np.zeros(n, dtype=bool)
    events: list[Dropout] = []

    pos = 0
    n_cand = len(candidate_idx)
    while pos < n_cand:
        onset = int(candidate_idx[pos])
        ref = float(candidate_ref[pos])
        last_dip = onset
        p_min = float(p[onset])
        count = 1
        recovered_at: int | None = None
        j = onset + 1
        while j < n:
            pj = p[j]
            if not np.isfinite(pj):
                j += 1
                continue
            if pj >= ref - recover_tol:
                recovered_at = j
                break
            last_dip = j
            p_min = min(p_min, pj)
            count += 1
            j += 1
        if recovered_at is None:
            # Never recovers before the record ends -- a crash, not a dropout. Nothing left to
            # scan (matches _dropout_scan_loop: no later candidate can start a fresh dip once the
            # record itself has run out).
            break
        duration = float(dt[last_dip] - dt[onset])
        sustained = duration >= sustain_s and count >= sustain_samples
        momentary = (not sustained) and duration < max_s
        if momentary:
            leadin_lo = onset
            rec_vals: list[float] = []
            k = recovered_at
            while k < n and len(rec_vals) < 5:
                if np.isfinite(p[k]):
                    rec_vals.append(float(p[k]))
                k += 1
            R = float(np.median(rec_vals)) if rec_vals else ref
            s = onset - 1
            stop_idx: int | None = None
            hit_cap = False
            while s >= 0:
                if dt[onset] - dt[s] > sustain_s:
                    hit_cap = True
                    break
                ps = p[s]
                if not np.isfinite(ps):
                    s -= 1
                    continue
                if ps < R - recover_tol:
                    s -= 1
                    continue
                stop_idx = s
                break
            if (not hit_cap and stop_idx is not None
                    and abs(p[stop_idx] - R) <= leadin_match):
                leadin_lo = stop_idx + 1
            mask[leadin_lo:last_dip + 1] = True
            events.append(Dropout(dt_start=float(dt[onset]), dt_end=float(dt[last_dip]),
                                   n_samples=count, p_min=float(p_min)))
        # Skip every remaining candidate at/before recovered_at -- ref resets from the unmasked
        # (recovered) sample there either way, matching _dropout_scan_loop's ``i = recovered_at
        # + 1``.
        pos = int(np.searchsorted(candidate_idx, recovered_at, side="right"))
    return mask, events


def _dropout_scan_loop(
    dt: np.ndarray,
    p: np.ndarray,
    floor_frac: float,
    min_ref: float,
    recover_tol: float,
    sustain_s: float,
    sustain_samples: int,
    max_s: float,
    leadin_match: float,
) -> tuple[np.ndarray, list[Dropout]]:
    """Reference implementation: the original, unoptimized per-sample scan -- see
    ``detect_dropouts`` for the algorithm. No longer called by ``detect_dropouts`` itself
    (``_dropout_scan_candidates`` is, for performance -- a record with a real dropout otherwise
    still pays the full per-sample cost outside the dip, which is most of the record); kept only
    so tests can fuzz it against the fast path for agreement."""
    n = len(p)
    mask = np.zeros(n, dtype=bool)
    events: list[Dropout] = []

    ref: float | None = None
    i = 0
    while i < n:
        pi = p[i]
        if not np.isfinite(pi):
            i += 1
            continue
        if ref is not None and ref > min_ref and pi <= floor_frac * ref:
            onset = i
            last_dip = onset
            p_min = pi
            count = 1
            recovered_at: int | None = None
            j = onset + 1
            while j < n:
                pj = p[j]
                if not np.isfinite(pj):
                    j += 1
                    continue
                if pj >= ref - recover_tol:
                    recovered_at = j
                    break
                last_dip = j
                p_min = min(p_min, pj)
                count += 1
                j += 1
            if recovered_at is None:
                # Never recovers before the record ends -- a crash, not a dropout. Nothing left
                # to scan.
                break
            duration = float(dt[last_dip] - dt[onset])
            sustained = duration >= sustain_s and count >= sustain_samples
            momentary = (not sustained) and duration < max_s
            if momentary:
                leadin_lo = onset
                rec_vals: list[float] = []
                k = recovered_at
                while k < n and len(rec_vals) < 5:
                    if np.isfinite(p[k]):
                        rec_vals.append(float(p[k]))
                    k += 1
                R = float(np.median(rec_vals)) if rec_vals else ref
                s = onset - 1
                stop_idx: int | None = None
                hit_cap = False
                while s >= 0:
                    if dt[onset] - dt[s] > sustain_s:
                        hit_cap = True
                        break
                    ps = p[s]
                    if not np.isfinite(ps):
                        s -= 1
                        continue
                    if ps < R - recover_tol:
                        s -= 1
                        continue
                    stop_idx = s
                    break
                if (not hit_cap and stop_idx is not None
                        and abs(p[stop_idx] - R) <= leadin_match):
                    leadin_lo = stop_idx + 1
                mask[leadin_lo:last_dip + 1] = True
                events.append(Dropout(dt_start=float(dt[onset]), dt_end=float(dt[last_dip]),
                                       n_samples=count, p_min=float(p_min)))
            # Continue scanning after recovery either way -- ref resets from the unmasked
            # (recovered) sample, whether or not this run turned out to be momentary.
            ref = float(p[recovered_at])
            i = recovered_at + 1
            continue
        ref = pi
        i += 1
    return mask, events


@dataclass
class Diagnostics:
    G: np.ndarray         # G-time
    dPdG: np.ndarray      # first derivative dP/dG, positive-up
    GdPdG: np.ndarray     # superposition semilog derivative G*dP/dG, positive-up
    d2PdG2: np.ndarray    # slope of the (positive-up) dP/dG curve: np.gradient(dPdG, G)
    # log-log falloff diagnostics vs actual shut-in time:
    t: np.ndarray         # shut-in elapsed time (> 0 only)
    p: np.ndarray         # BHP aligned with t (psi)
    dp: np.ndarray        # pressure drop since first resampled point, positive-up
    tdpdt: np.ndarray     # t * dP/dt (log-log derivative), positive-up


def diagnostics(rs: Resampled, te: float, alpha: float = 1.0) -> Diagnostics:
    """Compute G-function and log-log diagnostic curves from a resampled falloff.

    All derivatives use ``np.gradient`` against the actual abscissa (G or t), so they are correct on
    the non-uniform pressure-increment spacing.
    """
    dt = rs.dt
    p = rs.p
    G = g_time(dt, te, alpha)

    # G-function derivatives (positive-up: negate because p declines as G grows).
    dPdG = -np.gradient(p, G)
    GdPdG = G * dPdG
    d2PdG2 = np.gradient(dPdG, G) if len(G) > 1 else np.zeros_like(G)

    # Log-log falloff: use strictly positive shut-in times.
    pos = dt > 0
    t = dt[pos]
    p_pos = p[pos]
    dp = p_pos[0] - p_pos if len(p_pos) else p_pos
    tdpdt = -t * np.gradient(p_pos, t) if len(p_pos) > 1 else np.zeros_like(t)

    return Diagnostics(G=G, dPdG=dPdG, GdPdG=GdPdG, d2PdG2=d2PdG2, t=t, p=p_pos, dp=dp, tdpdt=tdpdt)
