"""Matplotlib rendering for triage: one small PNG per file (cached, keyed by content signature)
and the per-folder review grid that assembles those cached PNGs into one screen.

OO API only (`Figure`, no `pyplot`) -- no Tkinter either -- so both renderers run headlessly
under Agg, matching the layering rule `dfit_tool/plots.py` follows.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib.image as mpimg
import numpy as np
from matplotlib.figure import Figure

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dfit_tool import io_load  # noqa: E402
from dfit_tool.plots import _decimate  # noqa: E402

from .features import FileFeatures, FolderScan, monotonic_prefix, png_path_for  # noqa: E402

_ERROR_VERDICTS = ("load_error", "no_pressure")
_GRID_NCOLS = 4


# --------------------------------------------------------------------------------------------------
# per-file PNG
# --------------------------------------------------------------------------------------------------
def _error_panel(fig, feat: FileFeatures, message: str) -> None:
    """A panel with no axes -- just the filename and an error message, centered -- so the
    reviewer can tell "broken file" from "loaded fine but rendered nothing"."""
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.5, 0.5, f"{os.path.basename(feat.path)}\n\n{message}",
        ha="center", va="center", wrap=True, fontsize=9, transform=ax.transAxes,
    )


def render_file_png(feat: FileFeatures, out_path: str, figsize=(6.0, 3.2), dpi: int = 90) -> str:
    """Render `feat`'s raw pressure channel (and rate, if present) against elapsed hours to
    `out_path`. No BHP conversion -- mud weight/TVD aren't available here and shape is all that
    matters for triage. `load_error`/`no_pressure` files get an axes-free error panel instead;
    any other failure while reloading/plotting falls back to the same kind of panel rather than
    raising, since a broken triage PNG must never abort a batch render."""
    fig = Figure(figsize=figsize)

    if feat.verdict in _ERROR_VERDICTS:
        _error_panel(fig, feat, feat.load_error or "no recognizable pressure channel")
        fig.savefig(out_path, dpi=dpi)
        return out_path

    try:
        td = io_load.load(feat.path)
        guess = io_load.suggest_channels(td.columns)
        pressure_col = feat.pressure_col or guess.get("pressure")
        rate_col = feat.rate_col if feat.rate_col is not None else guess.get("rate")
        if not pressure_col:
            raise ValueError("no pressure column found on reload")

        # FIX 1: drop non-finite t_s samples BEFORE calling monotonic_prefix -- same masking step
        # `extract` applies, and for the same reason: a NaN timestamp compares False against
        # everything, so handing one to monotonic_prefix either raises or silently truncates the
        # record at the first gap. Then truncate to the genuinely usable span of what remains --
        # same helper `extract` uses -- so a trailing block of DBS `idx == 0` padding rows never
        # draws a spurious return-to-zero segment that would disagree with the numbers in the
        # annotation below.
        t_s_full = np.asarray(td.t_s, dtype=float)
        finite_t = np.isfinite(t_s_full)
        t_s_nonan = t_s_full[finite_t]
        mono = monotonic_prefix(t_s_nonan)
        t_h = t_s_nonan[mono] / 3600.0
        p = td.column(pressure_col)[finite_t][mono]
        xt, xp = _decimate(t_h, p)

        ax = fig.add_subplot(111)
        ax.plot(xt, xp, color="black", lw=0.8)
        ax.set_xlabel("elapsed time (h)")
        ax.set_ylabel(f"pressure ({feat.unit_guess})" if feat.unit_guess else "pressure (raw)")
        ax.grid(True, alpha=0.3)

        if rate_col:
            rate = td.column(rate_col)[finite_t][mono]
            ax2 = ax.twinx()
            _, xr = _decimate(t_h, rate)
            ax2.plot(xt, xr, color="tab:blue", lw=0.6, alpha=0.6)
            ax2.set_ylabel("rate", color="tab:blue")
            ax2.tick_params(axis="y", labelcolor="tab:blue")

        if feat.duration_hr is not None and feat.post_shutin_hr is not None:
            shutin_hr = feat.duration_hr - feat.post_shutin_hr
            ax.axvline(shutin_hr, color="tab:red", lw=1.0, ls="--")

        lines = []
        if feat.duration_hr is not None:
            lines.append(f"dur {feat.duration_hr:.1f}h")
        if feat.post_shutin_hr is not None:
            lines.append(f"post-SI {feat.post_shutin_hr:.1f}h")
        if feat.drop is not None:
            lines.append(f"drop {feat.drop:.0f} {feat.unit_guess}".rstrip())
        if feat.decline_fraction is not None:
            lines.append(f"decl {feat.decline_fraction:.2f}")
        if feat.trailing_dropped:
            lines.append(f"trailing pad dropped: {feat.trailing_dropped}")
        # FIX 4 (third review round): conditional on non-zero, same as trailing_dropped above.
        # PNG_RENDER_VERSION was bumped for this change (see its definition in features.py): a
        # corpus measurement showed roughly 275 of 1,237 files (~22%) carry a non-zero
        # nonfinite_time_dropped, not the ~1% originally assumed, and this annotation is the only
        # channel by which a human reviewer learns samples were dropped -- so every cached PNG
        # must be invalidated and re-rendered for it to ever appear.
        if feat.nonfinite_time_dropped:
            lines.append(f"nonfinite timestamps dropped: {feat.nonfinite_time_dropped}")
        lines.append(f"verdict: {feat.verdict}")
        if feat.same_bytes_as:
            others = feat.same_bytes_as
            note = f"also at: {os.path.basename(others[0])}"
            if len(others) > 1:
                note += f" (+{len(others) - 1} more)"
            lines.append(note)

        ax.text(
            0.02, 0.97, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
            fontsize=7, bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        ax.set_title(os.path.basename(feat.path), fontsize=9)
        fig.subplots_adjust(left=0.14, right=0.88, bottom=0.18, top=0.88)
        fig.savefig(out_path, dpi=dpi)
    except Exception as e:
        fig = Figure(figsize=figsize)
        _error_panel(fig, feat, f"{type(e).__name__}: {e}")
        fig.savefig(out_path, dpi=dpi)

    return out_path


# --------------------------------------------------------------------------------------------------
# per-folder review grid
# --------------------------------------------------------------------------------------------------
def page_count(scan: FolderScan, per_page: int = 8) -> int:
    return max(1, math.ceil(len(scan.files) / per_page))


def render_grid(
    scan: FolderScan,
    root: str,
    page: int = 0,
    per_page: int = 8,
    keeps: set[str] = frozenset(),
    figsize=(15.0, 9.0),
) -> Figure:
    """One page of `scan`'s files, `per_page` panels laid out `_GRID_NCOLS` wide, each `imshow`ing
    its cached `render_file_png` output. Panel N (1-indexed within the page, matching the
    review keys `1`-`8`) is titled `"N. <filename>"`. Spine color/weight encodes state: thick
    green for a kept file, thick amber dashed for `scan.suggested` when not (yet) kept, thin grey
    otherwise -- EXCEPT a `verdict == "duplicate"` panel, which always gets a thick dotted blue
    spine (regardless of keep/suggested state) plus a `"DUPLICATE"`-prefixed title and an overlay
    naming its representative (`dup_of`), so it can never be mistaken for an independent
    candidate. `png_path_for` is keyed by content signature, so a duplicate's cached PNG is
    the REPRESENTATIVE's plot/annotation (same bytes, different path) -- the overlay is what
    actually tells the reviewer this panel is not its own file. A missing cached PNG gets a
    "(no plot)" placeholder rather than raising. Cells past the end of this page's files are
    hidden."""
    fig = Figure(figsize=figsize)
    nrows = max(1, math.ceil(per_page / _GRID_NCOLS))

    start = page * per_page
    files = scan.files[start:start + per_page]

    for i in range(per_page):
        ax = fig.add_subplot(nrows, _GRID_NCOLS, i + 1)
        if i >= len(files):
            ax.axis("off")
            continue

        feat = files[i]
        is_duplicate = feat.verdict == "duplicate"
        ax.set_xticks([])
        ax.set_yticks([])
        title_prefix = "DUPLICATE: " if is_duplicate else ""
        ax.set_title(f"{i + 1}. {title_prefix}{os.path.basename(feat.path)}", fontsize=8)

        png_path = png_path_for(root, feat)
        if os.path.exists(png_path):
            try:
                ax.imshow(mpimg.imread(png_path))
            except Exception:
                ax.text(0.5, 0.5, "(no plot)", ha="center", va="center", transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "(no plot)", ha="center", va="center", transform=ax.transAxes)

        if is_duplicate:
            rep_name = os.path.basename(feat.dup_of) if feat.dup_of else "?"
            ax.text(
                0.5, 0.5, f"DUPLICATE OF\n{rep_name}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, fontweight="bold", color="tab:blue",
                bbox=dict(boxstyle="round", fc="white", alpha=0.75),
            )

        is_keep = feat.path in keeps
        is_suggested = feat.path in scan.suggested and not is_keep
        if is_duplicate:
            color, lw, ls = "tab:blue", 2.5, "dotted"
        elif is_keep:
            color, lw, ls = "green", 3.0, "solid"
        elif is_suggested:
            color, lw, ls = "darkgoldenrod", 3.0, "dashed"
        else:
            color, lw, ls = "grey", 0.8, "solid"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(lw)
            spine.set_linestyle(ls)

    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.93, wspace=0.15, hspace=0.35)
    return fig
