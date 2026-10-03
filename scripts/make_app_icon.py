"""Build the app icon: the Liberty mark over a tapered pressure fall-off curve.

    python scripts/make_app_icon.py [--source path/to/lib_l_mark_square.jpeg]

Writes dfit_tool/assets/app_icon.png (256 px, used by Tk) and app_icon.ico (16-256 px, used
for the Windows taskbar). --source re-cuts dfit_tool/assets/liberty_mark.png from the square
white-background JPEG; without it the committed mark is used. Needs Pillow (a matplotlib
dependency).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
from PIL import Image, ImageFilter

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "dfit_tool" / "assets"
MARK = ASSETS / "liberty_mark.png"
LBRT_BLACK = "#262626"
ICO_SIZES = [16, 20, 24, 32, 40, 48, 56, 64, 80, 96, 128, 256]  # 32*{1,1.25,...,3} for the taskbar
MASTER_PX = 1024
SUPERSAMPLE = 4
MARK_X, MARK_Y, MARK_W = 0.15, 0.36, 0.80  # mark left, bottom, width (fractions)


def cut_mark(source: pathlib.Path) -> None:
    """Knock out the white background and crop to the ellipse."""
    a = np.asarray(Image.open(source).convert("RGBA")).astype(float)
    a[..., 3] = np.clip((255 - a[..., :3].min(axis=2)) / 40, 0, 1) * 255
    Image.fromarray(a[245:530, 25:750].astype(np.uint8)).save(MARK)


def falloff(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A hyperbolic decline from the top-left corner that flattens under the mark."""
    u = np.linspace(0, 1, n)
    x = 0.08 + 0.84 * u
    y = 0.03 + (0.86 - 0.03) / (1 + 25.0 * u) ** 0.7
    return x, y, u


def tapered_stroke(ax, x, y, u, w0: float, w1: float, color: str) -> None:
    """A filled stroke whose width runs linearly from w0 to w1, with square-cut ends."""
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    nx, ny = -dy / norm, dx / norm
    half = (w0 + (w1 - w0) * u) / 2
    pts = np.r_[np.c_[x + nx * half, y + ny * half], np.c_[x - nx * half, y - ny * half][::-1]]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=2))


def _render_tile_and_curve(px: int) -> Image.Image:
    """Tile and curve drawn at 4x and box-filtered down, so each size is drawn natively."""
    ss = px * SUPERSAMPLE
    fig = plt.figure(figsize=(ss / 100, ss / 100), dpi=100)
    fig.patch.set_alpha(0)  # transparent outside the rounded tile
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    small = px <= 48
    # The grey outline is a blurry 1 px ring at small sizes; drop it there.
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.14",
                                fc="white", ec="none" if small else "#D0D0D0",
                                lw=4.8 * ss / MASTER_PX, zorder=0))
    x, y, u = falloff()
    # Keep the thin end at least ~1 px wide so it does not vanish at 16-32 px.
    tapered_stroke(ax, x, y, u, max(0.012, 1.2 / px), 0.075, LBRT_BLACK)
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy())
    plt.close(fig)
    return img.resize((px, px), Image.BOX)


def render_icon(px: int) -> Image.Image:
    """One icon frame at `px`: the mark is resampled straight from the source to its pixel size."""
    img = _render_tile_and_curve(px)
    mark = Image.open(MARK).convert("RGBA")
    w = round(MARK_W * px)
    h = round(w * mark.height / mark.width)
    mark = mark.resize((w, h), Image.LANCZOS)
    if px <= 48:
        mark = mark.filter(ImageFilter.UnsharpMask(radius=0.6, percent=60, threshold=0))
    left = round(MARK_X * px)
    top = px - round(MARK_Y * px) - h
    img.alpha_composite(mark, (left, top))
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=pathlib.Path)
    args = ap.parse_args()
    if args.source:
        cut_mark(args.source)
    frames = {px: render_icon(px) for px in ICO_SIZES}
    frames[256].save(ASSETS / "app_icon.png")
    frames[256].save(ASSETS / "app_icon.ico", sizes=[(s, s) for s in ICO_SIZES],
                     append_images=[frames[s] for s in ICO_SIZES if s != 256])
    print(f"wrote {ASSETS / 'app_icon.png'} and {ASSETS / 'app_icon.ico'}")


if __name__ == "__main__":
    main()
