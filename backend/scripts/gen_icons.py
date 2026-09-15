"""
backend/scripts/gen_icons.py — generate PWA placeholder icons.

Run once:  python -m backend.scripts.gen_icons

Why generated rather than hand-drawn: a manifest that references a missing
icon fails the install criteria silently on Android (no "Add to Home Screen"
prompt appears, with no error), and iOS falls back to a screenshot of the
page. "Icons are a design task for later" therefore breaks the install step
that Day 15's entire push flow depends on.

The maskable variant keeps its artwork inside the inner 80% safe zone, which
is what stops Android cropping the shape into a circle and cutting it.
"""
from __future__ import annotations

from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ICON_DIR = _PROJECT_ROOT / "frontend" / "public" / "icons"

BG = (15, 23, 42)        # slate-900, matches manifest background_color
ACCENT = (239, 68, 68)   # red-500, the alert colour used across the dashboard
RING = (30, 41, 59)      # slate-800, matches theme_color


def gen_icon(size: int, path: Path, maskable: bool = False) -> None:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (size, size), color=BG)
    draw = ImageDraw.Draw(img)

    # Maskable icons get cropped to a circle/squircle by the launcher, so the
    # mark is drawn smaller to stay inside the 80% safe zone.
    scale = 0.52 if maskable else 0.68
    inset = (1 - scale) / 2
    box = [size * inset, size * inset, size * (1 - inset), size * (1 - inset)]

    ring_pad = size * 0.04
    draw.ellipse(
        [box[0] - ring_pad, box[1] - ring_pad, box[2] + ring_pad, box[3] + ring_pad],
        fill=RING,
    )
    draw.ellipse(box, fill=ACCENT)

    # Inner cut-out gives the mark a recognisable "aperture" silhouette at
    # 48px, where a plain disc is indistinguishable from any other app.
    c = size / 2
    r = size * scale * 0.17
    draw.ellipse([c - r, c - r, c + r, c + r], fill=BG)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    print(f"  wrote {path.relative_to(_PROJECT_ROOT)} ({size}x{size})")


def main() -> None:
    print(f"Generating PWA icons into {_ICON_DIR.relative_to(_PROJECT_ROOT)}")
    gen_icon(192, _ICON_DIR / "icon-192.png")
    gen_icon(512, _ICON_DIR / "icon-512.png")
    gen_icon(512, _ICON_DIR / "icon-512-maskable.png", maskable=True)
    print("Done.")


if __name__ == "__main__":
    main()
