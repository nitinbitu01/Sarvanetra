"""backend/scripts/build_osd_masks.py — locate each camera's burned-in overlay.

WHY THIS EXISTS
  These feeds carry a station caption burned into the video ("Chiman bhai
  Bridge CSITMS-32_P"). It is large, white, high-contrast text present in every
  single frame, and it has now corrupted three separate results:

    1. Phase 2 v1 - 11.2% "findable plates" were mostly 'Bridge' and 'bhai'
    2. the camera survey - inflated text-region counts
    3. plate detector v1 - fires on the caption at 350px wide, four times
       wider than any plate it was trained on, giving a fake 87.5% hit rate
       on CAM_01

  Filtering by text content ('BRIDGE', 'BHAI'...) only ever caught the cases
  we had already seen. The overlay needs removing at the pixel level, before
  any detector or OCR looks at the frame.

HOW IT IS FOUND - no hardcoded coordinates
  A burned-in overlay is, by definition, the part of the image that never
  changes. Sample N frames spread across a clip and compute each pixel's
  temporal standard deviation:

      moving traffic, shadows, lighting  -> high variance
      burned-in caption                  -> variance ~0

  Static-and-bright pixels are the overlay. Road markings are also static but
  dark and low-contrast, so requiring brightness separates them. The mask is
  dilated slightly to cover anti-aliased glyph edges.

  This adapts per camera automatically - no coordinates to maintain, and it
  will find any future overlay at any position.

USAGE
  python -m backend.scripts.build_osd_masks
  python -m backend.scripts.build_osd_masks --cameras CAM_01 --samples 60
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np

OUT = Path("config/osd_masks")


def mask_for_clip(clip: Path, samples: int, std_thresh: float,
                  bright_thresh: int, edge_thresh: float = 8.0,
                  max_frac: float = 0.06) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(clip))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, samples).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 8:
        return None

    stack = np.stack(frames)
    std = stack.std(axis=0)
    mean = stack.mean(axis=0)
    # "Static and bright" alone is not enough: sunlit concrete, walls and sky
    # are also static and bright, and on CAM_01/CAM_05 that masked 20-33% of
    # the frame. Text has a third property those lack - strong local edges.
    # Requiring high gradient keeps glyph strokes and drops flat bright areas.
    g = cv2.Laplacian(cv2.GaussianBlur(mean.astype(np.uint8), (3, 3), 0),
                      cv2.CV_32F)
    edge = cv2.dilate(np.abs(g), np.ones((5, 5), np.uint8))
    m = ((std < std_thresh) & (mean > bright_thresh) &
         (edge > edge_thresh)).astype(np.uint8) * 255
    # Join glyph strokes into solid blocks, then cover anti-aliased edges.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 25), np.uint8))
    m = cv2.dilate(m, np.ones((7, 7), np.uint8), iterations=1)
    # Drop specks: a caption is a large connected region.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 400:
            out[lab == i] = 255

    # A caption occupies a small slice of the frame. If the mask is huge, the
    # thresholds have latched onto scenery rather than text - keep only the
    # largest components until within budget, rather than returning a mask
    # that would blind the detector to a third of the image.
    if (out > 0).mean() > max_frac:
        comps = sorted(((stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n)
                        if stats[i, cv2.CC_STAT_AREA] >= 400), reverse=True)
        out = np.zeros_like(m)
        budget = max_frac * m.size
        used = 0
        for area, i in comps:
            if used + area > budget:
                break
            out[lab == i] = 255
            used += area
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default="ALL")
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--std-thresh", type=float, default=6.0,
                    help="Pixel temporal std below this counts as static.")
    ap.add_argument("--bright-thresh", type=int, default=150,
                    help="Static pixels darker than this are road, not caption.")
    args = ap.parse_args()

    root = Path("data/clips")
    cams = (sorted(d.name for d in root.iterdir() if d.is_dir())
            if args.cameras.strip().upper() == "ALL"
            else [c.strip() for c in args.cameras.split(",") if c.strip()])

    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    print(f"building OSD masks for {len(cams)} cameras\n")

    for cam in cams:
        clips = sorted((root / cam).glob("*.mp4"))
        if not clips:
            continue
        # Prefer a daylight clip: the caption is most separable when the scene
        # behind it is bright and busy.
        clip = next((c for c in clips if c.stem.endswith(("0730", "0830"))),
                    clips[0])
        m = mask_for_clip(clip, args.samples, args.std_thresh,
                          args.bright_thresh)
        if m is None:
            print(f"  {cam:<8} could not sample")
            continue
        frac = float((m > 0).mean())
        p = OUT / f"{cam}.png"
        cv2.imwrite(str(p), m)
        ys, xs = np.where(m > 0)
        bbox = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                if len(xs) else None)
        summary[cam] = {"clip": clip.name, "masked_frac": round(frac, 5),
                        "bbox": bbox}
        print(f"  {cam:<8} {frac*100:>5.2f}% of frame masked   bbox {bbox}",
              flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nmasks -> {OUT}")
    print("Apply before detection/OCR: any box overlapping the mask is the")
    print("caption, not a plate.")


if __name__ == "__main__":
    main()
