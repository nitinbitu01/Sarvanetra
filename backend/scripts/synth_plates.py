"""backend/scripts/synth_plates.py — generate synthetic Indian number plates.

WHY SYNTHETIC IS THE UNLOCK
  The bootstrap loop can only label plates the weak OCR pipeline could already
  read: 395 boxes, 70% from one camera, six non-Gujarat examples in the entire
  corpus. Training a recogniser on that teaches it CAM_09's geometry and the
  Gujarat character distribution, which is precisely the model that fails on a
  Maharashtra truck at a junction it has never seen.

  Synthetic data breaks the dependency. Every state, every series pattern,
  every character, at any distance and degradation, with perfect ground truth
  - which is exactly what a recogniser needs and what no amount of harvesting
  from 27 clips will provide.

THE PART THAT DECIDES WHETHER THIS WORKS
  Not the rendering - the DEGRADATION. A model trained on crisp synthetic
  plates learns crisp glyphs and collapses on real 70px CCTV crops. The
  degradation chain here is ordered to match how the real image was actually
  formed:

      render clean  ->  perspective (camera looks down at traffic)
                    ->  downscale to real plate widths (45-150px, measured)
                    ->  motion blur (vehicles are moving; the close ones,
                                     the ones with enough pixels, move fastest)
                    ->  defocus, lighting, sensor noise
                    ->  JPEG at the quality the recorder actually uses
                    ->  upscale back, as the inference pipeline does

  Applying these in the wrong order produces images that look degraded but are
  not degraded the way real frames are, and the model learns the wrong
  invariances.

PLATE SPECIFICATION (as issued in India)
  private       white ground, black text
  commercial    yellow ground, black text
  electric      green ground, white text
  rental        black ground, yellow text
  format        LL DD L{1,3} DDDD, plus the BH series DD BH DDDD LL
  Two-row layouts are included: they are common on motorcycles and many cars,
  and their characters are smaller, so a model that never sees them fails on
  a large share of real traffic.

USAGE
  python -m backend.scripts.synth_plates --n 40 --out output/synth_preview
  python -m backend.scripts.synth_plates --n 100000 --out data/synth_plates
"""
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

import cv2
import numpy as np

STATE_CODES = [
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP",
    "WB",
]
# Gujarat dominates local traffic but must NOT be the only thing the model
# sees well - an out-of-state plate missed during an incident is the failure
# mode that matters. Weight toward realism while keeping every state frequent
# enough to learn.
STATE_WEIGHTS = {"GJ": 30, "MH": 8, "RJ": 7, "MP": 5, "DL": 4, "UP": 4,
                 "HR": 3, "PB": 3, "KA": 3, "TN": 2, "TS": 2, "AP": 2,
                 "WB": 2, "BR": 2, "OD": 2, "CG": 2, "JH": 2, "UK": 2,
                 "KL": 2, "HP": 2}

# (background, text) BGR. Proportions reflect what is actually on the road.
PLATE_STYLES = [
    ((242, 242, 242), (18, 18, 18), 0.70),      # private
    ((40, 200, 235), (18, 18, 18), 0.20),       # commercial (yellow)
    ((60, 140, 60), (245, 245, 245), 0.06),     # electric (green)
    ((25, 25, 25), (40, 200, 235), 0.04),       # rental (black)
]


def random_plate() -> str:
    if random.random() < 0.02:                                  # BH series
        return (f"{random.randint(20, 25):02d}BH{random.randint(1000, 9999)}"
                f"{''.join(random.choices(string.ascii_uppercase, k=2))}")
    states = list(STATE_WEIGHTS) if random.random() < 0.85 else STATE_CODES
    w = [STATE_WEIGHTS.get(s, 1) for s in states]
    st = random.choices(states, weights=w, k=1)[0]
    dist = random.randint(1, 38)
    series = "".join(random.choices(string.ascii_uppercase,
                                    k=random.choices([1, 2, 3],
                                                     weights=[25, 60, 15])[0]))
    return f"{st}{dist:02d}{series}{random.randint(0, 9999):04d}"


def render_plate(text: str, two_row: bool) -> np.ndarray:
    bg, fg, _ = random.choices(PLATE_STYLES,
                               weights=[s[2] for s in PLATE_STYLES], k=1)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    if two_row:
        # Split at the end of the district digits: state+district above,
        # series+number below. This is how two-row plates are actually laid out.
        i = 4
        rows = [text[:i], text[i:]]
        scale, thick = 2.0, 4
        sizes = [cv2.getTextSize(r, font, scale, thick)[0] for r in rows]
        tw = max(s[0] for s in sizes)
        th = sum(s[1] for s in sizes)
        W, H = tw + 60, th + 70
        img = np.full((H, W, 3), bg, np.uint8)
        y = 20 + sizes[0][1]
        for r, s in zip(rows, sizes):
            cv2.putText(img, r, ((W - s[0]) // 2, y), font, scale, fg,
                        thick, cv2.LINE_AA)
            y += s[1] + 26
    else:
        scale, thick = 2.2, 4
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        W, H = tw + 56, th + 44
        img = np.full((H, W, 3), bg, np.uint8)
        cv2.putText(img, text, (28, H - 22), font, scale, fg, thick, cv2.LINE_AA)
    cv2.rectangle(img, (4, 4), (W - 5, H - 5), fg, 3)          # embossed border
    return img


def degrade(img: np.ndarray, target_w: int) -> np.ndarray:
    """Apply the real image-formation chain, in order."""
    h, w = img.shape[:2]

    # 1. Perspective - these cameras look DOWN at traffic, so plates are
    #    foreshortened vertically and skewed horizontally.
    mag = random.uniform(0.02, 0.16)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * random.uniform(0, mag), h * random.uniform(0, mag)],
        [w * (1 - random.uniform(0, mag)), h * random.uniform(0, mag)],
        [w * (1 - random.uniform(0, mag)), h * (1 - random.uniform(0, mag))],
        [w * random.uniform(0, mag), h * (1 - random.uniform(0, mag))]])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                              borderMode=cv2.BORDER_REPLICATE)

    # 2. Downscale to a REAL plate width. This is the step that destroys
    #    detail, and it must happen before blur and noise - not after - or the
    #    degradation sits at the wrong spatial frequency.
    s = target_w / w
    img = cv2.resize(img, (max(8, int(w * s)), max(4, int(h * s))),
                     interpolation=cv2.INTER_AREA)

    # 3. Motion blur. Directional, because vehicles move along an axis.
    if random.random() < 0.6:
        k = random.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        if random.random() < 0.75:
            kern[k // 2, :] = 1.0                    # horizontal travel
        else:
            np.fill_diagonal(kern, 1.0)
        img = cv2.filter2D(img, -1, kern / kern.sum())

    # 4. Defocus.
    if random.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.4, 1.3))

    # 5. Lighting: glare, shadow, low contrast.
    a = random.uniform(0.55, 1.45)
    b = random.uniform(-45, 45)
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)

    # 6. Sensor noise.
    if random.random() < 0.7:
        img = np.clip(img.astype(np.int16) +
                      np.random.normal(0, random.uniform(2, 11), img.shape),
                      0, 255).astype(np.uint8)

    # 7. H.264/JPEG compression - small high-frequency glyph detail is the
    #    first thing an encoder discards, which is why real plates lose their
    #    thin strokes.
    q = random.randint(24, 68)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def legible(img: np.ndarray, min_detail: float = 300.0) -> bool:
    """Does any glyph structure survive the degradation?

    WHY THIS GATE EXISTS
      Roughly a third of an unfiltered batch comes out completely unreadable
      while still carrying an exact ground-truth label. Training a recogniser
      on "no information -> confident answer" teaches it to GUESS, and a
      confidently wrong plate on a suspect vehicle is worse than no read at
      all: nobody checks it, and the vehicle is lost anyway.

      So keep hard samples - they are what makes the model robust - but drop
      the ones where the information is genuinely gone. Variance of Laplacian
      measures surviving edge detail; below the threshold there are no glyph
      strokes left to learn from.

    MEASURE THE INTERIOR, NOT THE WHOLE PLATE
      The plate's border rectangle keeps strong edges even when every glyph
      has been smeared away, so whole-image Laplacian variance cannot tell a
      readable plate from an empty one. Measured on a sample: 'MP21O3147'
      scored 1006 over the full image while barely legible, and 'UP28CI8579'
      scored 24 - the two are indistinguishable by that metric. Restricted to
      the text area the same pair reads 415 vs 60, which separates cleanly.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    inner = g[int(h * 0.22):int(h * 0.78), int(w * 0.10):int(w * 0.90)]
    if inner.size == 0:
        return False
    return float(cv2.Laplacian(inner, cv2.CV_64F).var()) >= min_detail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="output/synth_preview")
    ap.add_argument("--min-w", type=int, default=45,
                    help="Narrowest rendered plate width, in px. Matches the "
                         "measured corpus (median 80px, p10 69).")
    ap.add_argument("--max-w", type=int, default=160)
    ap.add_argument("--two-row-frac", type=float, default=0.22)
    ap.add_argument("--out-w", type=int, default=192,
                    help="Final canvas width. Degradation is applied at the "
                         "REAL size, then upscaled - exactly as the inference "
                         "pipeline does.")
    ap.add_argument("--min-detail", type=float, default=300.0,
                    help="Interior Laplacian-variance floor, measured at the "
                         "final upscaled size. Calibrated by eye on a sample: "
                         "unreadable plates scored 60-230, readable ones 300+. "
                         "Samples below it are regenerated larger rather than "
                         "taught as confident answers.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    labels = out / "labels.jsonl"

    widths = []
    rejected = 0
    with labels.open("w", encoding="utf-8") as lf:
        for i in range(args.n):
            text = random_plate()
            two = random.random() < args.two_row_frac
            clean = render_plate(text, two)
            # Retry with progressively larger render widths rather than
            # silently emitting an unreadable image with a definite label.
            # Legibility MUST be judged at the size the model will see, after
            # the upscale - Laplacian variance is scale-dependent, so a
            # threshold calibrated on 192px canvases does not transfer to a
            # 72px degraded image. Measuring pre-upscale silently passed every
            # sample, including ones with no glyphs left at all.
            for attempt in range(4):
                tw = random.randint(args.min_w, args.max_w)
                if attempt:
                    tw = min(args.max_w, int(tw * (1.0 + 0.35 * attempt)))
                small = degrade(clean.copy(), tw)
                f = args.out_w / max(small.shape[1], 1)
                img = cv2.resize(small, None, fx=f, fy=f,
                                 interpolation=cv2.INTER_CUBIC)
                if legible(img, args.min_detail):
                    break
            else:
                rejected += 1
                continue
            widths.append(tw)
            name = f"{i:07d}.jpg"
            cv2.imwrite(str(out / "images" / name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            lf.write(json.dumps({"file": name, "text": text,
                                 "two_row": two, "render_w": tw}) + "\n")
            if args.n > 5000 and (i + 1) % 10000 == 0:
                print(f"  {i+1}/{args.n}", flush=True)

    w = np.array(widths)
    print(f"generated {args.n - rejected} plates -> {out.resolve()}")
    print(f"rejected as illegible : {rejected} "
          f"({rejected/max(args.n,1)*100:.1f}%) - no glyph detail survived, "
          f"so labelling them would teach guessing")
    print(f"render width: median {np.median(w):.0f}px  "
          f"range {w.min()}-{w.max()}")
    print(f"labels -> {labels}")
    print("\nLOOK AT THE SAMPLES. If they are noticeably cleaner or dirtier")
    print("than real crops, the model trains on the wrong distribution and")
    print("nothing downstream will fix it.")


if __name__ == "__main__":
    main()
