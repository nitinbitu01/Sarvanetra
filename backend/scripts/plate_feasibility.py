"""backend/scripts/plate_feasibility.py — CAN we read plates at all?

THE QUESTION THIS SETTLES
  Camera geometry says a plate is ~30px wide in the full frame and ~20px
  after the imgsz=1280 downscale, giving ~4px characters against the ~20-25px
  OCR needs. On that basis full-frame ANPR is hopeless - and indeed the
  system has produced 72 anpr_uncertain alerts and zero successful reads.

  The cascade's claim is that the pixels DO exist, and the full-frame
  downscale is what destroys them:

      crop the vehicle from the ORIGINAL 1920 frame, then upscale
      -> a 107px-wide car becomes 640px = 6x
      -> its ~30px plate becomes ~180px, characters ~4px -> ~25px

  That is a testable claim, and it should be tested BEFORE anyone spends
  hours hand-labelling plates or building a cascade around it.

WHAT THIS DOES
  Pulls frames from daylight clips, detects vehicles with the deployed
  model, crops each from the SOURCE resolution, upscales, and runs OCR.
  Reports what came back and how big the text was - so the answer is
  evidence, not arithmetic.

WHAT IT DOES NOT DO
  It is not an accuracy measurement. Without hand-labelled ground truth
  there is no way to know whether a returned string is correct. It answers
  only "is there readable text there at all", which is the question that
  gates everything else.

USAGE
  python -m backend.scripts.plate_feasibility
  python -m backend.scripts.plate_feasibility --clips CAM_04,CAM_08 --scale 8
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

OUT = Path("output/plate_feasibility")
# Indian plate: 2 letters, 1-2 digits, 1-3 letters, 4 digits (plus BH series).
PLATE_RE = re.compile(r"^(?:[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$")
VEHICLE_CLASSES = {1, 2, 3, 4}      # car, motorcycle, bus, truck


def plateish(text: str) -> tuple[bool, str]:
    """Is this string plausibly a plate? Cleaned, then regex-checked."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(cleaned) < 6:
        return False, cleaned
    return bool(PLATE_RE.match(cleaned)), cleaned


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", default="CAM_04,CAM_08,CAM_20,CAM_19",
                    help="Cameras to sample daylight clips from.")
    ap.add_argument("--time", default="0830", help="Clip time target (daylight).")
    ap.add_argument("--frames-per-clip", type=int, default=25)
    ap.add_argument("--scale", type=int, default=640,
                    help="Upscale each vehicle crop to this width before OCR. "
                         "The whole point: a 107px car at 640px is 6x, which "
                         "turns a 30px plate into ~180px.")
    ap.add_argument("--min-car-width", type=int, default=80,
                    help="Ignore vehicles smaller than this - below roughly "
                         "80px the plate stays under ~25px even after "
                         "upscaling, and no OCR will read it.")
    ap.add_argument("--max-crops", type=int, default=120)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])
    print(f"detector: {cfg['model']['path']}")

    try:
        import easyocr
    except ImportError:
        print("easyocr not installed", file=sys.stderr)
        sys.exit(1)
    print("loading EasyOCR (~15s)...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    cams = [c.strip() for c in args.clips.split(",") if c.strip()]
    crops = 0
    any_text = 0
    plate_like = 0
    rows = []

    for cam in cams:
        clip = Path("data/clips") / cam / f"{cam}_{args.time}.mp4"
        if not clip.is_file():
            print(f"  {cam}: no clip at {clip}")
            continue
        cap = cv2.VideoCapture(str(clip))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        step = max(1, int(total / max(args.frames_per_clip, 1)))
        print(f"\n=== {cam} ({total:.0f} frames, sampling every {step}) ===",
              flush=True)

        taken = 0
        idx = 0
        while taken < args.frames_per_clip and crops < args.max_crops:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            taken += 1

            # Detect on the frame as the pipeline would...
            r = model(frame, imgsz=cfg["processing"]["input_width"],
                      conf=0.45, verbose=False, quantize=16)[0]
            if r.boxes is None:
                continue
            H, W = frame.shape[:2]

            for b in r.boxes:
                if crops >= args.max_crops:
                    break
                if int(b.cls[0]) not in VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                cw = x2 - x1
                if cw < args.min_car_width:
                    continue

                # ...but crop from the ORIGINAL frame, not the downscaled one.
                # This is the entire mechanism: the 1280 downscale destroys
                # plate detail before the detector ever sees it.
                pad = int(cw * 0.05)
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                f = args.scale / max(crop.shape[1], 1)
                up = cv2.resize(crop, None, fx=f, fy=f,
                                interpolation=cv2.INTER_CUBIC)
                crops += 1

                found = reader.readtext(up)
                texts = [(t, float(cf)) for _, t, cf in found if cf > 0.3]
                if texts:
                    any_text += 1
                best = ""
                for t, cf in texts:
                    ok_plate, cleaned = plateish(t)
                    if ok_plate:
                        plate_like += 1
                        best = cleaned
                        p = OUT / f"{cam}_{crops:03d}_{cleaned}.jpg"
                        cv2.imwrite(str(p), up)
                        print(f"  PLATE-LIKE  car {cw}px -> {up.shape[1]}px  "
                              f"'{cleaned}' (conf {cf:.2f})", flush=True)
                        break
                rows.append((cam, cw, len(texts), best))

        cap.release()

    print("\n" + "=" * 60)
    print(f"vehicle crops examined : {crops}")
    print(f"crops with ANY text    : {any_text} "
          f"({any_text/max(crops,1)*100:.0f}%)")
    print(f"crops with PLATE-FORMAT text : {plate_like} "
          f"({plate_like/max(crops,1)*100:.0f}%)")
    if rows:
        widths = np.array([r[1] for r in rows])
        print(f"vehicle width (source px): median {np.median(widths):.0f}, "
              f"p90 {np.percentile(widths,90):.0f}")
    print()
    if plate_like == 0 and any_text == 0:
        print("VERDICT: no text recovered at all. The cascade will not work at")
        print("this camera geometry - do not build it. Route plates to human")
        print("review and say so plainly.")
    elif plate_like == 0:
        print("VERDICT: OCR sees text but nothing plate-shaped. Likely reading")
        print("signage/bodywork. A DEDICATED plate detector (stage 3) is")
        print("needed before OCR - full-vehicle OCR is not enough.")
    else:
        print("VERDICT: plate-format strings ARE recoverable from upscaled")
        print("crops. The cascade is worth building. Next: a plate detector")
        print("to localise before OCR, then hand-label ~200 for real CER.")
    print(f"\ncrops saved to {OUT}/ - open them to check by eye.")


if __name__ == "__main__":
    main()
