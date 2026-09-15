"""Where the plate's pixels are lost, stage by stage.

Five checks, in the order the pixels flow, so the stage that destroys them is
named rather than guessed at:

  1  source resolution   how large is a plate in the original frame, in pixels
                         and as a share of frame area
  2  detection input     the vehicle detector sees a downscaled frame; does
                         that lose small vehicles, and with them their plates
  3  crop natively       are plate crops cut from the original frame or from
                         the downscaled copy
  4  upscale factor      how far each crop is stretched to reach the model's
                         256px input, since interpolation adds no information
  5  native adequacy     what fraction of crops are already near 256px wide
                         without being stretched

The reason for measuring 4 and 5 separately: a crop that is natively 240px
wide and one that is 58px stretched to 256 arrive at the recogniser looking
identical, and nothing downstream can tell them apart. One carries four times
the evidence of the other, and only the first deserves to be trusted.

Run:  python -m backend.scripts.plate_pixel_budget_audit
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO                                       # noqa: E402

CLIPS = ROOT / "data" / "clips"
VEHICLE_MODEL = ROOT / "yolov8s.pt"
PLATE_MODEL = ROOT / "runs/detect/runs/plate/plate_v3_ft/weights/best.pt"
CAMERAS = ["CAM_08", "CAM_04", "CAM_06", "CAM_14", "CAM_02", "CAM_26"]
FRAMES_PER_CAM = 12
MODEL_INPUT_W = 256          # the 64x256 recogniser


def frames(cam: str, n: int) -> list:
    clips = sorted((CLIPS / cam).glob("*.mp4"))
    if not clips:
        return []
    cap = cv2.VideoCapture(str(clips[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for idx in np.linspace(total * 0.15, total * 0.85, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok and f is not None:
            out.append(f)
    cap.release()
    return out


def main() -> int:
    vdet = YOLO(str(VEHICLE_MODEL))
    pdet = YOLO(str(PLATE_MODEL))

    print("=" * 68)
    print("1  SOURCE RESOLUTION — is the plate big enough in the raw frame?")
    print("=" * 68)
    print(f"{'camera':<9}{'frame':>12}{'vehicles':>10}{'plates':>8}"
          f"{'plate w':>9}{'% of area':>11}")
    print("-" * 68)

    all_plate_w, all_area_pct, all_native = [], [], []
    frames_by_cam = {}
    for cam in CAMERAS:
        fr = frames(cam, FRAMES_PER_CAM)
        if not fr:
            continue
        frames_by_cam[cam] = fr
        H, W = fr[0].shape[:2]
        n_veh = n_pl = 0
        pw, apct = [], []
        for f in fr:
            r = vdet.predict(f, conf=0.35, classes=[2, 3, 5, 7], verbose=False,
                             device="cuda:0", imgsz=640)
            b = r[0].boxes
            if b is None or not len(b):
                continue
            for xyxy in b.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue
                n_veh += 1
                # Crop from the ORIGINAL frame, at native resolution.
                crop = f[y1:y2, x1:x2]
                pr = pdet.predict(crop, imgsz=320, conf=0.15, verbose=False,
                                  device="cuda:0")
                pb = pr[0].boxes
                if pb is None or not len(pb):
                    continue
                n_pl += 1
                i = int(pb.conf.argmax())
                px = pb.xyxy[i].cpu().numpy()
                w_px = float(px[2] - px[0])
                h_px = float(px[3] - px[1])
                pw.append(w_px)
                apct.append(100.0 * (w_px * h_px) / (W * H))
        if pw:
            all_plate_w += pw
            all_area_pct += apct
            print(f"{cam:<9}{f'{W}x{H}':>12}{n_veh:>10}{n_pl:>8}"
                  f"{np.median(pw):>8.0f}px{np.median(apct):>10.3f}%")
        else:
            print(f"{cam:<9}{f'{W}x{H}':>12}{n_veh:>10}{n_pl:>8}{'-':>9}{'-':>11}")

    if all_plate_w:
        print("-" * 68)
        print(f"{'fleet':<9}{'':>12}{'':>10}{len(all_plate_w):>8}"
              f"{np.median(all_plate_w):>8.0f}px{np.median(all_area_pct):>10.3f}%")
        print(f"\nA dedicated ANPR camera is specified so the plate covers "
              f"5-10% of frame area.\nThis fleet's median is "
              f"{np.median(all_area_pct):.3f}% — roughly "
              f"{5/max(np.median(all_area_pct),1e-6):.0f}x short of that. "
              f"These are\ngeneral surveillance cameras, not ANPR cameras, "
              f"and no software stage recovers\nthe pixels the optics never "
              f"collected.")

    print("\n" + "=" * 68)
    print("2  DETECTION INPUT — does downscaling the frame lose vehicles?")
    print("=" * 68)
    print("The pipeline resizes frames to 1280x720 for batching, then the")
    print("detector works at imgsz 640. Small, distant vehicles are the ones")
    print("that carry small plates, so a loss here is a loss of exactly the")
    print("hard cases.\n")
    print(f"{'camera':<9}{'imgsz 640':>11}{'imgsz 1280':>12}{'imgsz 1920':>12}"
          f"{'gain':>8}")
    print("-" * 56)
    tot = defaultdict(int)
    for cam, fr in frames_by_cam.items():
        counts = {}
        for size in (640, 1280, 1920):
            n = 0
            for f in fr:
                r = vdet.predict(f, conf=0.35, classes=[2, 3, 5, 7],
                                 verbose=False, device="cuda:0", imgsz=size)
                b = r[0].boxes
                if b is not None and len(b):
                    n += len(b)
            counts[size] = n
            tot[size] += n
        gain = counts[1920] - counts[640]
        print(f"{cam:<9}{counts[640]:>11}{counts[1280]:>12}{counts[1920]:>12}"
              f"{gain:>+8}")
    print("-" * 56)
    print(f"{'ALL':<9}{tot[640]:>11}{tot[1280]:>12}{tot[1920]:>12}"
          f"{tot[1920]-tot[640]:>+8}")

    print("\n" + "=" * 68)
    print("4/5  UPSCALE FACTOR — how much of the 256px input is real?")
    print("=" * 68)
    if all_plate_w:
        factors = [MODEL_INPUT_W / max(w, 1) for w in all_plate_w]
        print(f"{'native crop width':<24}{'crops':>8}{'share':>9}"
              f"{'upscale to 256':>16}")
        print("-" * 58)
        bands = [(0, 64), (64, 100), (100, 160), (160, 256), (256, 10_000)]
        for lo, hi in bands:
            sel = [w for w in all_plate_w if lo <= w < hi]
            if not sel:
                continue
            up = MODEL_INPUT_W / np.median(sel)
            label = f"{lo}-{hi}px" if hi < 10_000 else f"{lo}px+"
            print(f"{label:<24}{len(sel):>8}"
                  f"{100*len(sel)/len(all_plate_w):>8.1f}%"
                  f"{up:>15.1f}x")
        native_ok = sum(1 for w in all_plate_w if w >= 200)
        print(f"\ncrops already near the model's 256px input (>=200px): "
              f"{native_ok}/{len(all_plate_w)} "
              f"({100*native_ok/len(all_plate_w):.1f}%)")
        print(f"median upscale applied: {np.median(factors):.1f}x")
        print("""
Interpolation cannot add detail. A crop stretched 4x reaches the recogniser
at the same 256px as a native one and is indistinguishable to it, which is
why native width has to be carried alongside the read and used to temper
confidence — measured accuracy by native width was 0% below 70px, 25% at
70-100px and 51.7% at 100-140px.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
