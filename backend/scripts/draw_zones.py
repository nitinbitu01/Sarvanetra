"""backend/scripts/draw_zones.py — draw loitering zones on a real camera frame.

WHY DRAWN, NOT TYPED
  Zone coordinates written by hand are guesses. What matters is where the bus
  stop actually is in THIS camera's frame, which is only knowable by looking
  at the frame. So this pulls a real frame from the camera's own footage and
  lets a person draw on it.

WHAT THE THREE TYPES MEAN
  exempt      Bus stop, auto stand, hospital gate, shop frontage. People wait
              here as a matter of course. Never alerts, at any duration - this
              is what removes most false alarms.
  normal      Footpath, general road. Long threshold (10 min day / 5 min night).
  sensitive   Parked vehicles, ATM, an isolated corner. Short threshold
              (3 min day / 90s night).

  Draw exempt zones generously. A missed alert in a bus stop costs nothing; a
  stream of alerts from a bus stop costs the operator's attention, and once
  that is gone the real alerts go unread too.

CONTROLS
  left click        add a point
  right click       undo last point
  ENTER             close the polygon and choose its type
  1 / 2 / 3         set type for the polygon just closed
                    (1 exempt, 2 normal, 3 sensitive)
  u                 delete the last completed zone
  s                 save and quit
  q                 quit without saving

USAGE
  python -m backend.scripts.draw_zones --camera CAM_04 \
      --clip demo/clips/cam_04_0700.mp4 --time 30
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

OUT_DIR = Path("config/zones")

TYPES = {
    1: ("exempt", (90, 200, 90), "EXEMPT — never alert"),
    2: ("normal", (60, 190, 250), "NORMAL — 10min day / 5min night"),
    3: ("sensitive", (48, 80, 245), "SENSITIVE — 3min day / 90s night"),
}
DEFAULTS = {
    "exempt": (None, None),
    "normal": (600.0, 300.0),
    "sensitive": (180.0, 90.0),
}

state = {"points": [], "zones": [], "pending": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["points"].append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and state["points"]:
        state["points"].pop()


def render(base):
    img = base.copy()
    overlay = img.copy()

    for z in state["zones"]:
        col = next(c for _, (t, c, _) in TYPES.items() if t == z["type"])
        pts = np.array(z["polygon"], np.int32)
        cv2.fillPoly(overlay, [pts], col)
        cv2.polylines(img, [pts], True, col, 2)
        cx = int(np.mean([p[0] for p in z["polygon"]]))
        cy = int(np.mean([p[1] for p in z["polygon"]]))
        cv2.putText(img, f"{z['type']}: {z['label']}", (cx - 70, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)

    pts = state["points"]
    for i, p in enumerate(pts):
        cv2.circle(img, p, 5, (255, 255, 255), -1)
        if i:
            cv2.line(img, pts[i - 1], p, (255, 255, 255), 2)
    if len(pts) > 2:
        cv2.line(img, pts[-1], pts[0], (160, 160, 160), 1)

    h = img.shape[0]
    band = np.full((96, img.shape[1], 3), 18, np.uint8)
    if state["pending"] is not None:
        cv2.putText(band, "CHOOSE TYPE:  1=EXEMPT   2=NORMAL   3=SENSITIVE",
                    (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (90, 230, 205), 2)
    else:
        cv2.putText(band, "L-click add point | R-click undo | ENTER close "
                          "polygon | u delete last zone | s save | q quit",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215, 215, 215), 1)
    cv2.putText(band, f"points: {len(pts)}    zones: {len(state['zones'])}"
                      f"    (draw EXEMPT zones generously — bus stops, "
                      f"auto stands, shop fronts)",
                (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)
    for i, (t, c, desc) in TYPES.items():
        cv2.putText(band, f"{i}={desc}", (14 + (i - 1) * 330, 86),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 1)
    return np.vstack([img, band])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", required=True, help="e.g. CAM_04")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--time", type=float, default=30.0,
                    help="Seconds into the clip to grab the reference frame. "
                         "Pick a busy moment so the scene is representative.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"cannot read frame at {args.time}s")

    print(f"camera : {args.camera}")
    print(f"frame  : {frame.shape[1]}x{frame.shape[0]} at {args.time}s\n")
    print("Draw EXEMPT zones over bus stops, auto stands, shop frontages,")
    print("hospital gates — anywhere waiting is normal. Those are what stop")
    print("the false alerts.\n")

    win = f"Zones — {args.camera}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, frame.shape[1]),
                     min(950, frame.shape[0] + 96))
    cv2.setMouseCallback(win, on_mouse)

    while True:
        cv2.imshow(win, render(frame))
        k = cv2.waitKey(20) & 0xFF

        if k in (13, 10):                       # ENTER — close polygon
            if len(state["points"]) >= 3:
                state["pending"] = list(state["points"])
                state["points"] = []
            else:
                print("  need at least 3 points")
        elif state["pending"] is not None and k in (ord("1"), ord("2"), ord("3")):
            ztype, _, _ = TYPES[k - ord("0")]
            day, night = DEFAULTS[ztype]
            zid = f"{ztype}_{len(state['zones']) + 1}"
            state["zones"].append({
                "id": zid, "type": ztype, "label": zid,
                "polygon": state["pending"],
                "threshold_day_sec": day, "threshold_night_sec": night,
            })
            print(f"  + {ztype:<10} {len(state['pending'])} points  "
                  f"(day {day}, night {night})")
            state["pending"] = None
        elif k == ord("u"):
            if state["zones"]:
                z = state["zones"].pop()
                print(f"  - removed {z['type']} zone")
        elif k == ord("s"):
            break
        elif k == ord("q"):
            print("quit without saving")
            cv2.destroyAllWindows()
            return

    cv2.destroyAllWindows()

    if not state["zones"]:
        print("no zones drawn — nothing saved")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.camera}.json"
    out.write_text(json.dumps({
        "camera_id": args.camera,
        "frame_width": int(frame.shape[1]),
        "frame_height": int(frame.shape[0]),
        "source_clip": args.clip,
        "zones": state["zones"],
    }, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for z in state["zones"]:
        counts[z["type"]] = counts.get(z["type"], 0) + 1
    print(f"\nsaved {len(state['zones'])} zones -> {out.resolve()}")
    print(f"   {counts}")
    print("\nNOTE: polygons are in THIS frame's pixel coordinates "
          f"({frame.shape[1]}x{frame.shape[0]}). If the camera is re-aimed or "
          "the resolution changes, redraw them.")


if __name__ == "__main__":
    main()
