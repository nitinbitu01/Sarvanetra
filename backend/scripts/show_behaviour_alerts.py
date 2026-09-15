"""backend/scripts/show_behaviour_alerts.py — see the frames behind behaviour alerts.

WHY THIS IS NEEDED
  The behaviour alerts carry `evidence_path=None`: evidence capture never ran
  for them, so there is no stored image of what triggered a detection. An
  alert nobody can look at is not usable by an officer and not demonstrable to
  anyone - "the system detected loitering" has to come with the frame.

  This reconstructs the moment from the clip: seek to the alert's video_time,
  run person detection on that frame, and draw what was there.

WHAT IT CAN AND CANNOT SHOW
  meta_json records identity_key, span_sec, max_dist_px and video_time - but
  NOT the bounding box. So every person present at that instant is drawn,
  rather than the specific one that loitered. That is an honest limitation of
  what was stored, not a rendering choice, and the fix is to persist the bbox
  when the alert fires (see --note-fix).

USAGE
  python -m backend.scripts.show_behaviour_alerts --clip demo/clips/cam_04_0700.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

OUT = Path("output/behaviour_evidence")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="demo/clips/cam_04_0700.mp4")
    ap.add_argument("--camera", default="CAM-01",
                    help="Only alerts whose camera_id matches; the synthetic "
                         "test alerts used 'CAM_01' (underscore) and are "
                         "excluded by default.")
    ap.add_argument("--types", default="LOITERING,CROWD_ANOMALY")
    ap.add_argument("--limit", type=int, default=12)
    args = ap.parse_args()

    types = tuple(t.strip() for t in args.types.split(",") if t.strip())
    con = sqlite3.connect("output/sentinel.db")
    q = (f"SELECT id, alert_type, subject_label, danger_score, meta_json, timestamp "
         f"FROM alerts WHERE alert_type IN ({','.join('?' * len(types))}) "
         f"AND camera_id = ? ORDER BY timestamp")
    rows = list(con.execute(q, (*types, args.camera)))
    con.close()
    if not rows:
        raise SystemExit(f"no {types} alerts for camera {args.camera}")
    print(f"alerts found : {len(rows)}")

    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f"clip         : {args.clip}")
    print(f"               {total:.0f} frames @ {fps:.1f}fps = "
          f"{total/fps:.0f}s of video\n")

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])

    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    for i, (aid, atype, label, danger, meta, ts) in enumerate(rows[:args.limit]):
        m = json.loads(meta or "{}")
        vt = float(m.get("video_time", 0.0))
        # video_time is CUMULATIVE across loops when camera.loop_video is on -
        # main.py computes it as (loop_count * duration) + frame/fps so that
        # timestamps stay strictly increasing. A value of 985s on a 90s clip is
        # loop 10, not a different clip, so map it back with modulo.
        clip_secs = total / fps
        loop_no = int(vt // clip_secs)
        pos = vt % clip_secs
        frame_no = min(int(pos * fps), int(total) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            print(f"  [{i+1}] could not read frame {frame_no}")
            continue

        r = model(frame, imgsz=cfg["processing"]["input_width"], conf=0.35,
                  verbose=False)[0]
        n_people = 0
        vis = frame.copy()
        # The alert records the subject's centroid, so the person who actually
        # triggered it can be singled out instead of boxing the whole crowd.
        sx, sy = m.get("cx"), m.get("cy")
        subject = None
        best_d = 1e9
        boxes = []
        if r.boxes is not None:
            for b in r.boxes:
                if int(b.cls[0]) != 0:            # persons only
                    continue
                n_people += 1
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                boxes.append((x1, y1, x2, y2))
                if sx is not None:
                    d = ((x1 + x2) / 2 - sx) ** 2 + ((y1 + y2) / 2 - sy) ** 2
                    if d < best_d:
                        best_d, subject = d, (x1, y1, x2, y2)
        for bx in boxes:
            if bx == subject:
                continue
            cv2.rectangle(vis, bx[:2], bx[2:], (110, 110, 110), 1)
        if subject is not None:
            x1, y1, x2, y2 = subject
            cv2.rectangle(vis, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (40, 90, 250), 4)
            cv2.circle(vis, (int(sx), int(sy)), int(m.get("radius_px", 120)),
                       (40, 90, 250), 2)
            lab = f"LOITERING {m.get('span_sec', 0):.0f}s"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(vis, (x1 - 3, y1 - th - 12), (x1 + tw + 8, y1 - 3),
                          (40, 90, 250), -1)
            cv2.putText(vis, lab, (x1 + 2, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
        elif sx is not None:
            cv2.circle(vis, (int(sx), int(sy)), 26, (40, 90, 250), 3)

        band = np.full((78, vis.shape[1], 3), 22, np.uint8)
        loop_tag = f"  [loop {loop_no}]" if loop_no else "  [first pass]"
        cv2.putText(band, f"{atype}  t={pos:.1f}s in clip{loop_tag}  frame {frame_no}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                    (90, 230, 200) if not loop_no else (120, 180, 240), 2)
        cv2.putText(band, f"{label}", (12, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1)
        cv2.putText(band, f"track {m.get('identity_key','?')}  "
                          f"drift {m.get('max_dist_px','?')}px  "
                          f"radius {m.get('radius_px','?')}px  "
                          f"persons in frame: {n_people}",
                    (12, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        out = np.vstack([band, vis])
        p = OUT / f"{i+1:02d}_{atype}_t{vt:.0f}s.jpg"
        cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        print(f"  [{i+1}] {atype} @ {vt:6.1f}s  {n_people} persons  -> {p.name}")
        cells.append(cv2.resize(out, (760, int(out.shape[0] * 760 / out.shape[1]))))

    cap.release()

    if cells:
        h = min(c.shape[0] for c in cells)
        grid = np.vstack([np.hstack([c[:h] for c in cells[i:i + 2]])
                          for i in range(0, len(cells) - len(cells) % 2, 2)])
        cv2.imwrite(str(OUT / "_all.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\ncontact sheet -> {(OUT / '_all.jpg').resolve()}")
    print(f"individual frames -> {OUT.resolve()}")
    print("\nGreen boxes are every person detected at that instant. The stored")
    print("meta_json has no bbox, so the specific loiterer cannot be singled")
    print("out — persisting the bbox at fire time would fix that.")


if __name__ == "__main__":
    main()
