"""backend/scripts/find_real_loitering.py — find a loitering event that is
unambiguously real, and produce the evidence for it.

WHY SEARCH FIRST INSTEAD OF JUST RUNNING THE DETECTOR
  Running the pipeline and taking whatever it emits gives an alert, not a
  demonstration. What is needed is one case where a human looking at the frames
  reaches the same conclusion the system did - otherwise "the system detected
  loitering" rests entirely on trusting the threshold.

  So this scans a clip end to end, measures every track's dwell, and ranks by
  how INDEFENSIBLE the case would be to argue with:

    duration      minutes, not the 60s minimum
    tightness     movement well inside the radius, not scraping it
    continuity    tracked without long gaps, so the dwell is one episode
                  rather than several people sharing a recycled track id

  The top candidate is then rendered with its full path drawn, so the claim can
  be checked rather than believed.

WHAT WOULD MAKE A CANDIDATE FALSE
  Two failure modes are reported rather than hidden:
    - a track id reused by BoT-SORT after losing the original person, which
      looks like a long dwell but is two different people
    - a stationary false detection (a poster, a parked bike read as a person)
      which never moves at all and so scores perfectly on every dwell test
  Both are visible in the rendered path and the gap statistics.

USAGE
  python -m backend.scripts.find_real_loitering --clip data/clips/CAM_04/CAM_04_0830.mp4
"""
from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

OUT = Path("output/real_loitering")


def max_pairwise(pts) -> float:
    if len(pts) < 2:
        return 0.0
    a = np.asarray(pts, dtype=np.float32)
    d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="data/clips/CAM_04/CAM_04_0830.mp4")
    ap.add_argument("--radius-px", type=float, default=120.0)
    ap.add_argument("--min-dwell-sec", type=float, default=120.0,
                    help="Only consider dwells well past the 60s rule — a "
                         "two-minute stand is far harder to argue with.")
    ap.add_argument("--max-gap-sec", type=float, default=4.0,
                    help="A longer tracking gap probably means the id was "
                         "reused for a different person.")
    ap.add_argument("--max-crowd", type=float, default=4.0,
                    help="Average people in frame during the dwell for it to "
                         "count as isolated. A 98s dwell at an auto stand "
                         "surrounded by twenty others is a man waiting for a "
                         "fare; the same dwell alone is not.")
    ap.add_argument("--save-clip", action="store_true",
                    help="Render the best candidate's dwell as a video. Three "
                         "still frames prove position at three instants; a "
                         "video shows the traffic changing around someone who "
                         "does not move, which is what makes the case obvious "
                         "to a viewer rather than requiring trust in a number.")
    ap.add_argument("--clip-pad-sec", type=float, default=15.0,
                    help="Context before and after the dwell — arriving and "
                         "leaving is what distinguishes a person from a "
                         "static false detection.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    stride = max(1, round(fps / float(cfg["processing"]["fps"])))
    eff = fps / stride
    print(f"clip     : {args.clip}")
    print(f"           {total:.0f} frames @ {fps:.0f}fps = {total/fps/60:.1f} min")
    print(f"sampling : every {stride} frames ({eff:.1f} fps)\n", flush=True)

    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])

    tracks: dict[int, list] = defaultdict(list)   # tid -> [(t, cx, cy, bbox)]
    # People present at each sampled instant. A person standing at an auto
    # stand surrounded by twenty others is waiting; the same dwell alone in an
    # empty frame is not. Duration cannot tell those apart - company can.
    crowd_at: dict[float, int] = {}
    stream = model.track(source=args.clip, stream=True, persist=True,
                         tracker="botsort.yaml", conf=0.35,
                         imgsz=cfg["processing"]["input_width"],
                         classes=[0], vid_stride=stride, verbose=False)
    for fi, r in enumerate(stream):
        t = fi * stride / fps
        if r.boxes is None or r.boxes.id is None:
            crowd_at[round(t, 1)] = 0
            continue
        crowd_at[round(t, 1)] = len(r.boxes)
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            tracks[int(b.id[0])].append(
                (t, (x1 + x2) / 2, (y1 + y2) / 2, (x1, y1, x2, y2)))
        if fi % 300 == 0:
            print(f"  {fi*stride}/{total:.0f} frames", flush=True)

    print(f"\ntracks seen: {len(tracks)}")

    cands = []
    for tid, obs in tracks.items():
        if len(obs) < 10:
            continue
        times = [o[0] for o in obs]
        gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        max_gap = max(gaps) if gaps else 0.0
        dur = times[-1] - times[0]
        if dur < args.min_dwell_sec:
            continue
        pts = [(o[1], o[2]) for o in obs]
        spread = max_pairwise(pts)
        if spread > args.radius_px:
            continue
        # Total path distinguishes a real person shifting weight from a
        # stationary false detection that never moves a single pixel.
        path = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
        # Average company during the dwell. This is what separates the auto
        # stand case - a man standing among twenty others, plainly waiting -
        # from someone holding position alone in an empty frame.
        near = [crowd_at.get(round(o[0], 1), 0) for o in obs]
        avg_crowd = sum(near) / max(len(near), 1)
        cands.append({
            "tid": tid, "dur": dur, "spread": spread, "path": path,
            "max_gap": max_gap, "obs": obs, "avg_crowd": avg_crowd,
            "continuous": max_gap <= args.max_gap_sec,
            "moves": path > 20.0,
            "isolated": avg_crowd <= args.max_crowd,
        })

    if not cands:
        print(f"\nNo track stayed within {args.radius_px:.0f}px for "
              f"{args.min_dwell_sec:.0f}s+ in this clip.")
        return

    # Strongest case first: long, tight, continuous, and demonstrably a person
    # rather than a static false positive.
    # Isolation ranks first: a dwell in an empty frame is the case that is
    # actually worth an officer's time, and the one this search exists to find.
    cands.sort(key=lambda c: (c["isolated"], c["continuous"], c["moves"],
                              c["dur"]), reverse=True)

    print(f"\n{'tid':>7} {'dwell':>8} {'spread':>8} {'path':>8} "
          f"{'gap':>6} {'crowd':>7}  isolated continuous moves")
    print("-" * 78)
    for c in cands[:10]:
        print(f"{c['tid']:>7} {c['dur']:>7.0f}s {c['spread']:>7.0f}px "
              f"{c['path']:>7.0f}px {c['max_gap']:>5.1f}s "
              f"{c['avg_crowd']:>6.1f}"
              f"{str(c['isolated']):>10}{str(c['continuous']):>11}"
              f"{str(c['moves']):>7}")

    best = cands[0]
    print(f"\nBEST CANDIDATE: track {best['tid']}")
    print(f"  stayed {best['dur']:.0f}s ({best['dur']/60:.1f} min) within "
          f"{best['spread']:.0f}px (limit {args.radius_px:.0f}px)")
    print(f"  walked {best['path']:.0f}px total — "
          + ("moves like a person" if best["moves"]
             else "BARELY MOVES — could be a static false detection"))
    print(f"  largest tracking gap {best['max_gap']:.1f}s — "
          + ("continuous" if best["continuous"]
             else "GAP: id may have been reused for another person"))
    print(f"  {best['avg_crowd']:.1f} people in frame on average — "
          + ("ALONE, no obvious reason to be waiting"
             if best["isolated"]
             else "in company: likely a waiting area, not suspicious"))

    # Render the evidence: three frames across the dwell, path overlaid.
    OUT.mkdir(parents=True, exist_ok=True)
    obs = best["obs"]
    cap = cv2.VideoCapture(args.clip)
    picks = [obs[0], obs[len(obs) // 2], obs[-1]]
    cells = []
    for i, (t, cx, cy, bb) in enumerate(picks):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if not ok:
            continue
        for j in range(len(obs) - 1):
            cv2.line(frame, (int(obs[j][1]), int(obs[j][2])),
                     (int(obs[j + 1][1]), int(obs[j + 1][2])), (60, 200, 250), 2)
        cv2.circle(frame, (int(cx), int(cy)), int(args.radius_px), (48, 80, 245), 2)
        x1, y1, x2, y2 = (int(v) for v in bb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (48, 80, 245), 4)
        band = np.full((74, frame.shape[1], 3), 18, np.uint8)
        cv2.putText(band, f"t = {t:.0f}s   ({t/60:.1f} min into clip)",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 230, 205), 2)
        cv2.putText(band, f"track {best['tid']}  |  in place {best['dur']:.0f}s  "
                          f"|  total drift {best['spread']:.0f}px "
                          f"(limit {args.radius_px:.0f}px)  |  path "
                          f"{best['path']:.0f}px",
                    (14, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
        img = np.vstack([band, frame])
        p = OUT / f"{i+1}_t{t:.0f}s.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cells.append(cv2.resize(img, (900, int(img.shape[0] * 900 / img.shape[1]))))
        print(f"  saved {p.name}")
    cap.release()

    if len(cells) == 3:
        h = min(c.shape[0] for c in cells)
        cv2.imwrite(str(OUT / "_proof.jpg"),
                    np.hstack([c[:h] for c in cells]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\nproof sheet -> {(OUT / '_proof.jpg').resolve()}")
    print("\nThe orange line is the person's ENTIRE path over the dwell. If it "
          "stays inside the red circle across all three frames, they did not "
          "leave — check that by eye before quoting this anywhere.")

    if args.save_clip:
        _render_clip(args, best, fps, eff)


def _render_clip(args, best, fps: float, eff_fps: float) -> None:
    """Render the dwell as video, with a live timer on the subject.

    Position is interpolated between the sampled observations so the box
    tracks smoothly at full frame rate rather than jumping five frames at a
    time - a stuttering box reads as a rendering artifact and undermines the
    very thing the clip is meant to show.
    """
    obs = best["obs"]
    t0 = max(0.0, obs[0][0] - args.clip_pad_sec)
    t1 = obs[-1][0] + args.clip_pad_sec
    start_f, end_f = int(t0 * fps), int(t1 * fps)

    cap = cv2.VideoCapture(args.clip)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    raw = OUT / "loitering_clip_raw.mp4"
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (W, H + 92))

    times = [o[0] for o in obs]
    print(f"\nrendering {t1-t0:.0f}s of video "
          f"(dwell {obs[0][0]:.0f}s -> {obs[-1][0]:.0f}s + "
          f"{args.clip_pad_sec:.0f}s padding either side)...", flush=True)

    n = 0
    for fi in range(start_f, end_f + 1):
        ok, frame = cap.read()
        if not ok:
            break
        t = fi / fps

        # Nearest sampled observation, and whether we are inside the dwell.
        idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
        near = obs[idx]
        active = times[0] <= t <= times[-1] and abs(near[0] - t) < 1.0

        if active:
            x1, y1, x2, y2 = (int(v) for v in near[3])
            held = t - times[0]
            fired = held >= 60.0
            col = (48, 80, 245) if fired else (60, 190, 250)
            # Path so far only - drawing the whole path from the first frame
            # would show the future and make the timer meaningless.
            pts = [(int(o[1]), int(o[2])) for o in obs if o[0] <= t]
            for j in range(len(pts) - 1):
                cv2.line(frame, pts[j], pts[j + 1], (60, 200, 250), 2)
            cv2.circle(frame, (int(near[1]), int(near[2])), 120, col, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 4)
            lab = "LOITERING" if fired else f"{held:.0f}s"
            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - th - 14), (x1 + tw + 12, y1 - 3),
                          col, -1)
            cv2.putText(frame, lab, (x1 + 5, y1 - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        band = np.full((92, W, 3), 18, np.uint8)
        cv2.putText(band, "SENTINEL — LOITERING", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (120, 235, 205), 2)
        if active:
            held = t - times[0]
            cv2.putText(band, f"in place {held:>5.0f}s / 60s      drift "
                              f"{best['spread']:.0f}px (limit 120px)      "
                              f"track {best['tid']}",
                        (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (48, 80, 245) if held >= 60 else (60, 190, 250), 2)
        else:
            cv2.putText(band, "watching — subject not yet stationary",
                        (16, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (150, 150, 150), 1)
        cv2.putText(band, f"t={t:.0f}s", (W - 150, 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 190, 190), 1)
        writer.write(np.vstack([band, frame]))
        n += 1

    writer.release()
    cap.release()
    print(f"  {n} frames written")

    # mp4v does not play in browsers or VS Code; re-encode to H.264 so the
    # clip can actually be watched by whoever it is shown to.
    out = OUT / "loitering_proof.mp4"
    import shutil
    import subprocess
    ff = shutil.which("ffmpeg")
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(raw),
                        "-vf", "scale=1280:-2", "-c:v", "libx264",
                        "-preset", "medium", "-crf", "23",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        str(out)], check=False)
        if out.is_file():
            raw.unlink(missing_ok=True)
            print(f"\nCLIP -> {out.resolve()}  "
                  f"({out.stat().st_size/1_048_576:.1f} MB, H.264)")
            return
    print(f"\nCLIP -> {raw.resolve()}  (mp4v — may not play in a browser)")


if __name__ == "__main__":
    main()
