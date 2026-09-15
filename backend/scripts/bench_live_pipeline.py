"""Run the real pipeline over all cameras and report whether it keeps up.

This replaces the "27+ cameras concurrently with zero lag" claim with a
measurement. Zero lag was never checked, and two mechanisms would have hidden
its absence: readers dropped frames on a full queue without counting them, and
the inference loop kept only the newest frame per camera and threw the backlog
away. Both are now counted, so a shortfall shows up instead of disappearing.

What the numbers mean:

  realtime_factor   video seconds consumed per wall-clock second, per camera.
                    1.0 is real time. This is the honest form of "no lag".
  frames_dropped    handed to a full queue and lost. Any non-zero value is
                    work the system did not do.
  queue_depth       standing backlog. Flat is healthy, growing is not.

Run:  python -m backend.scripts.bench_live_pipeline [seconds]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.live_24x7_pipeline import (                # noqa: E402
    READER_FPS, YOLO_IMGSZ, Live24x7Pipeline,
)


def main() -> int:
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0

    pipe = Live24x7Pipeline()
    pipe.start()
    n_cams = len(pipe._reader_threads)
    if not n_cams:
        print("no cameras started — is data/clips populated?")
        pipe.stop()
        return 1

    target = n_cams * READER_FPS
    print(f"cameras            {n_cams}")
    print(f"device             {pipe.device}")
    print(f"detector           yolov8s @ imgsz={YOLO_IMGSZ}")
    print(f"sampling           {READER_FPS} fps per camera")
    print(f"aggregate target   {target} frames/s "
          f"({1000/target:.1f} ms per frame budget)")
    print(f"\nrunning {duration:.0f}s — first samples include model warm-up\n")

    print(f"{'t':>5}{'proc fps':>10}{'queue':>7}{'dropped':>9}"
          f"{'rtf min':>9}{'rtf mean':>10}{'keeping up':>12}")
    print("-" * 62)

    t0 = time.time()
    last_frames, last_t = 0, t0
    samples = []
    while time.time() - t0 < duration:
        time.sleep(10.0)
        h = pipe.health
        now = time.time()
        frames = h["total_frames_processed"]
        fps = (frames - last_frames) / max(1e-6, now - last_t)
        last_frames, last_t = frames, now
        row = {
            "t": round(now - t0),
            "fps": fps,
            "queue": h["queue_depth"],
            "dropped": h.get("frames_dropped_total", 0),
            "rtf_min": h.get("realtime_factor_min"),
            "rtf_mean": h.get("realtime_factor_mean"),
            "keeping_up": h.get("cameras_keeping_up"),
        }
        samples.append(row)
        print(f"{row['t']:>5}{fps:>10.1f}{row['queue']:>7}"
              f"{row['dropped']:>9}{row['rtf_min']:>9.2f}"
              f"{row['rtf_mean']:>10.2f}"
              f"{str(row['keeping_up']) + '/' + str(n_cams):>12}")

    h = pipe.health
    pipe.stop()

    # The first sample carries model load and CUDA warm-up; steady state is
    # what a 24/7 deployment actually runs at.
    steady = samples[1:] or samples
    mean_fps = sum(s["fps"] for s in steady) / len(steady)

    print("\n" + "=" * 62)
    print("steady state")
    print("=" * 62)
    print(f"  aggregate throughput   {mean_fps:.1f} frames/s "
          f"(target {target})")
    print(f"  per camera             {mean_fps/n_cams:.2f} fps "
          f"(sampling {READER_FPS})")
    print(f"  frames processed       {h['total_frames_processed']:,}")
    print(f"  frames dropped         {h.get('frames_dropped_total', 0):,} "
          f"({h.get('frames_dropped_pct', 0)}%)")
    print(f"  cameras keeping up     {h.get('cameras_keeping_up')}/{n_cams}")
    print(f"  realtime factor        min {h.get('realtime_factor_min')}, "
          f"mean {h.get('realtime_factor_mean')}")
    print(f"  tracks persisted       {h['total_tracks_persisted']:,}")
    print(f"  vault crops harvested  {h['total_vault_harvested']:,}")

    phases = h.get("ms_per_frame", {})
    if phases:
        print("\n  where the time goes, ms per frame:")
        for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<8} {v:>8.2f}")
        print(f"    {'TOTAL':<8} {sum(phases.values()):>8.2f}")

    lag = h.get("per_camera_lag", {})
    behind = {c: v for c, v in lag.items() if v["rtf"] < 0.95}
    if behind:
        print(f"\n  {len(behind)} cameras below real time:")
        for c, v in sorted(behind.items(), key=lambda kv: kv[1]["rtf"])[:10]:
            print(f"    {c:<9} rtf {v['rtf']:.2f}  queued {v['queued']:>5}  "
                  f"dropped {v['dropped']:>5}")
    else:
        print("\n  every camera at or above real time, no frames dropped"
              if not h.get("frames_dropped_total")
              else "\n  every camera at or above real time")

    out = ROOT / "output" / "pipeline_bench.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "cameras": n_cams, "device": pipe.device, "imgsz": YOLO_IMGSZ,
        "reader_fps": READER_FPS, "aggregate_target_fps": target,
        "samples": samples, "final": h,
    }, indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
