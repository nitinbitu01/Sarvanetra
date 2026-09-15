"""
tests/behavior_eval/eval_runner.py — Day 8 §5 eval harness.

Replays synthetic labeled "clips" against the real loitering/crowd detector
code (not a reimplementation of the detection math) and diffs fired alerts
against ground truth, reporting precision/recall/false-positive-rate per
detector type.

Why synthetic clips, not real video files: both detectors consume exactly
(camera_id, track_id, cx, cy, video_time) or (camera_id, video_time, count)
tuples — nothing about their logic depends on pixels, video decoding, or
tracker internals (see the detector module docstrings: the ONLY inputs are
these tuples, produced upstream by connection_manager.py's Day 8 hook from
real detection events). A synthetic position/count sequence exercises
IDENTICAL code paths to a real clip would, without needing real CCTV
footage this environment doesn't have. Swapping in real annotated clips
later means only replacing the clips/*.json files — eval_runner.py and the
detectors themselves would not change.

Each clip in clips/*.json is either:
  {"detector": "loitering", "camera_id": ..., "segments": [{"track_id":...,
   "positions": [{"video_time","cx","cy"}, ...]}, ...], "expected_alerts": [...]}
  {"detector": "crowd", "camera_id": ..., "ticks": [{"video_time","count"}, ...],
   "expected_alerts": [...]}

expected_alerts is a list of {"video_time_window": [start, end]}. An empty
list means "this clip must produce zero alerts" (a negative case).

Uses a real Redis (settings.REDIS_URL) and a scratch SQLite DB, isolated from
whatever the app's actual output/sentinel.db is running.

HOW TO RUN:
    python -m tests.behavior_eval.eval_runner
    python -m tests.behavior_eval.eval_runner --redis-url redis://localhost:6379/1
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CLIPS_DIR = Path(__file__).resolve().parent / "clips"


def _load_clips() -> list[dict]:
    clips = []
    for path in sorted(_CLIPS_DIR.glob("*.json")):
        with path.open() as f:
            clips.append(json.load(f))
    return clips


async def _run_loitering_clip(clip: dict, detector) -> list[float]:
    """Replay all segments' positions in video_time order. Returns the list
    of video_times at which an alert actually fired."""
    events = []  # (video_time, track_id, cx, cy)
    for seg in clip["segments"]:
        for p in seg["positions"]:
            events.append((p["video_time"], seg["track_id"], p["cx"], p["cy"]))
    events.sort(key=lambda e: e[0])

    fired_at: list[float] = []
    for video_time, track_id, cx, cy in events:
        result = await detector.on_track_position(
            camera_str_id=clip["camera_id"], camera_db_id=None,
            track_id=track_id, cx=cx, cy=cy, video_time=video_time,
        )
        if result.decision == "LOITERING_FIRED":
            fired_at.append(video_time)
    return fired_at


async def _run_crowd_clip(clip: dict, detector) -> list[float]:
    fired_at: list[float] = []
    for tick in sorted(clip["ticks"], key=lambda t: t["video_time"]):
        result = await detector.on_frame_tick(
            camera_str_id=clip["camera_id"], camera_db_id=None,
            video_time=tick["video_time"], current_count=tick["count"],
        )
        if result.decision == "CROWD_ANOMALY_FIRED":
            fired_at.append(tick["video_time"])
    return fired_at


def _score_clip(fired_at: list[float], expected_alerts: list[dict]) -> dict:
    """Match fired video_times against expected windows.

    Each expected window can be matched by at most one fire (true positive);
    any fire outside all windows, or an extra fire inside an already-matched
    window, counts as a false positive. Any expected window with no matching
    fire counts as a false negative.
    """
    windows = [tuple(e["video_time_window"]) for e in expected_alerts]
    matched = [False] * len(windows)
    tp = fp = 0

    for t in fired_at:
        hit_idx = None
        for i, (lo, hi) in enumerate(windows):
            if lo <= t <= hi and not matched[i]:
                hit_idx = i
                break
        if hit_idx is not None:
            matched[hit_idx] = True
            tp += 1
        else:
            fp += 1

    fn = matched.count(False)
    return {"tp": tp, "fp": fp, "fn": fn, "fired_at": fired_at}


async def run_all(redis_url: str | None) -> int:
    if redis_url:
        os.environ["REDIS_URL"] = redis_url

    # Isolated scratch SQLite DB — never touches the app's real output/sentinel.db.
    tmpdir = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/behavior_eval.db"

    from backend.db.session import SessionLocal, engine
    from backend.db.models import Base, CameraCalibration, Alert
    from backend.services.loitering_detector import get_loitering_detector
    from backend.services.crowd_detector import get_crowd_detector
    from backend.services.state import get_behavior_state

    Base.metadata.create_all(bind=engine)

    redis_ok = await get_behavior_state().ping()
    if not redis_ok:
        print("[FATAL] Redis is not reachable at the configured REDIS_URL. "
              "The eval harness needs a real Redis to exercise the actual "
              "detector state backend (not a fake) — start one with "
              "`redis-server` and retry.", file=sys.stderr)
        return 1
    print("Redis: OK")

    clips = _load_clips()

    # Clear this harness's own keyspace before replaying.
    #
    # Without this the harness is not idempotent and only passes against a
    # Redis that has never run it: the loitering debounce flags
    # (behavior:active:LOITERING:*) and the crowd cooldown keys are built to
    # survive across ticks, so they also survive across RUNS. A second run
    # finds every alert already debounced and reports recall 0.00 on both
    # detectors — indistinguishable from a real detector regression. That also
    # made verify_day8.py (which shells out to this harness) fail for anyone
    # who had run it before.
    #
    # The patterns are derived from the clips' own camera_ids rather than a
    # hardcoded prefix, so adding a clip with a new camera id cannot silently
    # leave stale keys behind. Never a FLUSHDB: pointing --redis-url at a
    # shared Redis must not be able to destroy anything but these clips' keys.
    cleared = 0
    for camera_id in sorted({c["camera_id"] for c in clips if c.get("camera_id")}):
        cleared += await get_behavior_state().delete_matching(f"*{camera_id}*")
    print(f"Cleared {cleared} leftover key(s) from a previous run.\n")
    if not clips:
        print(f"[FATAL] No clips found in {_CLIPS_DIR}", file=sys.stderr)
        return 1

    # Pre-seed the one calibration row a clip depends on.
    db = SessionLocal()
    try:
        for clip in clips:
            if clip.get("requires_calibration_seed"):
                db.add(CameraCalibration(
                    camera_id=clip["camera_id"], px_per_meter=40.0,
                    calibration_method="manual_two_point", notes="eval harness seed",
                ))
        db.commit()
    finally:
        db.close()
    from backend.services.camera_calibration import get_calibration_cache
    get_calibration_cache().reload()

    loiter_detector = get_loitering_detector()
    crowd_detector = get_crowd_detector()

    results = []
    for clip in clips:
        if clip["detector"] == "loitering":
            fired_at = await _run_loitering_clip(clip, loiter_detector)
        elif clip["detector"] == "crowd":
            fired_at = await _run_crowd_clip(clip, crowd_detector)
        else:
            print(f"[WARN] Unknown detector type in {clip['clip_id']}: {clip['detector']}")
            continue

        score = _score_clip(fired_at, clip.get("expected_alerts", []))
        results.append({"clip_id": clip["clip_id"], "detector": clip["detector"], **score})

    # ── Verify the calibration-tagging checkpoint directly against the DB ──
    calibration_tag_ok = True
    db = SessionLocal()
    try:
        rows = db.query(Alert).filter(Alert.alert_type == "LOITERING").all()
        for clip in clips:
            if clip["detector"] != "loitering" or not clip.get("expected_alerts"):
                continue
            for r in rows:
                meta = json.loads(r.meta_json) if r.meta_json else {}
                if meta.get("identity_key", "").startswith(clip["camera_id"] + ":"):
                    expected_method = "manual_two_point" if clip.get("requires_calibration_seed") else "uncalibrated"
                    if meta.get("calibration_method") != expected_method:
                        calibration_tag_ok = False
                        print(f"[FAIL] {clip['clip_id']}: expected calibration_method="
                              f"{expected_method!r}, got {meta.get('calibration_method')!r}")
    finally:
        db.close()

    # ── Print summary ────────────────────────────────────────────────────
    print(f"{'clip_id':<32} {'detector':<10} {'tp':>3} {'fp':>3} {'fn':>3}  fired_at")
    print("-" * 90)
    for r in results:
        print(f"{r['clip_id']:<32} {r['detector']:<10} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3}  "
              f"{[round(t, 1) for t in r['fired_at']]}")

    by_detector: dict[str, dict[str, int]] = {}
    for r in results:
        d = by_detector.setdefault(r["detector"], {"tp": 0, "fp": 0, "fn": 0})
        d["tp"] += r["tp"]; d["fp"] += r["fp"]; d["fn"] += r["fn"]

    print("\n" + "=" * 90)
    print("PER-DETECTOR SUMMARY")
    print("=" * 90)
    overall_pass = calibration_tag_ok
    for detector, agg in by_detector.items():
        tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0
        fpr_clips = sum(1 for r in results if r["detector"] == detector and r["fp"] > 0)
        total_clips = sum(1 for r in results if r["detector"] == detector)
        print(f"{detector:<12} precision={precision:.2f}  recall={recall:.2f}  "
              f"tp={tp} fp={fp} fn={fn}  (false positives on {fpr_clips}/{total_clips} clips)")
        if fp > 0 or fn > 0:
            overall_pass = False

    print(f"\ncalibration tagging check: {'PASS' if calibration_tag_ok else 'FAIL'}")
    print(f"\n{'ALL CHECKS PASSED' if overall_pass else 'SOME CHECKS FAILED'} "
          f"(baseline: zero false positives on negative clips, zero false negatives on positive clips)")

    return 0 if overall_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 8 behavior engine eval harness")
    parser.add_argument("--redis-url", default=None,
                         help="Override REDIS_URL (default: from .env/settings)")
    args = parser.parse_args()
    exit_code = asyncio.run(run_all(args.redis_url))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
