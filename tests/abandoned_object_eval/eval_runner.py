"""
tests/abandoned_object_eval/eval_runner.py — Day 9 §5 eval harness.

Replays synthetic labeled clips against the real AbandonedObjectDetector
code (not a reimplementation) and reports precision/recall/false-positive-rate
specifically for ABANDONED_OBJECT alerts.

WHY SYNTHETIC CLIPS:
  AbandonedObjectDetector.on_object_update() consumes exactly:
    (camera_str_id, camera_db_id, object_track_id, object_class, cx, cy,
     video_time, is_baseline)
  on_person_positions() consumes:
    (camera_str_id, object_track_ids_with_nearby_person, video_time)
  Nothing in the detector logic depends on pixels or video decoding — a
  synthetic position sequence exercises identical code paths. Swapping in
  real annotated clips means only replacing the clips/*.json files.

CLIP FORMAT:
  {
    "clip_id": str,
    "description": str,
    "detector": "abandoned_object",
    "camera_id": str,
    "requires_calibration_seed": bool,   // if true, seeds camera_calibration row
    "segments": [                         // one per static object track
      {
        "object_track_id": str,
        "object_class": str,
        "is_baseline": bool,
        "ticks": [{"video_time": float, "cx": float, "cy": float}, ...]
      }
    ],
    "proximity_events": [               // per-frame proximity updates
      {"video_time": float, "object_track_ids_nearby": [str, ...]},
      ...
    ],
    "expected_alerts": [                // empty = must produce zero alerts
      {"video_time_window": [start, end]},
      ...
    ]
  }

PASS CRITERION (Day 9 §5):
  Zero false positives on negative clips is the bar. Every negative clip that
  fires an alert is a regression. Positive clips must also recall correctly.

HOW TO RUN:
  python -m tests.abandoned_object_eval.eval_runner
  python -m tests.abandoned_object_eval.eval_runner --redis-url redis://localhost:6379/2

RE-RUN WHEN:
  STATIC_MOVEMENT_PX_THRESHOLD, ABANDON_DURATION_SEC, or OWNERSHIP_RADIUS_METERS
  are changed in config.py / .env — these three carry the highest FP risk.
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


def _score_clip(fired_at: list[float], expected_alerts: list[dict]) -> dict:
    """Match fired video_times against expected windows.

    Each window can match at most one fire (true positive). A fire outside
    all windows, or an extra fire inside an already-matched window, is a
    false positive. An expected window with no matching fire is a false negative.
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


async def _run_clip(clip: dict, detector) -> list[float]:
    """Replay one clip against the detector. Returns video_times where alert fired."""
    # Merge all object ticks + proximity events into a single sorted timeline.
    # Each entry: (video_time, event_type, payload)
    timeline: list[tuple[float, str, dict]] = []

    for seg in clip.get("segments", []):
        for tick in seg["ticks"]:
            timeline.append((
                tick["video_time"],
                "object_tick",
                {
                    "object_track_id": seg["object_track_id"],
                    "object_class": seg["object_class"],
                    "is_baseline": seg.get("is_baseline", False),
                    "cx": tick["cx"],
                    "cy": tick["cy"],
                },
            ))

    for prox in clip.get("proximity_events", []):
        timeline.append((
            prox["video_time"],
            "proximity",
            {"object_track_ids_nearby": prox.get("object_track_ids_nearby", [])},
        ))

    # Sort: at the same video_time, process proximity events BEFORE object ticks
    # so the proximity flag is set before the tick's "PERSON_NEARBY" check runs.
    timeline.sort(key=lambda e: (e[0], 0 if e[1] == "proximity" else 1))

    fired_at: list[float] = []
    camera_id = clip["camera_id"]

    for video_time, event_type, payload in timeline:
        if event_type == "proximity":
            await detector.on_person_positions(
                camera_str_id=camera_id,
                object_track_ids_with_nearby_person=payload["object_track_ids_nearby"],
                video_time=video_time,
                proximity_ttl_sec=15,  # generous for eval; real value tuned to frame rate
            )
        elif event_type == "object_tick":
            result = await detector.on_object_update(
                camera_str_id=camera_id,
                camera_db_id=None,
                object_track_id=payload["object_track_id"],
                object_class=payload["object_class"],
                cx=payload["cx"],
                cy=payload["cy"],
                video_time=video_time,
                is_baseline=payload["is_baseline"],
            )
            if result.decision == "ABANDONED_FIRED":
                fired_at.append(video_time)

    return fired_at


async def run_all(redis_url: str | None) -> int:
    if redis_url:
        os.environ["REDIS_URL"] = redis_url

    # Isolated scratch SQLite DB — never touches the app's output/sentinel.db.
    tmpdir = tempfile.mkdtemp()
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/abandoned_eval.db"

    from backend.db.session import SessionLocal, engine
    from backend.db.models import Base, CameraCalibration, Alert
    from backend.services.abandoned_object_detector import get_abandoned_object_detector
    from backend.services.state import get_behavior_state
    from backend.services.camera_calibration import get_calibration_cache

    Base.metadata.create_all(bind=engine)

    redis_ok = await get_behavior_state().ping()
    if not redis_ok:
        print(
            "[FATAL] Redis is not reachable at the configured REDIS_URL.\n"
            "The eval harness needs a real Redis to exercise the actual detector\n"
            "state backend (not a fake). Start one with `redis-server` and retry.",
            file=sys.stderr,
        )
        return 1
    print("Redis: OK")

    clips = _load_clips()
    if not clips:
        print(f"[FATAL] No clips found in {_CLIPS_DIR}", file=sys.stderr)
        return 1

    # Same idempotency guard as tests/behavior_eval/eval_runner.py — the
    # object_track hashes and proximity markers persist across runs, so a
    # replay would resume from the previous run's "alert_fired" status and
    # report a false negative. Patterns derived from the clips' own
    # camera_ids; never a FLUSHDB.
    cleared = 0
    for camera_id in sorted({c["camera_id"] for c in clips if c.get("camera_id")}):
        cleared += await get_behavior_state().delete_matching(f"*{camera_id}*")
    print(f"Cleared {cleared} leftover key(s) from a previous run.\n")

    abandoned_clips = [c for c in clips if c.get("detector") == "abandoned_object"]
    if not abandoned_clips:
        print("[FATAL] No abandoned_object clips found.", file=sys.stderr)
        return 1

    # Pre-seed calibration rows for clips that require it.
    db = SessionLocal()
    try:
        for clip in abandoned_clips:
            if clip.get("requires_calibration_seed"):
                existing = db.query(CameraCalibration).filter(
                    CameraCalibration.camera_id == clip["camera_id"]
                ).first()
                if not existing:
                    db.add(CameraCalibration(
                        camera_id=clip["camera_id"],
                        px_per_meter=40.0,
                        calibration_method="manual_two_point",
                        notes="eval harness seed",
                    ))
        db.commit()
    finally:
        db.close()

    get_calibration_cache().reload()

    detector = get_abandoned_object_detector()

    results = []
    for clip in abandoned_clips:
        print(f"Running: {clip['clip_id']} — {clip.get('description', '')[:70]}")
        fired_at = await _run_clip(clip, detector)
        score = _score_clip(fired_at, clip.get("expected_alerts", []))
        results.append({
            "clip_id": clip["clip_id"],
            "detector": "abandoned_object",
            **score,
        })

    # ── Verify calibration tagging in fired alerts ────────────────────────────
    calibration_tag_ok = True
    db = SessionLocal()
    try:
        rows = db.query(Alert).filter(Alert.alert_type == "ABANDONED_OBJECT").all()
        for clip in abandoned_clips:
            if not clip.get("expected_alerts"):
                continue
            for r in rows:
                meta = json.loads(r.meta_json) if r.meta_json else {}
                oid = meta.get("object_track_id", "")
                if not oid.startswith(clip["camera_id"] + ":"):
                    continue
                expected_method = (
                    "manual_two_point" if clip.get("requires_calibration_seed")
                    else "uncalibrated"
                )
                actual_method = meta.get("calibration_method")
                if actual_method != expected_method:
                    calibration_tag_ok = False
                    print(
                        f"[FAIL] {clip['clip_id']}: expected calibration_method="
                        f"{expected_method!r}, got {actual_method!r}"
                    )
    finally:
        db.close()

    # ── Print summary table ───────────────────────────────────────────────────
    print()
    print(f"{'clip_id':<36} {'tp':>3} {'fp':>3} {'fn':>3}  fired_at")
    print("-" * 90)
    for r in results:
        print(
            f"{r['clip_id']:<36} {r['tp']:>3} {r['fp']:>3} {r['fn']:>3}  "
            f"{[round(t, 1) for t in r['fired_at']]}"
        )

    # ── Aggregate metrics ────────────────────────────────────────────────────
    tp_total = sum(r["tp"] for r in results)
    fp_total = sum(r["fp"] for r in results)
    fn_total = sum(r["fn"] for r in results)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 1.0
    recall    = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 1.0
    fpr_clips = sum(1 for r in results if r["fp"] > 0)
    total_clips = len(results)
    negative_clips = [r for r in results if r.get("fn", 0) == 0 and
                      not any(True for clip in abandoned_clips
                              if clip["clip_id"] == r["clip_id"] and clip.get("expected_alerts"))]

    print("\n" + "=" * 90)
    print("ABANDONED_OBJECT DETECTOR EVAL SUMMARY")
    print("=" * 90)
    print(f"precision = {precision:.3f}   recall = {recall:.3f}")
    print(f"tp={tp_total}  fp={fp_total}  fn={fn_total}")
    print(f"False positives on {fpr_clips}/{total_clips} clips")
    print(f"\ncalibration tagging check: {'PASS' if calibration_tag_ok else 'FAIL'}")

    # Pass criterion: zero FPs on negative clips (fp_total == 0) AND zero FNs on positive clips
    overall_pass = (fp_total == 0) and (fn_total == 0) and calibration_tag_ok
    print(
        f"\n{'ALL CHECKS PASSED' if overall_pass else 'SOME CHECKS FAILED'} "
        "(baseline: zero false positives on negative clips, "
        "zero false negatives on positive clips)"
    )
    if not overall_pass:
        print(
            "\nREMINDER: Re-run this harness whenever STATIC_MOVEMENT_PX_THRESHOLD, "
            "ABANDON_DURATION_SEC, or OWNERSHIP_RADIUS_METERS are changed."
        )

    return 0 if overall_pass else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 9 §5 abandoned-object detector eval harness"
    )
    parser.add_argument(
        "--redis-url", default=None,
        help="Override REDIS_URL (default: from .env/settings)",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(run_all(args.redis_url))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
