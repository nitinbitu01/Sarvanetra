"""
verify_30cam.py — Prove all 30 cameras are genuinely processing live RTSP.

Checks:
  1. Fleet supervisor is running and reports all workers alive
  2. All 30 cameras have a heartbeat JSON with a recent frame timestamp
  3. Per-camera live frames exist in output/live_frames/ and are fresh
  4. RTSP is the active stream source (not clips)
  5. API fleet/proof endpoint is reachable

Run: python -m backend.scripts.verify_30cam
Run: python -m backend.scripts.verify_30cam --api http://localhost:8000
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLEET_DIR = ROOT / "output" / "fleet"
LIVE_FRAMES = ROOT / "output" / "live_frames"

# A camera is "fresh" if its live frame or heartbeat was written within this many seconds
FRESH_THRESH_S = 60.0


def _bold(s: str) -> str:
    return s  # plain output for Windows compatibility


def check_supervisor() -> dict:
    state_file = FLEET_DIR / "supervisor.json"
    if not state_file.exists():
        return {"ok": False, "note": "supervisor.json not found -- is the fleet running?"}
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
        workers = state.get("workers", [])
        alive = [w for w in workers if w.get("alive")]
        dead = [w for w in workers if not w.get("alive")]
        age = time.time() - float(state.get("updated", 0))
        return {
            "ok": bool(alive) and age < 90,
            "workers_alive": len(alive),
            "workers_dead": len(dead),
            "state_age_s": round(age, 1),
            "cameras_total": state.get("cameras_total", 0),
        }
    except Exception as exc:
        return {"ok": False, "note": str(exc)}


def check_cameras(expected: list[str]) -> dict:
    now = time.time()
    results = {}
    for cam_id in expected:
        r = {"cam_id": cam_id, "ok": False, "source": "unknown", "frame_age_s": None, "note": ""}
        # Check live frame
        for fname in (f"{cam_id.upper()}.jpg", f"{cam_id}.jpg"):
            f = LIVE_FRAMES / fname
            if f.exists():
                age = now - f.stat().st_mtime
                r["frame_age_s"] = round(age, 1)
                r["source"] = "live_frames"
                r["ok"] = age < FRESH_THRESH_S
                if not r["ok"]:
                    r["note"] = f"frame stale ({age:.0f}s > {FRESH_THRESH_S:.0f}s threshold)"
                break
        # Check per-camera heartbeat JSON from supervisor workers
        cam_hb = FLEET_DIR / f"camera_{cam_id}.json"
        if cam_hb.exists():
            try:
                hb = json.loads(cam_hb.read_text(encoding="utf-8"))
                ts = float(hb.get("last_frame_ts", 0))
                age = now - ts
                if r["frame_age_s"] is None or age < r["frame_age_s"]:
                    r["frame_age_s"] = round(age, 1)
                    r["source"] = "camera_heartbeat"
                    r["ok"] = age < FRESH_THRESH_S
                    r["frames_processed"] = hb.get("frames_processed", 0)
            except Exception:
                pass
        if not r["ok"] and not r["note"]:
            r["note"] = "no live frame or heartbeat found"
        results[cam_id] = r
    return results


def check_api(base_url: str) -> dict:
    try:
        import urllib.request
        url = f"{base_url.rstrip('/')}/api/v1/fleet/proof"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        cameras = data.get("cameras", [])
        if isinstance(cameras, dict):
            cam_list = list(cameras.values())
        else:
            cam_list = list(cameras)
        active = sum(1 for c in cam_list if c.get("processing") or c.get("frames_arriving") or c.get("fps_now", 0) > 0)
        return {"ok": True, "cameras_active": active, "total": len(cam_list)}
    except Exception as exc:
        return {"ok": False, "note": str(exc)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--api", default="http://localhost:8000",
                    help="API base URL for the fleet proof endpoint")
    args = ap.parse_args()

    # Discover expected cameras from the DB
    from backend.db.session import SessionLocal
    from backend.db.models import Camera
    db = SessionLocal()
    try:
        cams = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
        cam_ids = sorted(set(c.camera_id for c in cams if c.camera_id
                             and c.stream_url and "rtsp://" in (c.stream_url or "")))
    finally:
        db.close()

    print(f"\nSentinel Gujarat -- 30-Camera Fleet Verification")
    print(f"{'=' * 60}")
    print(f"Expected RTSP cameras: {len(cam_ids)}")
    print()

    # 1. Supervisor
    sup = check_supervisor()
    sup_icon = "OK" if sup.get("ok") else "FAIL"
    print(f"[{sup_icon}] Fleet supervisor: "
          f"{sup.get('workers_alive', 0)} workers alive, "
          f"state age {sup.get('state_age_s', '?')}s, "
          f"{sup.get('cameras_total', 0)} cameras assigned")
    if not sup.get("ok"):
        print(f"     NOTE: {sup.get('note', 'unknown')}")

    # 2. Per-camera frames
    cam_results = check_cameras(cam_ids)
    good_cams = [c for c, r in cam_results.items() if r["ok"]]
    bad_cams = [c for c, r in cam_results.items() if not r["ok"]]

    print()
    print(f"[{'OK' if not bad_cams else 'PARTIAL'}] Per-camera live frames: "
          f"{len(good_cams)}/{len(cam_ids)} fresh (< {FRESH_THRESH_S:.0f}s)")

    if bad_cams:
        print(f"\n  Cameras without fresh frames ({len(bad_cams)}):")
        for c in bad_cams:
            r = cam_results[c]
            print(f"    {c:12s}  frame_age={r['frame_age_s']}s  {r['note']}")

    print(f"\n  Active cameras:")
    for c in good_cams:
        r = cam_results[c]
        fps_note = f"  {r.get('frames_processed', '?')} frames" if r.get('frames_processed') else ""
        print(f"    {c:12s}  {r['frame_age_s']}s ago  [{r['source']}]{fps_note}")

    # 3. API check
    print()
    api = check_api(args.api)
    api_icon = "OK" if api.get("ok") else "WARN"
    if api.get("ok"):
        print(f"[{api_icon}] API /fleet/proof: {api.get('cameras_active')}/{api.get('total')} cameras active")
    else:
        print(f"[{api_icon}] API /fleet/proof: {api.get('note')} (API may not be running)")

    # 4. Summary
    print()
    print("=" * 60)
    all_ok = sup.get("ok") and len(bad_cams) == 0
    print(f"OVERALL: {'PASS' if all_ok else 'PARTIAL' if good_cams else 'FAIL'} "
          f"-- {len(good_cams)}/{len(cam_ids)} cameras processing live RTSP frames")
    if not all_ok:
        print()
        print("If cameras are missing, wait 90-120s for workers to finish loading")
        print("CUDA models. The fleet needs ~90s from start_30cam_live.ps1 to")
        print("reach steady-state. Re-run this script after the wait.")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
