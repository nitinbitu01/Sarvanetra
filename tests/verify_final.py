"""
Sentinel Gujarat — 29-Point Final Verification
Run: python tests/verify_final.py
"""

import asyncio
import os
import sys
import time
import yaml
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class V:
    def __init__(self):
        self.ok = self.fail = 0

    def check(self, name, cond, detail=""):
        if cond:
            self.ok += 1
            print(f"  ✅ {name}")
        else:
            self.fail += 1
            print(f"  ❌ {name}" + (f"\n     → {detail}" if detail else ""))

    def section(self, t):
        print(f"\n{'═'*60}\n  {t}\n{'═'*60}")

    def summary(self):
        total = self.ok + self.fail
        pct = (self.ok / total * 100) if total else 0
        print(f"\n{'═'*60}")
        print(f"  RESULT: {self.ok}/{total} passed ({pct:.0f}%)")
        if self.fail == 0:
            print("  🎉 ALL CHECKS PASSED — Ready for live deployment!")
        else:
            print(f"  ⚠️  {self.fail} issue(s) need attention")
        print(f"{'═'*60}")


async def run():
    v = V()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 1. Config
    v.section("1. Configuration")
    cams = cfg.get("demo_cameras", [])
    v.check("Exactly 30 cameras", len(cams) == 30, f"Got {len(cams)}")
    v.check("All have GPS", all("lat" in c and "lon" in c for c in cams))
    v.check("All have URLs", all(c.get("url", "").startswith("http") for c in cams))
    v.check("All have departments", all(c.get("department") for c in cams))
    v.check("5 FPS target", cfg.get("stream_settings", {}).get("target_fps") == 5)
    v.check("3 worker groups", cfg.get("workers", {}).get("num_groups") == 3)

    # 2. Database
    v.section("2. Database Models")
    try:
        from backend.db.models import (
            Camera, Detection, Alert, JourneyEvent,
            ANPREvent, Officer, WatchlistPlate, WatchlistPerson, User
        )
        v.check("All 9 models import", True)
        v.check("Alert.evidence_hash", hasattr(Alert, "evidence_hash"))
        v.check("Alert.is_simulated", hasattr(Alert, "is_simulated"))
        v.check("JourneyEvent.is_cross_zone", hasattr(JourneyEvent, "is_cross_zone"))
        v.check("WatchlistPerson.reid_id", hasattr(WatchlistPerson, "reid_id"))
        v.check("User.role", hasattr(User, "role"))
    except Exception as e:
        v.check("Models import", False, str(e))

    # 3. Auth
    v.section("3. Auth Layer")
    try:
        from backend.auth.jwt_handler import create_token, decode_token, UserRole, hash_password, verify_password
        token = create_token("u01", UserRole.operator, "Test User")
        v.check("JWT token created", bool(token))
        decoded = decode_token(token)
        v.check("JWT decodes correctly", decoded["sub"] == "u01")
        v.check("Role encoded", decoded["role"] == "operator")
        hashed = hash_password("test123")
        v.check("Password hash works", verify_password("test123", hashed))
    except Exception as e:
        v.check("Auth layer", False, str(e))

    # 4. Services
    v.section("4. Service Modules")
    services = [
        ("backend.services.stream_reader", "FFmpegStreamReader"),
        ("backend.services.redis_queue", "SentinelRedisQueue"),
        ("backend.services.tracker", "BoTSORTTracker"),
        ("backend.services.reid_engine", "ReIDEngine"),
        ("backend.services.anpr_engine", "ANPREngine"),
        ("backend.services.crime_detector", "CrimeIncidentDetector"),
        ("backend.services.danger_scorer", "DangerScorer"),
        ("backend.services.evidence_locker", "EvidenceLocker"),
        ("backend.services.tts_engine", "HindiTTSEngine"),
        ("backend.services.gpu_manager", "GPUMemoryManager"),
    ]
    for mod, cls in services:
        try:
            import importlib
            m = importlib.import_module(mod)
            getattr(m, cls)
            v.check(f"{cls} importable", True)
        except Exception as e:
            v.check(f"{cls} importable", False, str(e))

    # 5. Crime Detector
    v.section("5. Crime Detector — 7 Categories")
    try:
        from backend.services.crime_detector import CrimeIncidentDetector
        cam_reg = {c["id"]: c for c in cams}
        det = CrimeIncidentDetector(cfg, cam_reg)
        for method in [
            "_check_wanted_suspects", "_check_stolen_vehicles",
            "_check_night_intrusion", "_check_loitering",
            "_check_crowd_surge", "_check_abandoned_objects",
            "_check_impossible_speed"
        ]:
            v.check(f"Method: {method}", hasattr(det, method))
    except Exception as e:
        v.check("Crime detector", False, str(e))

    # 6. Danger Scorer
    v.section("6. Danger Scoring")
    try:
        from backend.services.danger_scorer import DangerScorer
        scorer = DangerScorer(cfg)
        cam = cams[2]
        r = scorer.compute("stolen_vehicle", cam, time.time())
        v.check("Score returns dict", isinstance(r, dict))
        v.check("final_score present", "final_score" in r)
        v.check("severity present", r.get("severity") in ["low", "medium", "high", "critical"])
        v.check("breakdown present", bool(r.get("breakdown")))
        # Night multiplier test
        import time as t
        night_ts = t.mktime(t.strptime("2026-08-21 23:30:00", "%Y-%m-%d %H:%M:%S"))
        rn = scorer.compute("stolen_vehicle", cam, night_ts)
        v.check("Night multiplier applied", rn["breakdown"].get("time_mult", 1.0) > 1.0)
    except Exception as e:
        v.check("Danger scorer", False, str(e))

    # 7. Evidence Locker
    v.section("7. Evidence Locker")
    try:
        from backend.services.evidence_locker import EvidenceLocker
        from datetime import datetime
        locker = EvidenceLocker(cfg)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await locker.save(
            frame=frame, camera_id="CAM_TEST",
            timestamp=datetime.utcnow(), alert_type="verify_test"
        )
        v.check("Evidence saved", bool(result))
        v.check("SHA-256 is 64 chars", len(result.get("sha256", "")) == 64)
        v.check("Custody JSON created", os.path.exists(result.get("custody_path", "")))
    except Exception as e:
        v.check("Evidence locker", False, str(e))

    # 8. ANPR Regex
    v.section("8. ANPR Plate Validation")
    try:
        from backend.services.anpr_engine import ANPREngine
        eng = ANPREngine(cfg)
        valid = ["GJ05AB1234", "GJ01CD5678", "MH12DE3456"]
        invalid = ["INVALID", "123456", "HELLO"]
        v.check("Valid plates match", all(eng.INDIAN_PLATE_PATTERN.match(p) for p in valid))
        v.check("Invalid plates rejected", all(not eng.INDIAN_PLATE_PATTERN.match(p) for p in invalid))
        v.check("Min confidence ≥ 0.75", cfg.get("ai_models", {}).get("anpr", {}).get("min_confidence", 0.75) >= 0.75)
    except Exception as e:
        v.check("ANPR validation", False, str(e))

    # 9. Supervisor
    v.section("9. Process Supervisor")
    try:
        from backend.supervisor.watchdog import ProcessSupervisor
        sup = ProcessSupervisor()
        v.check("Supervisor instantiates", True)
        v.check("add_worker method exists", hasattr(sup, "add_worker"))
        v.check("monitor_loop method exists", hasattr(sup, "monitor_loop"))
        v.check("stop_all method exists", hasattr(sup, "stop_all"))
    except Exception as e:
        v.check("Supervisor", False, str(e))

    # 10. Live API (optional)
    v.section("10. Live API (requires server)")
    try:
        import httpx
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get("http://localhost:8000/api/system/health")
            v.check("/health returns 200", r.status_code == 200)
            v.check("Version 3.0.0", r.json().get("version") == "3.0.0")
            v.check("30 cameras registered", r.json().get("cameras") == 30)
    except Exception:
        print("  ⚠️  Server offline — start with: uvicorn backend.main:app --reload")

    v.summary()


if __name__ == "__main__":
    asyncio.run(run())
