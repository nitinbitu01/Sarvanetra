from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
fleet_dir = root / "output" / "fleet"

for w in range(1, 6):
    f = fleet_dir / f"worker_w{w}.json"
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        print(f"\n==================== WORKER w{w} ====================")
        print(f"ANPR Attempts: {data.get('anpr_attempts')} | Successes: {data.get('anpr_successes')}")
        plates = data.get("confirmed_plates", {})
        print(f"Total confirmed plates held: {len(plates)}")
        for gid, pinfo in list(plates.items())[:10]:
            print(f"  GID {gid:10s} -> Plate: {pinfo.get('plate'):12s} | Conf: {pinfo.get('confidence')} | Locked: {pinfo.get('locked')}")
