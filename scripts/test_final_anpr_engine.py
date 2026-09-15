import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

engine = get_anpr_engine()
print("Engine successfully loaded!")
print("plate_detector:", engine.plate_detector is not None)
print("plate_detector_small:", engine.plate_detector_small is not None)

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Testing real anpr_engine on {len(crops)} motorcycle crops...")

successes = 0

for i, cpath in enumerate(crops):
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    # Synthetic track id
    tid = 1000 + i
    res = engine.process_vehicle_track(img, [0, 0, w, h], cls_id=3, track_id=tid)
    if res and res.get("plate"):
        successes += 1
        print(f"  [SUCCESS] {cpath.name:30s} -> Plate: {res['plate']:12s} | Conf: {res.get('confidence'):.2f} | State: {res.get('state')} | Locked: {res.get('locked')}")
    else:
        # Check why
        prep, raw, q = engine.extract_plate_candidate(img, [0, 0, w, h], cls_id=3)
        sz = f"{raw.shape[1]}x{raw.shape[0]}" if raw is not None else "no-box"
        print(f"  [NO-PLATE] {cpath.name:30s} -> Crop: {sz} (q={q:.1f})")

print(f"\nFinal Score: {successes}/{len(crops)} real motorcycle crops successfully confirmed!")
