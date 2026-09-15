import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

engine = get_anpr_engine()
print("Engine loaded! Device:", engine.device)

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Testing OCR pipeline on {len(crops)} crops...")

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    # Synthetic bbox around motorcycle
    bbox = [0, 0, w, h]
    # cls_id 3 = motorcycle
    res = engine.process_vehicle_track(img, bbox, cls_id=3, track_id=hash(cpath.name) % 100000)
    if res and res.get("plate"):
        print(f"SUCCESS: {cpath.name} ({w}x{h}) -> Plate: {res['plate']} | Conf: {res.get('confidence')} | Raw: {res.get('raw_read')}")
    else:
        # Check why it didn't pass
        prep, raw_bgr, q = engine.extract_plate_candidate(img, bbox, cls_id=3)
        status = "No plate box found" if raw_bgr is None else f"Plate crop {raw_bgr.shape[1]}x{raw_bgr.shape[0]} (q={q:.1f})"
        print(f"FAILED: {cpath.name} ({w}x{h}) -> {status}")
