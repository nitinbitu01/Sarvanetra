import sys, cv2, re, time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.scripts.night_enhancer import process_frame_adaptive
from backend.services.anpr_engine import ANPREngine

print("=" * 60)
print("FAST NIGHT ANPR EVALUATION (Real-Time)")
print("=" * 60)

engine = ANPREngine()
vdet   = YOLO(str(ROOT / "yolov8s.pt"))
VALID  = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$")

clips = {
    "CAM_07_0100": ROOT / "data/clips/CAM_07/CAM_07_0100.mp4",
    "CAM_08_0100": ROOT / "data/clips/CAM_08/CAM_08_0100.mp4",
    "CAM_09_0100": ROOT / "data/clips/CAM_09/CAM_09_0100.mp4",
}

for label, clip in clips.items():
    if not clip.exists():
        print("%s: NOT FOUND" % label)
        continue

    print("Evaluating %s ..." % label, end=" ", flush=True)
    t0 = time.time()
    cap  = cv2.VideoCapture(str(clip))
    fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(fps / 3))

    frame_idx = 0
    plates    = {}
    tbase     = abs(hash(label)) % 100000
    enhanced_count = 0
    blind_count    = 0

    while frame_idx < 600:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if frame_idx % skip != 0: continue

        eframe, diag = process_frame_adaptive(frame)

        if diag["mode"] == "blind":
            blind_count += 1
            continue
        if diag["mode"] == "night":
            enhanced_count += 1

        res = vdet.predict(eframe, imgsz=640, conf=0.20,
                           classes=[2,3,5,7], verbose=False)
        if not res or res[0].boxes is None: continue

        for i, box in enumerate(res[0].boxes[:3]):
            bbox   = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            r = engine.process_vehicle_track(
                frame=eframe, bbox=bbox,
                cls_id=cls_id,
                track_id=tbase + frame_idx * 10 + i
            )
            if r and r.get("plate") and r.get("confidence", 0) >= 0.45:
                p = r["plate"]
                c = r["confidence"]
                if VALID.match(p):
                    if p not in plates or c > plates[p]:
                        plates[p] = c

    cap.release()
    engine._confirmed_plates.clear()
    engine._track_votes.clear()
    elapsed = time.time() - t0

    avg = sum(plates.values()) / max(1, len(plates)) * 100
    print("DONE (%.1fs) | Enhanced: %d | Blind: %d | Plates Found: %d | Avg Conf: %.1f%%" % (
        elapsed, enhanced_count, blind_count, len(plates), avg
    ))
    if plates:
        print("   Sample Plates:", ", ".join(list(plates.keys())[:5]))
    print()
