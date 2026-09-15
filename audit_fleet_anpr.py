import json
import os

fleet_dir = "output/fleet"
total_all_confirmed = 0
total_all_passed = 0

print(f"{'Camera':<8} | {'Total Plates':<12} | {'>=80% Conf':<12} | {'Accuracy %':<10} | Sample Plates")
print("-" * 75)

worker_files = sorted([f for f in os.listdir(fleet_dir) if f.startswith("worker_") and f.endswith(".json")])

for wf in worker_files:
    wp = os.path.join(fleet_dir, wf)
    try:
        with open(wp, "r", encoding="utf-8") as f:
            wdata = json.load(f)
            confirmed = wdata.get("confirmed_plates", {})
            anpr_cams = wdata.get("anpr_cameras", [])
            for cam in anpr_cams:
                cam_num = cam.replace("CAM_", "").lstrip("0") or "0"
                cam_plates = {k: v for k, v in confirmed.items() if str(k).startswith(cam_num)}
                tot = len(cam_plates)
                passed = sum(1 for v in cam_plates.values() if float(v.get("confidence", 0)) >= 0.80)
                acc = (passed / tot * 100) if tot > 0 else 0.0
                samples = [f"{v.get('plate')}({float(v.get('confidence',0)):.0%})" for v in list(cam_plates.values())[-2:]]
                print(f"{cam:<8} | {tot:<12} | {passed:<12} | {acc:<9.1f}% | {', '.join(samples)}")
                total_all_confirmed += tot
                total_all_passed += passed
    except Exception as e:
        print(f"Error {wf}: {e}")

print("-" * 75)
overall_acc = (total_all_passed / total_all_confirmed * 100) if total_all_confirmed > 0 else 0.0
print(f"FLEET TOTAL: {total_all_confirmed} Plates Confirmed across fleet | {total_all_passed} Passed (>=80%) | Overall Accuracy: {overall_acc:.1f}%")
