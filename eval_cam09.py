import json

w = json.load(open("output/fleet/worker_w5.json", "r", encoding="utf-8"))
conf = w.get("confirmed_plates", {})
cam09_plates = {k: v for k, v in conf.items() if str(k).startswith("9")}

print(f"Total CAM_09 Confirmed Plates: {len(cam09_plates)}")
high_conf = 0
for tid, info in sorted(cam09_plates.items(), key=lambda x: str(x[0])):
    plate = info.get("plate", "")
    c = float(info.get("confidence", 0))
    locked = info.get("locked", False)
    is_high = (c >= 0.80)
    if is_high:
        high_conf += 1
    tag = "PASS (>=80%)" if is_high else "FAIL (<80%)"
    print(f"  Track {tid:10s}: Plate={plate:14s} Conf={c:.1%} Locked={locked!s:5s} -> {tag}")

pct = (high_conf / len(cam09_plates) * 100) if cam09_plates else 0
print(f"\n=======================================================")
print(f"CAM_09 ACCURACY SUMMARY:")
print(f"  Total Plates Confirmed: {len(cam09_plates)}")
print(f"  High Confidence (>=80%): {high_conf}")
print(f"  Recognition Accuracy:    {pct:.1f}%")
print(f"=======================================================")
