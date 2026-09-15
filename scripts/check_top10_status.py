import urllib.request
import json

res = urllib.request.urlopen("http://localhost:8000/api/v1/top10/status")
data = json.loads(res.read().decode())
cameras = data.get("cameras", [])

print("=" * 65)
print(f"{'Camera':8s} | {'FPS':5s} | {'Drop %':7s} | {'Vehicles':9s} | {'Plates Read':11s}")
print("=" * 65)

total_plates = 0
total_fps = 0.0

for c in cameras:
    cid = c.get('cam_id', 'UNKNOWN')
    fps = c.get('per_camera_fps', 0.0) or 0.0
    drops = c.get('drop_pct', 0.0) or 0.0
    veh = c.get('vehicles_active', 0) or 0
    plates = c.get('plates_read', 0) or 0
    total_plates += plates
    total_fps += fps
    print(f"{cid:8s} | {fps:5.1f} | {drops:6.1f}% | {veh:9d} | {plates:11d}")

print("=" * 65)
print(f"Aggregate FPS: {total_fps:.1f} across {len(cameras)} cameras | Total Plates Read: {total_plates}")
