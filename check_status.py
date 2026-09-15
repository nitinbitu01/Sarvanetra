import urllib.request
import json
import os

try:
    with urllib.request.urlopen('http://127.0.0.1:8000/api/v1/top10/status', timeout=5) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print(f"Active Workers: {data.get('active_workers', 0)} | Total FPS: {data.get('total_fps', 0):.1f} | Total Plates: {data.get('total_plates', 0)}")
        cams = data.get('cameras', [])
        for c in cams:
            cid = c.get('cam_id', '')
            fps = c.get('per_camera_fps', 0)
            plates = c.get('plates_read', 0)
            stream_st = c.get('stream_state', '')
            ai_st = c.get('ai_state', '')
            veh = c.get('vehicles_active', 0)
            print(f"  [{cid}] Stream: {stream_st:8s} | AI: {ai_st:8s} | FPS: {fps:4.1f} | Vehicles: {veh:3d} | Plates Read: {plates:3d}")
except Exception as e:
    print("API Error:", e)

# Also inspect all worker files
fleet_dir = "output/fleet"
if os.path.exists(fleet_dir):
    print("\n--- Worker Detailed ANPR Diagnostics ---")
    for fn in sorted(os.listdir(fleet_dir)):
        if fn.startswith("worker_") and fn.endswith(".json"):
            fp = os.path.join(fleet_dir, fn)
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    wdata = json.load(f)
                    wid = wdata.get("worker_id", "")
                    anpr_cams = wdata.get("anpr_cameras", [])
                    attempts = wdata.get("anpr_attempts", 0)
                    successes = wdata.get("anpr_successes", 0)
                    confirmed = wdata.get("confirmed_plates", {})
                    print(f"Worker {wid} ({fn}): ANPR Cams: {anpr_cams} | Attempts: {attempts} | Successes: {successes} | Confirmed Count: {len(confirmed)}")
                    for tid, pinfo in sorted(confirmed.items(), key=lambda x: str(x[0]))[-6:]:
                        plate = pinfo.get('plate', '')
                        conf = pinfo.get('confidence', 0)
                        locked = pinfo.get('locked', False)
                        print(f"    Track {tid}: Plate='{plate}' Conf={conf:.1%} Locked={locked}")
            except Exception as e:
                print(f"  {fn} error: {e}")
