import urllib.request
import json

try:
    with urllib.request.urlopen("http://localhost:8000/api/v1/top10/status") as r:
        d = json.loads(r.read().decode())
    print("Summary:", d["summary"])
    print("GPU:", d.get("system", {}).get("gpu_pct"), "VRAM:", d.get("system", {}).get("vram_mb"))
    for c in d["cameras"]:
        print(f"{c['cam_id']}: state={c['ai_state']}, fps={c['per_camera_fps']}, drop={c['drop_pct']}%, anpr_active={c['anpr_active']}, plates={c['plates_read']}")
except Exception as e:
    print("Error:", e)
