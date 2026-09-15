import json
import urllib.request
import sys

# Ensure unbuffered flush
sys.stdout.reconfigure(line_buffering=True)

print("=== CORP8 CAMERA METADATA & STREAM PROBE ===", flush=True)
cameras = []

for i in range(1, 13):
    url = f"https://live.corp8.cloud/api/cameras/{i}/state"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=3) as res:
            data = json.loads(res.read().decode())
            name = data.get("name", f"Camera {i}")
            loc = data.get("location", "")
            rtsp = data.get("rtsp_url", "")
            hls_rel = data.get("hls_live_url", "")
            hls = f"https://live.corp8.cloud{hls_rel}" if hls_rel else ""
            stream_mp4 = f"https://live.corp8.cloud{data.get('stream_url', '')}" if data.get('stream_url') else ""
            status = data.get("status", "")

            # Check if HLS or Stream is reachable
            hls_ok = False
            if hls:
                try:
                    hls_req = urllib.request.Request(hls, headers={"User-Agent": "Mozilla/5.0"})
                    with urllib.request.urlopen(hls_req, timeout=3) as hls_res:
                        hls_ok = (hls_res.status == 200)
                except Exception:
                    hls_ok = False

            cameras.append({
                "id": f"CAM-{i:02d}",
                "number": i,
                "name": name,
                "location": loc,
                "rtsp_url": rtsp,
                "hls_url": hls,
                "stream_mp4": stream_mp4,
                "hls_reachable": hls_ok,
                "status": status
            })
            print(f"[{i:02d}] {name} ({loc}) -> Status: {status} | HLS HTTP Reachable: {hls_ok} | RTSP: {rtsp}", flush=True)
    except Exception as e:
        print(f"[{i:02d}] Error fetching state: {e}", flush=True)

# Save the discovered camera configuration as a json file
with open("data/corp8_cameras.json", "w") as f:
    json.dump(cameras, f, indent=2)

print("\nSaved configuration to data/corp8_cameras.json", flush=True)
