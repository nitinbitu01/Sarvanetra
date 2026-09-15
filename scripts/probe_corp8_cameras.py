import json
import urllib.request
import sys
import cv2

print("=== CORP8 CAMERA METADATA & STREAM PROBE ===")
cameras = []
for i in range(1, 13):
    url = f"https://live.corp8.cloud/api/cameras/{i}/state"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as res:
            data = json.loads(res.read().decode())
            name = data.get("name", f"Camera {i}")
            loc = data.get("location", "")
            rtsp = data.get("rtsp_url", "")
            hls = f"https://live.corp8.cloud{data.get('hls_live_url')}" if data.get("hls_live_url") else ""
            status = data.get("status", "")
            cameras.append({
                "id": f"CAM-{i:02d}",
                "number": i,
                "name": name,
                "location": loc,
                "rtsp_url": rtsp,
                "hls_url": hls,
                "status": status
            })
            print(f"[{i:02d}] {name} ({loc}) -> Status: {status} | RTSP: {rtsp} | HLS: {hls}")
    except Exception as e:
        print(f"[{i:02d}] Error fetching state: {e}")

print("\n=== TESTING OPENCV CONNECTION TO CAMERA 1 & 2 ===")
for cam in cameras[:2]:
    # Test HLS first then RTSP
    for stream_url in [cam["hls_url"], cam["rtsp_url"]]:
        if not stream_url:
            continue
        print(f"Testing {cam['name']} stream: {stream_url} ...")
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        opened = cap.isOpened()
        ret, frame = (False, None)
        if opened:
            ret, frame = cap.read()
        print(f"  -> Opened: {opened}, Read Frame: {ret}, Shape: {frame.shape if ret else 'None'}")
        cap.release()
