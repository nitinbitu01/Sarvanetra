import urllib.request
import urllib.parse
import http.cookiejar
import json
import re
import cv2
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_URL = "https://cctv.corp8.cloud"
PASSWORD = "6KL6-3ZBX-UJMA"

print("=" * 65)
print("CORP8 CCTV PORTAL — LOGIN & CAMERA DISCOVERY")
print("=" * 65)

cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [
    ("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"),
    ("Referer", f"{BASE_URL}/"),
    ("Origin", BASE_URL)
]

# 1. Login with password
login_data = urllib.parse.urlencode({"password": PASSWORD}).encode("utf-8")
login_req = urllib.request.Request(f"{BASE_URL}/auth/login", data=login_data, method="POST")

try:
    resp = opener.open(login_req, timeout=10)
    print("Authentication: SUCCESS (Status %d)" % resp.status)
    html = resp.read().decode("utf-8", errors="ignore")
except Exception as e:
    print("Authentication FAILED:", e)
    sys.exit(1)

# 2. Extract cameras from HTML
print("\n--- DISCOVERED CAMERAS ---")

# Search for video sources / camera elements
streams = set(re.findall(r'src=["\']([^"\']+\.m3u8[^"\']*)["\']', html))
streams.update(re.findall(r'(/live/stream/\d+/[^"\'\s<>]+)', html))
streams.update(re.findall(r'(https://[^"\'\s<>]+\.m3u8[^"\'\s<>]*)', html))

# Search for camera IDs and titles
cam_matches = re.findall(r'(Camera\s+\d+|CAM[_-]?\d+)', html, re.IGNORECASE)
print("Camera references found:", list(set(cam_matches)))

# Try to find all camera URLs
found_streams = []
for i in range(1, 25):
    # Test standard stream endpoints with authenticated session
    test_urls = [
        f"{BASE_URL}/live/stream/{i}/index.m3u8",
        f"https://live.corp8.cloud/live/stream/{i}/index.m3u8",
        f"{BASE_URL}/camera/{i}",
        f"{BASE_URL}/api/cameras/{i}/state",
    ]
    for u in test_urls:
        try:
            r = opener.open(u, timeout=2)
            if r.status == 200:
                content = r.read().decode("utf-8", errors="ignore")
                if "#EXTM3U" in content or "status" in content or "camera" in content.lower():
                    found_streams.append({"cam_id": i, "url": u, "type": "HLS" if ".m3u8" in u else "API"})
                    print(f"  [CAM {i:02d}] Active Stream -> {u}")
                    break
        except Exception:
            pass

print(f"\nTotal Active Stream Endpoints Found: {len(found_streams)}")

# 3. Test capturing a frame from the first active stream using OpenCV
if found_streams:
    hls_streams = [s for s in found_streams if s["type"] == "HLS"]
    if hls_streams:
        target_url = hls_streams[0]["url"]
        print(f"\nTesting OpenCV capture on: {target_url}")
        
        # In OpenCV, pass cookies / headers via ffmpeg options or direct session
        cap = cv2.VideoCapture(target_url)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        
        opened = cap.isOpened()
        ret, frame = (False, None)
        if opened:
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                out_path = ROOT / "live_corp8_cam1.jpg"
                cv2.imwrite(str(out_path), frame)
                print(f"Frame Captured Successfully: {w}x{h} px -> Saved as {out_path.name}")
            else:
                print("Stream opened but waiting for chunk data...")
        else:
            print("Direct cv2.VideoCapture requires session header cookie processor.")
        cap.release()
