"""
probe_corp8_endpoints.py — Find the underlying 12-hour video file on live.corp8.cloud.
"""
import requests

base_urls = [
    "https://live.corp8.cloud/videos/14.mkv",
    "https://live.corp8.cloud/videos/14.mp4",
    "https://live.corp8.cloud/videos/cam14.mkv",
    "https://live.corp8.cloud/videos/cam14.mp4",
    "https://live.corp8.cloud/videos/Camera 14.mkv",
    "https://live.corp8.cloud/videos/Camera_14.mkv",
    "https://live.corp8.cloud/videos/14_Delight.mkv",
    "https://live.corp8.cloud/media/14.mkv",
    "https://live.corp8.cloud/media/videos/14.mkv",
    "https://live.corp8.cloud/static/videos/14.mkv",
    "https://live.corp8.cloud/stream/14/video.mp4",
    "https://live.corp8.cloud/live/stream/14/video.mp4",
    "https://live.corp8.cloud/api/videos/14",
    "https://live.corp8.cloud/api/cameras/14/video",
    "https://live.corp8.cloud/api/cameras/14/stream",
    "https://live.corp8.cloud/api/cameras/14/download",
    "https://live.corp8.cloud/api/cameras/14/file",
]

print("[*] Probing corp8 raw video endpoints...")
for u in base_urls:
    try:
        r = requests.head(u, timeout=4)
        print(f"[{r.status_code}] {u} (Headers: {dict(r.headers).get('content-type')}, {dict(r.headers).get('content-length')})")
        if r.status_code == 200:
            print(f"  --> FOUND DIRECT FILE: {u} (Size: {r.headers.get('content-length')} bytes)")
    except Exception as e:
        print(f"[ERR] {u} : {e}")
