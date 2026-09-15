"""
test_hls_playlist.py — Inspect the HLS playlist and segment structure for camera 14.
"""
import requests

u = "https://live.corp8.cloud/live/stream/14/index.m3u8"
r = requests.get(u, timeout=5)
print("Status:", r.status_code)
print("Playlist content:")
print(r.text[:800])

lines = r.text.strip().split("\n")
for l in lines:
    if l.endswith(".m3u8"):
        sub_u = f"https://live.corp8.cloud/live/stream/14/{l}"
        sub_r = requests.get(sub_u, timeout=5)
        print("\n--- Sub-playlist:", sub_u)
        print(sub_r.text[:1000])
