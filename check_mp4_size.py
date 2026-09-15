"""
check_mp4_size.py — Find total size and structure of stream/14.
"""
import requests

# Request 1 byte at range 0-1
r = requests.get("https://live.corp8.cloud/stream/14", headers={"Range": "bytes=0-1"}, timeout=8)
print("Range Status:", r.status_code)
print("Content-Range:", r.headers.get("Content-Range"))
print("Content-Length:", r.headers.get("Content-Length"))
print("Total Headers:", dict(r.headers))
