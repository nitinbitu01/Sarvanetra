"""
test_stream_headers.py — Check the stream/14 protocol and format.
"""
import requests

r = requests.get("https://live.corp8.cloud/stream/14", headers={"Range": "bytes=0-1000"}, stream=True, timeout=8)
print("Status Code:", r.status_code)
print("Headers:", dict(r.headers))
content_chunk = r.raw.read(100)
print("First 100 bytes:", content_chunk[:32])
