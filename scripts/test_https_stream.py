import urllib.request
import cv2
import numpy as np
import tempfile
import os

print("=== TESTING HTTPS STREAM CAPTURE ===")

# Test 1: Fetch 500KB of progressive stream from https://live.corp8.cloud/stream/1
url = "https://live.corp8.cloud/stream/1"
req = urllib.request.Request(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Range": "bytes=0-2000000"
})

try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = resp.read()
        print(f"Downloaded {len(data)} bytes from {url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        print(f"OpenCV read from temp MP4: Success={ret}, Shape={frame.shape if ret else 'None'}")
        cap.release()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
except Exception as e:
    print(f"Error testing stream 1: {e}")
