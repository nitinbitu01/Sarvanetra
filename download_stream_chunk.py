"""
download_stream_chunk.py — Download a 10MB chunk of stream/14.
"""
import requests
import time

url = "https://live.corp8.cloud/stream/14"
print("[*] Downloading 10MB chunk...")
t0 = time.time()
r = requests.get(url, stream=True, timeout=10)
if r.status_code in (200, 206):
    total = 0
    with open("cam14_sample.mp4", "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
                total += len(chunk)
                if total >= 15 * 1024 * 1024:  # 15 MB
                    break
    t1 = time.time()
    print(f"[+] Downloaded {total / (1024*1024):.2f} MB in {t1-t0:.2f}s!")
else:
    print(f"[-] Status code: {r.status_code}")
