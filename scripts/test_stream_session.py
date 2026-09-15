import urllib.request
import http.cookiejar
import re

print("=== TESTING SESSION-AWARE HLS STREAM ===")

# Create cookie-enabled opener
cj = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")]

# Step 1: Visit camera page to establish session/cookies
print("Step 1: Visiting https://live.corp8.cloud/camera/1 ...")
resp1 = opener.open("https://live.corp8.cloud/camera/1")
print(f"  -> Page loaded with status {resp1.status}")

# Step 2: Fetch index.m3u8
print("Step 2: Fetching master playlist ...")
resp2 = opener.open("https://live.corp8.cloud/live/stream/1/index.m3u8")
master_content = resp2.read().decode()
print("Master playlist:\n" + master_content)

# Extract stream playlist line
lines = [l.strip() for l in master_content.splitlines() if l.strip() and not l.startswith("#")]
if lines:
    sub_url = "https://live.corp8.cloud/live/stream/1/" + lines[0]
    print(f"Step 3: Fetching chunklist from {sub_url} ...")
    resp3 = opener.open(sub_url)
    chunk_content = resp3.read().decode()
    print("Chunklist playlist:\n" + chunk_content[:500] + "...")
