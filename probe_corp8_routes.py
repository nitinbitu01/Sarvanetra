"""
probe_corp8_routes.py — Discover server structure and routes on live.corp8.cloud.
"""
import requests

routes = [
    "/openapi.json",
    "/docs",
    "/api/docs",
    "/api/cameras/14",
    "/api/cameras/14/segments",
    "/api/cameras/14/playlist",
    "/api/cameras/14/hls",
    "/api/cameras/14/manifest",
    "/api/cameras/14/info",
    "/api/cameras/14/clip",
    "/api/prepare/status",
    "/api/catalog",
    "/api/config",
    "/live/stream/14/index.m3u8",
]

for r in routes:
    u = f"https://live.corp8.cloud{r}"
    try:
        res = requests.get(u, timeout=3)
        print(f"[{res.status_code}] {r} -> {res.text[:120] if res.status_code == 200 else ''}")
    except Exception as e:
        print(f"[ERR] {r}: {e}")
