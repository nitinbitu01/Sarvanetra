"""
fetch_openapi.py — Fetch and inspect the entire openapi schema of live.corp8.cloud.
"""
import requests
import json

r = requests.get("https://live.corp8.cloud/openapi.json", timeout=8)
if r.status_code == 200:
    data = r.json()
    with open("corp8_openapi.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"[+] Downloaded openapi schema with {len(data.get('paths', {}))} endpoints:")
    for path, methods in data.get("paths", {}).items():
        for m, details in methods.items():
            print(f"  {m.upper():6s} {path:35s} - {details.get('summary', '')}")
