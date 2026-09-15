"""ONVIF discovery must not present the configured-camera fallback as a live
network find.

Regression this guards: discover_devices() used to fall back to the
configured camera list (config.yaml's demo_cameras — this deployment's real
30-camera fleet) whenever no physical device answered the WS-Discovery
multicast, and handed that back in the exact same shape as a genuine hit —
same field names, a guessed vendor="Hikvision", a constructed
rtsp://{ip}:554/onvif1 URL never actually verified. A caller (or the
onboarding UI built on top of it) could not tell a real discovery result
from a guess. Every device now carries `source`, and the route reports
`live_probe_count` so the two are never conflated.

Also covers: /adapters/supported and /discover/onvif previously had no auth
dependency at all — the only two routes in cameras.py like that — while the
fallback path can return real registry rows (IP, vendor, name).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
import backend.services.camera_adapters.onvif_adapter as onvif_mod

client = TestClient(app)

# Cached rather than logging in per-test: the login endpoint is rate-limited
# (10/min) and shared across every test file in a full-suite run — a fresh
# login per test here is exactly the pattern that starved
# test_department_scoping.py's own logins earlier in this project.
_TOKEN: str | None = None


@pytest.fixture(autouse=True)
def _real_auth(clear_get_current_user_override):
    yield


def _admin_headers() -> dict:
    global _TOKEN
    if _TOKEN is None:
        r = client.post("/api/v1/auth/login",
                        json={"username": "admin", "password": "admin123"})
        assert r.status_code == 200, r.text
        _TOKEN = r.json()["access_token"]
    return {"Authorization": f"Bearer {_TOKEN}"}


def test_no_auth_at_all_is_refused():
    r1 = client.get("/api/v1/cameras/adapters/supported")
    r2 = client.post("/api/v1/cameras/discover/onvif")
    assert r1.status_code == 401
    assert r2.status_code == 401


def test_fallback_devices_are_labeled_not_claimed_as_live(monkeypatch):
    async def _no_live_hits(subnet="192.168.1.255", timeout=2.0):
        return [{
            "ip": "10.0.0.5", "vendor": "unknown",
            "name": "Configured camera (10.0.0.5)",
            "source": "configured_fallback",
            "note": "not a verified ONVIF-reachable device",
        }]
    monkeypatch.setattr(onvif_mod.ONVIFAdapter, "discover_devices", _no_live_hits)

    h = _admin_headers()
    r = client.post("/api/v1/cameras/discover/onvif", headers=h)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["live_probe_count"] == 0
    assert all(d["source"] == "configured_fallback" for d in data["devices"])


def test_live_hit_is_counted_and_labeled(monkeypatch):
    async def _one_live_hit(subnet="192.168.1.255", timeout=2.0):
        return [{
            "ip": "192.168.1.50", "vendor": "Hikvision", "name": "Hikvision IP Camera",
            "service_url": "http://192.168.1.50:80/onvif/device_service",
            "stream": "rtsp://192.168.1.50:554/onvif1",
            "source": "live_probe",
        }]
    monkeypatch.setattr(onvif_mod.ONVIFAdapter, "discover_devices", _one_live_hit)

    h = _admin_headers()
    r = client.post("/api/v1/cameras/discover/onvif", headers=h)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["live_probe_count"] == 1
    assert data["devices"][0]["source"] == "live_probe"


def test_supported_adapters_requires_auth_but_then_works():
    h = _admin_headers()
    r = client.get("/api/v1/cameras/adapters/supported", headers=h)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "hikvision" in data["supported_vendors"]
    assert "hls" in data["supported_vendors"]
