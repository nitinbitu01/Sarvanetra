"""corp8 health-check safety: gated interval, circuit breaker, fail-closed.

Deliberately mocked rather than hitting the real portal. That account has
already absorbed a full day of rate-limit debugging, and this module's whole
purpose is to never repeat that pattern automatically in the background — so
the thing under test is exactly the logic that decides whether and how often
to make a real request, not the request itself.
"""
from __future__ import annotations

import time

import pytest

import backend.services.camera_heartbeat as hb


class _FakeCamera:
    def __init__(self, cam_id: str, url: str, status: str = "ONLINE",
                consecutive_failures: int = 0):
        self.id = cam_id
        self.name = cam_id
        self.url = url
        self.status = status
        self.consecutive_failures = consecutive_failures
        self.last_heartbeat_at = None


@pytest.fixture(autouse=True)
def reset_corp8_state():
    """Each test starts with a clean slate for the module-level trackers."""
    hb._corp8_last_check_at = 0.0
    hb._corp8_circuit_open_until = 0.0
    yield
    hb._corp8_last_check_at = 0.0
    hb._corp8_circuit_open_until = 0.0


def test_extracts_portal_id_from_stored_url():
    assert hb._corp8_camera_id(
        "https://cctv.corp8.cloud/cam21/index.m3u8") == "cam21"
    assert hb._corp8_camera_id("https://live.corp8.cloud/stream/21") is None
    assert hb._corp8_camera_id(None) is None
    assert hb._corp8_camera_id("") is None


def test_batch_probe_returns_only_resolved_cameras(monkeypatch):
    """A camera whose URL cannot be mapped to a portal id is skipped, not
    marked unhealthy — an unresolvable URL is not evidence of an outage."""
    cams = [
        _FakeCamera("CAM_21", "https://cctv.corp8.cloud/cam21/index.m3u8"),
        _FakeCamera("CAM_LOCAL", "file:///demo/clip.mp4"),
    ]

    class FakeSession:
        def probe(self, cid):
            assert cid == "cam21"
            return {"ok": True, "status": 200}

    monkeypatch.setattr(hb, "get_corp8_session", lambda: FakeSession(),
                        raising=False)
    import backend.services.corp8_session as corp8_mod
    monkeypatch.setattr(corp8_mod, "get_corp8_session", lambda: FakeSession())

    out = hb._corp8_batch_probe_sync(cams)
    assert out == {"CAM_21": True}


def test_circuit_breaker_trips_on_403_and_stops_the_batch(monkeypatch):
    cams = [
        _FakeCamera("CAM_01", "https://cctv.corp8.cloud/cam01/index.m3u8"),
        _FakeCamera("CAM_02", "https://cctv.corp8.cloud/cam02/index.m3u8"),
        _FakeCamera("CAM_03", "https://cctv.corp8.cloud/cam03/index.m3u8"),
    ]
    calls = []

    class FakeSession:
        def probe(self, cid):
            calls.append(cid)
            if cid == "cam02":
                return {"ok": False, "status": 403}
            return {"ok": True, "status": 200}

    import backend.services.corp8_session as corp8_mod
    monkeypatch.setattr(corp8_mod, "get_corp8_session", lambda: FakeSession())
    monkeypatch.setattr(hb.time, "sleep", lambda _s: None)  # don't slow the test

    out = hb._corp8_batch_probe_sync(cams)

    # cam03 must never have been probed once cam02 returned 403.
    assert calls == ["cam01", "cam02"]
    assert out == {"CAM_01": True}
    assert hb._corp8_circuit_open_until > time.monotonic()


def test_circuit_breaker_blocks_the_next_batch_entirely(monkeypatch):
    cams = [_FakeCamera("CAM_01", "https://cctv.corp8.cloud/cam01/index.m3u8")]
    calls = []

    class FakeSession:
        def probe(self, cid):
            calls.append(cid)
            return {"ok": True, "status": 200}

    import backend.services.corp8_session as corp8_mod
    monkeypatch.setattr(corp8_mod, "get_corp8_session", lambda: FakeSession())

    hb._corp8_circuit_open_until = time.monotonic() + 999
    out = hb._corp8_batch_probe_sync(cams)

    assert calls == []          # no request made at all while the breaker is open
    assert out == {}


def test_probe_exception_on_one_camera_does_not_abort_the_batch(monkeypatch):
    cams = [
        _FakeCamera("CAM_01", "https://cctv.corp8.cloud/cam01/index.m3u8"),
        _FakeCamera("CAM_02", "https://cctv.corp8.cloud/cam02/index.m3u8"),
    ]

    class FakeSession:
        def probe(self, cid):
            if cid == "cam01":
                raise ConnectionError("simulated network blip")
            return {"ok": True, "status": 200}

    import backend.services.corp8_session as corp8_mod
    monkeypatch.setattr(corp8_mod, "get_corp8_session", lambda: FakeSession())
    monkeypatch.setattr(hb.time, "sleep", lambda _s: None)

    out = hb._corp8_batch_probe_sync(cams)
    assert out == {"CAM_02": True}   # cam01 absent, not recorded as False


@pytest.mark.asyncio
async def test_cycle_skips_corp8_when_interval_has_not_elapsed(monkeypatch):
    """The outer cycle must not even attempt a probe before the interval —
    this is what stops a 30s heartbeat loop from hammering corp8 every tick."""
    calls = {"n": 0}

    def fake_batch(cams):
        calls["n"] += 1
        return {c.id: True for c in cams}

    monkeypatch.setattr(hb, "_corp8_batch_probe_sync", fake_batch)
    hb._corp8_last_check_at = time.monotonic()   # "just checked"

    corp8_cams = [_FakeCamera("CAM_01", "https://cctv.corp8.cloud/cam01/index.m3u8")]

    class FakeQuery:
        def filter(self, *a, **kw):
            return self
        def all(self):
            return corp8_cams

    class FakeDB:
        def query(self, *a, **kw):
            return FakeQuery()
        def commit(self):
            pass
        def rollback(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(hb, "SessionLocal", lambda: FakeDB())

    await hb._run_heartbeat_cycle()
    assert calls["n"] == 0, "probed corp8 before its interval had elapsed"


@pytest.mark.asyncio
async def test_cycle_leaves_unchecked_corp8_cameras_untouched(monkeypatch):
    """A camera the batch probe could not resolve/reach must keep its prior
    status and failure count rather than being nudged toward OFFLINE."""
    cam = _FakeCamera("CAM_01", "https://cctv.corp8.cloud/cam01/index.m3u8",
                      status="ONLINE", consecutive_failures=0)

    monkeypatch.setattr(hb, "_corp8_batch_probe_sync", lambda cams: {})
    hb._corp8_last_check_at = 0.0   # interval elapsed

    class FakeQuery:
        def filter(self, *a, **kw):
            return self
        def all(self):
            return [cam]

    class FakeDB:
        def query(self, *a, **kw):
            return FakeQuery()
        def commit(self):
            pass
        def rollback(self):
            pass
        def close(self):
            pass

    monkeypatch.setattr(hb, "SessionLocal", lambda: FakeDB())

    await hb._run_heartbeat_cycle()

    assert cam.status == "ONLINE"
    assert cam.consecutive_failures == 0
    assert cam.last_heartbeat_at is None, (
        "an untouched camera's heartbeat timestamp must not move — that "
        "timestamp is read elsewhere as 'last time we actually knew'")
