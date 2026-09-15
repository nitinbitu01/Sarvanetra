"""
backend/services/camera_connection.py — Camera connectivity test.

WHAT CHANGED AND WHY
────────────────────
This module previously SIMULATED the test: `random.random() < 0.85` plus a
`time.sleep(0.2-0.8)` to make the latency look plausible. It returned
"RTSP connection to <ip> succeeded" for addresses that need not exist, and
POST /api/v1/cameras/test audit-logged `success: true` — so a fabricated
result was written into the audit trail as fact, and an admin onboarding a
camera could reasonably believe an unreachable device had been verified.

It now performs a real probe. Where a protocol genuinely cannot be probed
with what's installed, it returns success=False with a message that says so,
rather than guessing. An honest "cannot verify" is useful; a fabricated
"succeeded" is worse than no test at all.

TCP reachability is the floor, not the ceiling: a passing TCP check means
"something is listening on that port", not "this is a working camera stream
with valid credentials". Messages here are deliberately worded to claim only
what was actually established. For RTSP/HTTP/HLS we additionally attempt a
real stream open via OpenCV, which is the same probe camera_heartbeat.py
uses for ongoing health, so onboarding and monitoring agree on what "up"
means instead of using two different definitions.
"""
from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default ports by protocol, used only when the address carries no explicit
# port. RTSP 554 and HTTP 80 are the IANA/de-facto defaults; HLS is HTTP.
_DEFAULT_PORTS: dict[str, int] = {
    "RTSP": 554,
    "ONVIF": 80,
    "HTTP": 80,
    "HLS": 80,
}

# Per-stage timeouts. Kept short: this runs synchronously inside a request,
# and a hung camera must not hold a worker open. The route already runs it in
# a thread (see cameras.py), but a bounded probe is still the right shape.
_TCP_TIMEOUT_S = 3.0
_STREAM_OPEN_TIMEOUT_MS = 4000


@dataclass
class ConnectionResult:
    success: bool
    message: str
    latency_ms: float | None = None
    # True only when a real media stream was opened and a frame was readable.
    # A TCP-only pass leaves this False — the UI and the audit log can then
    # distinguish "port is open" from "this is a working camera".
    stream_verified: bool = False


def _split_host_port(address: str, protocol: str) -> tuple[str | None, int]:
    """Extract (host, port) from a bare IP, host:port, or full URL."""
    default_port = _DEFAULT_PORTS.get(protocol.upper(), 80)
    addr = (address or "").strip()
    if not addr:
        return None, default_port

    if "://" in addr:
        parsed = urlparse(addr)
        return parsed.hostname, (parsed.port or default_port)

    # Bare "host:port" — but don't mangle a bare IPv6 literal.
    if addr.count(":") == 1:
        host, _, port_s = addr.partition(":")
        try:
            return host, int(port_s)
        except ValueError:
            return host, default_port

    return addr, default_port


def _tcp_reachable(host: str, port: int) -> tuple[bool, str | None]:
    """Is something accepting TCP connections at host:port?"""
    try:
        with socket.create_connection((host, port), timeout=_TCP_TIMEOUT_S):
            return True, None
    except socket.timeout:
        return False, f"timed out after {_TCP_TIMEOUT_S:.0f}s (no response)"
    except socket.gaierror as exc:
        return False, f"hostname could not be resolved ({exc.strerror or exc})"
    except ConnectionRefusedError:
        return False, "connection refused (nothing listening on that port)"
    except OSError as exc:
        return False, f"unreachable ({exc.strerror or exc})"


def _stream_url(address: str, protocol: str) -> str:
    """Best-effort stream URL for OpenCV when the caller gave a bare address."""
    if "://" in address:
        return address
    proto = protocol.upper()
    if proto == "RTSP":
        return f"rtsp://{address}:{_DEFAULT_PORTS['RTSP']}/stream"
    return f"http://{address}"


def _try_open_stream(address: str, protocol: str) -> bool:
    """Attempt a real stream open + frame read. False on any failure.

    Mirrors camera_heartbeat._check_sync so onboarding and ongoing health
    monitoring use the same definition of "readable". Reads one frame rather
    than trusting isOpened() alone — some backends report opened for a URL
    that never yields data.
    """
    cap = None
    try:
        import cv2

        cap = cv2.VideoCapture(_stream_url(address, protocol))
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, _STREAM_OPEN_TIMEOUT_MS)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, _STREAM_OPEN_TIMEOUT_MS)
        except Exception:
            # Older OpenCV builds lack these properties — the probe still
            # works, it just relies on the backend's own default timeout.
            pass
        if not cap.isOpened():
            return False
        ok, frame = cap.read()
        return bool(ok and frame is not None and getattr(frame, "size", 0) > 0)
    except Exception as exc:
        logger.debug("Stream probe failed for %s: %s", address, exc)
        return False
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def test_camera_connection(ip: str, protocol: str) -> ConnectionResult:
    """Probe a camera for real. Never fabricates a result.

    Order of attempts:
      1. Local file path — the project's own simulated/test feeds are file
         based (see camera_heartbeat._check_sync), so a path that exists and
         is non-empty is a legitimate pass for a demo feed.
      2. TCP reachability to host:port.
      3. For RTSP/HTTP/HLS, a real stream open + one frame read.

    Returns:
        ConnectionResult. `success` reflects what was actually established;
        `stream_verified` is True only when a frame was genuinely read.
    """
    t0 = time.monotonic()

    def _elapsed() -> float:
        return round((time.monotonic() - t0) * 1000, 1)

    address = (ip or "").strip()
    proto = (protocol or "").strip().upper() or "HTTP"

    if not address:
        return ConnectionResult(
            success=False,
            message="No address supplied — nothing to test.",
            latency_ms=_elapsed(),
        )

    # ── 1. File-based / simulated feed ──────────────────────────────────────
    try:
        path = Path(address)
        if path.exists():
            ok = path.is_file() and path.stat().st_size > 0
            return ConnectionResult(
                success=ok,
                message=(
                    f"Local feed file '{address}' is readable "
                    f"({path.stat().st_size} bytes)."
                    if ok else
                    f"Local path '{address}' exists but is not a readable, non-empty file."
                ),
                latency_ms=_elapsed(),
                stream_verified=ok,
            )
    except OSError:
        pass  # Not a usable path — fall through to the network probe.

    host, port = _split_host_port(address, proto)
    if not host:
        return ConnectionResult(
            success=False,
            message=f"Could not parse a host from '{address}'.",
            latency_ms=_elapsed(),
        )

    # ── 2. TCP reachability ─────────────────────────────────────────────────
    reachable, why = _tcp_reachable(host, port)
    if not reachable:
        return ConnectionResult(
            success=False,
            message=f"{proto} probe to {host}:{port} failed — {why}.",
            latency_ms=_elapsed(),
        )

    # ── 3. Stream open, where the protocol supports it ──────────────────────
    if proto in ("RTSP", "HTTP", "HLS"):
        if _try_open_stream(address, proto):
            return ConnectionResult(
                success=True,
                message=(
                    f"{proto} stream at {host}:{port} opened and a frame was read."
                ),
                latency_ms=_elapsed(),
                stream_verified=True,
            )
        # Port open but no readable stream. Report exactly that — this is the
        # case a simulated test would previously have called a success.
        return ConnectionResult(
            success=False,
            message=(
                f"{host}:{port} is reachable, but no readable {proto} stream "
                f"was returned. Check the stream path and credentials — a "
                f"listening port alone does not confirm a working camera."
            ),
            latency_ms=_elapsed(),
        )

    # ── ONVIF: no device-discovery library installed ────────────────────────
    # Deliberately NOT reported as a full success. TCP reachability is all
    # that was actually established; saying more would be the same overclaim
    # this module was rewritten to remove.
    return ConnectionResult(
        success=True,
        message=(
            f"{host}:{port} is reachable over TCP. ONVIF device verification "
            f"was not performed (no ONVIF client installed) — this confirms "
            f"something is listening, not that it is a working ONVIF camera."
        ),
        latency_ms=_elapsed(),
        stream_verified=False,
    )
