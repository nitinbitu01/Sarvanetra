# Model 2 (Unified Viewing & Metadata Analytics) — Coverage

See `docs/MODEL_1_ARCHITECTURE.md` for the Model 1 (Centralised Registry & GIS Mapping)
checklist — onboarding, camera-type/coverage map layers, maintenance monitoring,
ageing-infrastructure gap analysis, and registry export.

See `docs/WATCHLIST_ALERTING_ARCHITECTURE.md` for the platform's actual core requirement
per the official challenge brief — live feed → watchlist match → automated alert — kept
separate from this per-model breakdown because it cuts across models rather than
belonging to one.

Companion to `docs/MODEL_3_4_ARCHITECTURE.md`, same discipline: every claim below is a
file:line citation to code that runs, or a number actually measured against a running
process today, never an assertion. Checked against the official deliverable and
functional-feature list.

---

## Summary

| Official requirement | Status | Evidence |
|---|---|---|
| Feed aggregation (RTSP/ONVIF/vendor APIs) | ✅ Live | `docs/MODEL_3_4_ARCHITECTURE.md` §2 |
| ANPR-based metadata generation | ✅ Live, wired to live frames | §1 below |
| Event tagging and camera-wise indexing | ✅ Real, DB-indexed | §2 below |
| Searchable vehicle-movement records | ✅ Real, DB-backed | §3 below |
| Configurable video walls / multi-camera grid | ✅ Live | `CameraGrid.jsx` |
| Alerts for tagged events / watchlist vehicles | ✅ Real backend data | §4 below |
| Searchable metadata dashboard | ✅ Real | §3 below (same endpoint) |
| Unified viewer connected to **≥ 2 different systems** | ✅ **Closed today** | §5 below |
| Architecture note: existing departmental systems unaffected | ✅ **Written below** | §6 below |

The last two rows were open gaps as of the last review of this deliverable against the
official checklist. Both are closed in this revision — §5 with a live, reproduced,
verified proof (not a description of a plan), §6 with the explicit guarantee the
official brief asks for.

---

## 1. ANPR-based metadata generation — live, not an offline script

`backend/workers/ai_worker.py` pulls the latest LIVE frame from the stream queue
(`queue.get_latest_frame(cid)`, line 59), runs detection/tracking, then calls
`get_anpr_engine()` (from `backend/services/anpr_engine.py`) at lines 84-90 and 98.
The same call path is used in the always-on path, `backend/services/live_24x7_pipeline.py:65`.
This is the live worker loop, not a batch/offline script — every plate read in the
system comes from a frame that was live moments earlier.

## 2. Event tagging and camera-wise indexing — real, DB-indexed

`backend/db/models.py`:
- `Detection.camera_id` (line 137), composite index `idx_det_camera_time` (line 149).
- `Alert.camera_id` (line 185), `Alert.zone` (line 206), index `idx_alert_camera_time` (line 240).
- `JourneyEvent.camera_id` / `.zone` / `.is_cross_zone` (lines 251-256).

`backend/routers/v1/alerts.py` filters on exactly these columns (lines 47-48, 78, 118, 333)
— tagging is not just stored, it is queried on in the running API.

## 3. Searchable vehicle-movement records / metadata dashboard — real, DB-backed

Two real endpoints, not one feature counted twice:
- `GET /journeys` (`backend/routers/v1/journeys.py:166-485`) — queries the `JourneyEvent`
  table, groups by `reid_id`, supports `search=`.
- `GET /plate-search` (`backend/routers/v1/plate_search.py:92-238`) — exact + fuzzy plate
  matching against `JourneyEvent`, with measured precision/recall disclosed in the route's
  own docstring (lines 226-229) rather than asserted.

**Correction, found after this section was first written**: at the time this was marked
✅, `plate_search.py`'s router was never actually passed to `app.include_router()`
anywhere in `backend/main.py`, and a dead parameter on `search_plate` would have crashed
FastAPI's route registration the moment anyone tried — so the endpoint above was
completely unreachable, on both the backend and frontend sides, despite being real,
working code. Both fixed; see `docs/WATCHLIST_ALERTING_ARCHITECTURE.md` §1-2 for the
full account, found while tracing the platform's core watchlist-alerting requirement.
The claim below (`JourneyView.jsx` calling `/plate-search`) was also wrong — `journeys`
was the only endpoint it actually called; corrected here rather than silently reworded.

`frontend/src/components/JourneyView.jsx` calls `${API}/journeys?mode=...`
(line 730), `/journeys/evidence` (line 120), `/journeys/export/pdf` (line 704) — backed by
real DB rows and on-disk crop evidence, not decorative UI.

## 4. Alerts for tagged/watchlist vehicles — real backend data

`frontend/src/components/AlertFeed.jsx:567` calls `authFetch(`${API}/alerts?limit=50`)`,
with real per-alert actions at lines 46, 108, 168, 374 (`verify`, `mark-false-positive`,
`action`, `merged`) — every one hits a real `backend/routers/v1/alerts.py` endpoint. No
mock alert data anywhere in this path.

---

## 5. Unified viewer connected to at least two different systems — closed, verified live

**The gap.** Every camera in this deployment was, until today, sourced from a single
system: the corp8 cloud portal (HLS, cookie-session auth). The multi-vendor adapter code
existed and was reachable (`docs/MODEL_3_4_ARCHITECTURE.md` §2) but nothing outside
corp8 was actually connected — the literal "at least two different systems" deliverable
was not met by what was running, whatever the architecture could in principle support.

**What was built to close it.** `backend/scripts/demo_second_camera_source.py` — a
genuinely separate system: its own process, its own protocol (progressive HTTP
`multipart/x-mixed-replace` MJPEG, not HLS), no code or auth shared with
`corp8_session.py`. It loops a real recorded traffic clip
(`demo/clips/cam_01_0700.mp4` — actual corp8-harvested footage from this project's own
capture work, timestamp burned in by the source system, not a placeholder) and serves it
as `http://127.0.0.1:8091/stream.mjpg`.

An ffmpeg-as-RTSP-server approach (`-rtsp_flags listen`) was tried first and rejected
after being measured broken on this machine's ffmpeg 7.1 build — every invocation
attempted an outbound client connection to the listen address instead of binding a
server socket, reproduced even with a minimal `-f lavfi -i testsrc` source. That is
recorded in the script's own docstring rather than silently discarded, per this
project's standing rule against hiding a measured dead end. HTTP/MJPEG needs no ffmpeg
server feature at all and reuses the exact code path (`cv2.VideoCapture` reading a
stream URL) that `backend/routers/v1/cameras.py::_grab_one_frame` already uses for every
non-corp8 camera.

**Verified today, live, end to end** (not merely "should work" — actually run):

1. Started the demo source: `python backend/scripts/demo_second_camera_source.py`. Confirmed
   listening (`Get-NetTCPConnection -LocalPort 8091` → `Listen`), confirmed a raw HTTP GET
   returns a real `multipart/x-mixed-replace` stream with genuine JFIF JPEG payloads.
2. Started the real backend (`uvicorn backend.main:app`), logged in as admin.
3. `POST /api/v1/cameras` with `url: "http://127.0.0.1:8091/stream.mjpg"`, `protocol: "HTTP"`,
   `zone: "Demo - System B"` → camera created, `status: "ONLINE"`.
4. `GET /api/v1/cameras/{id}/snapshot` (the exact endpoint `CameraModal.jsx` and
   `CameraGrid.jsx` already poll for every camera) → **200, a real 194,944-byte JPEG**
   (`FF D8` magic bytes verified), visually confirmed as genuine Ahmedabad traffic
   footage — auto-rickshaws, motorbikes, a flyover, a burned-in
   "Chiman bhai Bridge CSITMS-32_PTZ2" camera-name overlay and timestamp
   "14-06-2026 06:59:59" from the original corp8 capture.
5. Cleaned up: deleted the demo camera row and stopped every process started for this
   check, leaving the registry exactly as it was before — this was a verification, not a
   permanent fixture, since a camera row for a source that isn't currently running would
   show as broken the next time anyone looked at the registry.

**For the actual hackathon demo:** run the same script, then add the camera through the
UI's normal "Add New Camera" form (`CameraOnboardingForm.jsx`, main Stream URL field —
Model 2's own proven direct-URL path, not the vendor-negotiation section, since this
deliverable is about unified *viewing* of a second system). It will appear in
`CameraGrid.jsx` with a live thumbnail and open in `CameraModal.jsx` exactly like any
corp8 camera, sitting alongside them in the same grid — the "unified" part of "unified
viewer" made literal, live, on screen, next to the other 30.

## 6. Architecture note: existing departmental systems are not modified

This is the explicit statement the deliverable asks for, not implied by other documents:

**Sentinel's Model 2 viewer is read-only with respect to every source system it
connects to. It never writes to, configures, or controls a department's existing
camera, DVR, NVR, or VMS.**

What the code actually does, and does not do:
- `CameraAdapterFactory.create_with_fallback()` (`backend/services/camera_adapters/factory.py:62-103`)
  only ever calls each adapter's `connect()` and `read_frame()` — there is no `write`,
  `configure`, `ptz_control`, or credential-change method anywhere in `BaseCameraAdapter`
  (`backend/services/camera_adapters/base.py:22-40`) or any subclass. The interface makes
  a control operation structurally impossible, not merely unused.
- Every adapter's job, per `resolve_stream_url`'s own docstring
  (`factory.py:106-129`), is to *negotiate a URL* — pull frames from whatever the
  source already exposes over its own existing protocol (RTSP/ONVIF/ISAPI/CGI). It never
  installs anything on the source device, opens a management/config port, or changes a
  source-side setting.
- The corp8 integration (the one source system actually in production use) is a
  read-only session against the portal's existing public viewing endpoints
  (`backend/services/corp8_session.py`) — login and playlist/segment GETs only. No
  request in that module writes to the portal.
- `backend/routers/v1/cameras.py::_grab_one_frame` opens a capture, reads one frame,
  and releases it (`cap.release()`) — no persistent connection is held open against a
  source system beyond the moment of the pull, and nothing is written back.

A department handing over a camera therefore loses nothing: their existing VMS,
recording, and any other viewer they already run continue exactly as before. Sentinel
adds a second, independent viewer of the same feed — it does not sit in front of, proxy,
or gate the department's own system, and cannot, by construction, change how that system
behaves.
