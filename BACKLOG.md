# Sentinel IQ — Backlog

Things deliberately not built, recorded so they are decisions rather than
oversights. Each entry says what is missing and what breaks because of it.

---

## Post-Demo Production Backlog (Day 16)

### HLS Streaming
- [ ] Replace cookie auth (Option A) with JWT in hls.js xhrSetup (Option B)
      for stateless multi-server deployments
- [ ] Replace ffmpeg direct spawn with MediaMTX (formerly rtsp-simple-server)
      for production RTSP ingestion, re-streaming, and access control
- [ ] Move HLS segment delivery to a CDN or nginx with X-Accel-Redirect —
      removes Python from the segment serving hot path
- [ ] Replace in-process ffmpeg_processes dict with Redis-backed process registry
      for multi-worker deployments (uvicorn --workers > 1)
- [ ] Add per-camera stream health endpoint: GET /stream/{camera_id}/health
      returns { running, pid, uptime_seconds, segment_count }

### Feedback Loop
- [ ] Replace in-process _feedback_rate dict with Redis for multi-worker rate limiting
- [ ] Add officer.role column to officers table for proper admin guard
      (currently falls back to officer_id = 1 for demo)
- [ ] Add GET /alerts/{alert_id}/feedback/history for audit trail per alert
- [ ] Add weekly false alarm report: scheduled job → email/Slack summary
- [ ] Dashboard: zone heat map showing false alarm rates by location_label

### Auth
- [ ] Migrate HLS auth from cookie session to signed JWT query param for Safari
      (Safari native HLS cannot set custom headers — query param is the standard approach)

---

## Push Notifications (Day 15)

### P0 — before any real deployment

- [ ] **Replace the in-memory rate limiter with Redis sorted sets.**
      `backend/push/service.py::_rate_store` is a process-local dict. Two
      consequences: the window resets on restart (a burst resumes after a
      redeploy), and with multiple workers each process allows the full limit
      independently — so the effective cap is `limit × workers`, not `limit`.

- [ ] **Move push fan-out off the alert pipeline.**
      `_send_push()` is synchronous: N officers means N sequential HTTPS calls
      to the push relay, blocking the detector that fired the alert for
      `N × latency`. Fine for a handful of officers; not fine for a shift
      roster. Needs a task queue.

- [ ] **VAPID key rotation runbook.**
      Rotation invalidates every existing subscription — the browser signed
      each one against the old public key. There is no server-side migration;
      each device must re-subscribe. Sends start returning 401/403, which the
      delivery log captures, but the officers' phones simply go quiet until
      they re-enable. Rotation is a user-facing event and needs a re-subscribe
      prompt (`GET /push/resubscribe-required` + a PWA check on open).

- [ ] **Rotate the committed dev VAPID keypair.**
      `backend/core/config.py` ships a working keypair as the default so the
      feature runs out of the box. It is in version control, therefore public,
      therefore compromised. Generate a real one
      (`python -m backend.scripts.gen_vapid_keys`) into `.env` before any
      deployment.

### P1 — before multi-officer production use

- [ ] Per-officer preferences: severity threshold, quiet hours, per-camera
      opt-in. Today every CRITICAL alert pushes to the assigned officer with
      no way to tune it.
- [ ] Push delivery dashboard over `push_delivery_log` — delivery rate per
      officer, per device, over time. The data is already being written; only
      the read surface is missing.
- [ ] Direct FCM/APNs integration. Web Push routes through the browser
      vendor's relay, which is the least reliable hop in the chain.

### P2 — scaling

- [ ] Shard subscription fan-out if the roster exceeds ~1000 officers.
- [ ] Batch the delivery-log writes; today it is one INSERT + COMMIT per send.

---

## On-device verification NOT covered by automated tests

`verify_day15.py` covers everything on the server side of the push relay. The
following require a physical handset, an HTTPS tunnel and a live push service,
and must be run by hand before any demo. **A green `verify_day15.py` does not
mean the phone buzzed.**

```
Setup
  1. Start backend and frontend.
  2. ngrok http 8000   (backend)     ngrok http 5173   (frontend)
     A LAN IP (192.168.x.x) is NOT a secure context — Service Workers and
     Web Push will silently fail to register. It must be the HTTPS URL.
  3. On the phone, open the HTTPS frontend URL.
  4. iOS: Share → Add to Home Screen → open from the icon.
     Push on iOS only works from the installed PWA, never a Safari tab.
  5. Tap "Enable Alerts", grant permission.
  6. Confirm a push_subscriptions row exists with the right officer_id.
```

| # | Test | Expected |
|---|------|----------|
| 2 | Background delivery — close the PWA, fire a CRITICAL alert | Phone vibrates, OS notification within ~5s |
| 3 | Notification tap | Opens `/?alert=<id>` directly to the full-screen card, not the dashboard |
| 4 | Hindi TTS | Speaks on card render; on iOS, tap the card once if it was gesture-blocked |
| 6 | Foreground push | Card appears with no tap AND no OS notification stacked behind it |
| 10 | WiFi kill | Fire alert, kill WiFi 1s later, restore within 30s → push still arrives (OS-level, TTL not expired) |

---

## Day 17 is missing (Day 16 has since landed)

Day 18's brief described itself as "integration, not new features" and listed
fifteen components to assemble. A pre-flight inventory at the time found none
of them, so Day 18 was built against what existed.

**Day 16 has since arrived** — `backend/stream/manager.py`, `HlsPlayer.jsx`
and `FeedbackButtons.jsx` are present, and `verify_day16.py` /
`verify_day16_2.py` pass 19/19 each. The Day 16 items below are kept only to
record what Day 18's ControlRoom does *not* yet consume: `CameraModal` and
`CameraGrid` were written before the stream manager existed and still do not
use it. Wiring them to the real HLS pipeline is now possible and is the single
highest-value cleanup left in the dashboard.

**Day 17 was never built** — no `OfflineHeader`, no `DemoControls`.

Day 18 was therefore built against what is actually here. The following
remain genuinely absent — not deferred polish, but whole features the
control-room layout has visible holes for:

### Day 16 — built, but not consumed by the Day 18 dashboard
- [ ] **Point `CameraModal` / `CameraGrid` at the real stream manager.**
      `backend/stream/manager.py` and `HlsPlayer.jsx` now exist and pass their
      verifiers, but the control-room components predate them.
      *Current state:* `CameraModal` embeds a third-party iframe
      (`live.corp8.cloud`) whose channel is derived from the camera name by
      regex, falling back to `((id - 1) % 30) + 1` — so the feed shown is not
      bound to the camera record it is labelled with. `CameraGrid` tiles show
      heartbeat status, not thumbnails.
      *Still claimed but not performed:* the "SENTINEL AI TRACKING ACTIVE —
      YOLOv8 + BoT-SORT · OSNet ReID · ANPR" overlay (no inference runs on
      that iframe), and the hardcoded `~24 ms` latency / `~1.9 Mbps` bitrate
      readouts. A null GPS renders as `23.0489, 72.5714` (Ahmedabad) rather
      than as unknown. These are known and deliberate; they are listed here so
      nobody later mistakes them for measurements.
      *Removed:* a "Capture Frame" button that popped "✓ Forensic Snapshot
      Captured & Hashed" while doing neither. It could not be made real —
      the feed is cross-origin, so the browser will not expose its pixels to
      canvas. Absence is pinned by
      `frontend/src/components/__tests__/CameraModalClaims.test.jsx`.
- [ ] `GET /stream/{camera_id}/thumbnail.jpg` — depends on HLS segments
      existing, so it cannot be built before the pipeline.
- [ ] **FeedbackButtons / SystemLearningBanner.** No false-alarm feedback
      loop, so nothing feeds detector tuning from operator judgement.

### Day 17 — offline resilience

- [x] **Offline ACK queue + sync-on-reconnect.** Built:
      `frontend/src/utils/ackQueue.js` (framework-agnostic outbox — enqueue,
      idempotent-by-alertId dedup, exponential backoff, pub/sub) +
      `frontend/src/hooks/useAckQueue.js` (React glue) wired into both real
      ACK surfaces, `AlertDetailCard.jsx` (officer/phone) and
      `RoutingBadge.jsx` (control room). A dropped connection or 5xx now
      queues the tap and shows "⏳ Queued — will send when back online"
      (disabled, so no duplicate tap) instead of silently reverting to a
      fresh button; a live 409 (server answers "no" over a working
      connection) is deliberately NOT queued — that's a real-time rejection,
      not a dropped-connection case, and retrying it would be wrong. No
      backend changes were needed: `POST /alerts/{id}/ack`
      (`backend/routers/v1/routing.py`) was already idempotent by
      construction (conditional UPDATE + rowcount guard), so replaying a
      queued ack any number of times is already safe.

      Persistence is best-effort IndexedDB, feature-detected — without it
      (unsupported browser, or a killed-and-reopened tab) the queue still
      works correctly for as long as the page stays open; it just doesn't
      survive a full kill. Retries trigger on `online`, `visibilitychange`,
      WebSocket reconnect (`useConnectionState() === 'CONNECTED'`), and a 5s
      backstop tick.

      **Deliberately not built: Background Sync API registration.** It only
      helps the fully-killed-tab case, and iOS Safari has zero support for
      it — this app's own Day 15 push work already treats iOS as a first-
      class target, so Chrome/Android-only partial coverage was judged a
      worse trade than a smaller mechanism verified to work everywhere the
      tab is open. The killed-tab case is a real, narrower remaining gap;
      it's smaller than what existed before this (nothing was queued at
      all), and is recorded here rather than silently absent.

      **Not wired: `OfficerApp.jsx`'s "ACK Incident" button.** That screen is
      a mockup — `activeAlerts` is hardcoded sample data, not a real alert
      with a real id, so queuing against it would queue forever and never
      resolve. Flagged in place with a comment; `AlertDetailCard.jsx` is the
      real officer-facing surface.

      Tests: `frontend/src/utils/__tests__/ackQueue.test.js` (13),
      `frontend/src/components/__tests__/AlertDetailCard.test.jsx` (6),
      `frontend/src/components/__tests__/RoutingBadge.test.jsx` (3) — 22
      total, all passing, each verified to actually fail on the mutation it
      guards (including a real bug the component test caught during
      development: `flush()` originally notified subscribers twice in the
      same synchronous stretch for a synced/failed outcome — once with the
      item present, once with it pruned — which React batches into one
      render, so the intermediate terminal state was invisible to any
      subscriber. Fixed by separating notification from removal: a
      component sees the terminal status and calls `dismiss()` itself; a
      15s safety-net timer prunes anything nothing was watching).

- [ ] **OfflineHeader / connection-loss UX.** `WebSocketContext` exposes
      `connectionState` and `DashboardHeader` renders it as a dot, but there
      is no full-width offline bar and no `[CACHED]` marking on stale rows.
- [ ] **CameraOfflineBannerList**, **DemoControls**, and WebSocket
      ping/pong heartbeat.

### Consequence for the Day 18 acceptance checklist
The queued-ACK-sync half of Test **6** is now covered (see above). The
`[CACHED]` / offline-header half of Test 6, and the Day 16/17 rows of Test
**9**, are not "untested" — they are **unbuildable** until the items above
exist. Tests 3, 7 and 8 need a running browser; the DOM-cap half of Test 7 is
covered by `frontend/src/components/__tests__/AlertFeedCap.test.jsx`.

---

## Control Room (Day 18)

- [ ] **Virtualize the alert feed.** `@tanstack/react-virtual` is installed
      but unused. The 200-row cap bounds DOM growth, which was the actual
      failure mode; virtualization would additionally cut the rendered set to
      the visible window. Deferred because `AlertRow` has variable height with
      expandable sub-panels, so it needs dynamic measurement — a retrofit with
      real risk of breaking a component that carries every Day 10–15 feature.
- [ ] **Implement search, or remove the box.** `DashboardHeader`'s input is
      inert and labelled as such. It needs a backend endpoint spanning plates,
      cameras and person IDs. A client-side filter over whatever happens to be
      loaded would return confident partial results and read as working.
- [ ] **Replace the ROI figures or delete the panel.** `ROIPanel`'s numbers
      are illustrative and carry a caption saying so, enforced by a test. The
      inputs a real ROI claim needs — officer hours, false-alarm rates,
      response times — are not measured anywhere in this system.

---

## Product-readiness blockers (unchanged, carried forward)

See `docs/PRODUCTION_READINESS_OPEN_ITEMS.md`. All four remain **UNASSIGNED**:
data retention ownership, formal approval for face recognition against a
police watchlist, bias/disparate-impact testing, and a mandated
human-in-the-loop procedure.

The face-match threshold is still unvalidated against real data. Auto-dispatch
is withheld for face matches by the interlock in
`backend/services/threshold_validation.py`; **push is not withheld** — an
unvalidated face match still notifies, because a human needs to look at it.
The payload carries `assigned: false` so the phone does not imply a dispatch
that did not happen.
