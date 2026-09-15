# Model 3 (VMS Federation & Middleware) and Model 4 (Central VMS) — Coverage and Scale

See `docs/MODEL_2_ARCHITECTURE.md` for the Model 2 (Unified Viewing & Metadata
Analytics) checklist — ANPR, event tagging, vehicle-movement search, the
now-closed "≥ 2 different systems" deliverable, and the non-interference
architecture note.

Companion to `docs/ARCHITECTURE_CORRECTIONS.md` and `docs/MACRO_TRAFFIC_IMPLEMENTATION_PLAN.md`: same
discipline — every claim below is either a file:line citation to code that runs, or a number already
measured elsewhere in this repo, or is explicitly labeled **roadmap** (designed, not built). Nothing
here is asserted without one of those three labels.

---

## Summary

| Claim | Status | Evidence |
|---|---|---|
| Model 1 (Registry/GIS) — mandatory | **Built, live** | `backend/routers/v1/cameras.py` — bulk/CSV onboarding, gap-analysis, health monitoring, audit search |
| Model 2 (Unified Viewing, direct connection) | **Built, live** | corp8 HLS path: `CameraModal.jsx`, `CameraGrid.jsx` → `GET /cameras/{id}/snapshot` |
| Model 3 (VMS Federation & Middleware) | **Built, live for negotiation; HLS path field-verified, other adapters unit-tested only** | `backend/services/camera_adapters/factory.py` |
| Model 4 (Central VMS, statewide) | **Architecture defined and load-bearing choices justified with numbers; not deployed at statewide scale** | this document, §3 |

The official FAQ states Model 2 "connects directly... unlike Model 3" and Model 3 "does NOT connect
directly, unlike Model 2" — for **the same camera** these are exclusive by definition. This platform
is not choosing one; it runs both concurrently, one per camera, decided by what that camera's source
system exposes (§2). The FAQ explicitly permits a hybrid of the four models — this is that hybrid,
not a hedge.

---

## 1. Model 2 vs Model 3 — why both, and how they coexist without contradiction

A single camera row can only be integrated one way at a time — either the platform already has a
working stream URL and pulls it directly (Model 2), or it doesn't, and something has to negotiate one
from vendor + IP + credentials before Model 2's direct pull can even start (Model 3). The two models
are exclusive **per camera**, not exclusive **per platform**.

Concretely, in `backend/main.py` (~line 210-248), every configured camera is checked for an explicit
`url`:

- **Has a URL already** (this deployment's 30 corp8 cameras — the portal hands back a ready HLS
  address at login): skip negotiation, connect directly. This is Model 2.
- **No URL, only vendor + IP + credentials** (the shape a real department's existing CCTV estate
  arrives in — nobody hands over a stream URL, they hand over a DVR's IP and a login): call
  `resolve_stream_url()`, which walks the adapter fallback chain in §2 and hands back whatever
  protocol actually answered. This is Model 3. Only once negotiation succeeds does that camera also
  get pulled directly like any other — Model 3's job is strictly to *produce* the URL Model 2 then
  uses, not to proxy traffic itself.

This is also why the boundary in `main.py:232-234` is a real code comment, not documentation written
after the fact: before this wiring existed, a camera with no preconfigured URL was silently skipped,
and the entire adapter layer had no caller in the running system (`factory.py:109-128` — its own
docstring records this). Fixing that dead-code path is what makes the hybrid genuine rather than
theoretical — it was verified by making the previously-unreachable code reachable, not by writing
new code.

---

## 2. Model 3 — VMS Federation & Middleware: what is actually built

**Location:** `backend/services/camera_adapters/` — `factory.py` (orchestrator),
`base.py` (`BaseCameraAdapter` contract), and one file per vendor:
`hikvision_adapter.py`, `dahua_adapter.py`, `onvif_adapter.py`, `rtsp_adapter.py`, `hls_adapter.py`.

**How it works** (`factory.py:24-103`, `CameraAdapterFactory`): each vendor maps to an ordered
fallback chain of adapter types (`FALLBACK_CHAINS`, `factory.py:38-49`) — e.g. `hikvision` tries
its own ISAPI adapter first, then falls back to generic ONVIF, then raw RTSP, then HLS. Registration
is a static dict (`ADAPTER_REGISTRY`, `factory.py:51-57`) — adding a sixth vendor means writing a
`BaseCameraAdapter` subclass and adding two dict entries, not a dynamic plugin loader. That is a
scope decision documented here rather than concealed: it is enough to onboard the vendors below
without a per-camera code change, and no requirement in the official brief asks for hot-loadable
adapters.

`create_with_fallback()` (`factory.py:62-103`) tries each adapter in the chain in turn with a 5s
connect timeout, remembers which one worked per camera (`_in_memory_preferences`) so the next
negotiation skips straight to it, and — critically — **never raises**: a camera whose entire chain
fails returns an unconnected `RTSPAdapter` rather than an exception, so one bad camera cannot stop
the other 25 departments' cameras from starting (`factory.py:126-128`).

**Exposed to the API surface**, not just internal:
- `GET /adapters/supported` (`cameras.py:818-827`) — returns the live `FALLBACK_CHAINS` and the
  protocol list (RTSP, ONVIF, ISAPI/Hikvision, CGI/Dahua, HLS, HTTP-MJPEG) directly from the
  factory object, so this document and the running API cannot drift out of sync with each other.
- `POST /discover/onvif` (`cameras.py:830-835`) — WS-Discovery probe for ONVIF Profile S devices on
  the local network, for the "auto-discover cameras" onboarding flow the official brief describes.

**Verification status — stated precisely, not rounded up:**
- The **HLS adapter path** is the one this deployment actually runs live, against the corp8 portal's
  30 cameras, end to end (session pooling, rate-limit handling, live frame grabs — see §3). This is
  field-verified, not a simulation.
- The **RTSP adapter** and the fallback-chain source-type detection logic have unit-test coverage
  (`tests/test_live_pipeline_sre_qa.py:306-321`), but not yet a live RTSP-server verification —
  `-rtsp_flags listen` (ffmpeg-as-RTSP-server, the natural way to stand up a local RTSP source for
  this) was tried and measured broken on this machine's ffmpeg 7.1 build: every invocation attempted
  an outbound client connect instead of binding a listen socket. `docs/MODEL_2_ARCHITECTURE.md` §5
  closes the adjacent "≥ 2 systems" deliverable via HTTP/MJPEG instead, which needs no ffmpeg server
  feature — that does not, on its own, verify the RTSP *adapter* path specifically, so RTSP stays
  unit-tested-only until a real RTSP source (a working server build, or real hardware) is available.
- The **Hikvision, Dahua, and ONVIF adapters** are implemented against each vendor's published
  protocol (ISAPI, CGI, WS-Discovery/ONVIF Profile S respectively) and are wired into the same live
  negotiation path as HLS — but this environment has no such hardware to connect to, so they have
  not been exercised against a real device this session. Stating this plainly is the same standard
  applied everywhere else in this repo's docs (`ANPR_FEASIBILITY.md`, `ARCHITECTURE_CORRECTIONS.md`):
  a claim without a measurement behind it is not made.

**What this means for the hackathon's live test case:** if the organizers' ~50 simulated feeds are
handed over as vendor + IP + credentials (the realistic shape for a government camera estate) rather
than ready-made URLs, this negotiation path is what onboards them — the same bulk-onboarding endpoint
built for Model 1 (`POST /cameras/bulk`, `cameras.py:469-478`) accepts a `vendor` field per row for
exactly this. If they are handed over as direct stream URLs instead, Model 2's path is used and Model
3 is not invoked for those cameras at all — which is correct behavior, not a gap.

---

## 3. Model 4 — Central VMS: current measured scale, and what statewide requires

This section separates three things that are easy to blur under deadline pressure: what is running
right now, the concrete number that will eventually break it, and what changes at that point. All
three come from load-bearing engineering already done in this repo, not from this document.

### 3.1 What is running now, measured

| Component | Measured value | Source |
|---|---|---|
| corp8 session pool | 6 concurrent sessions (`SESSION_POOL_SIZE`) | `corp8_session.py:66` |
| Per-session request budget | ~45 requests before HTTP 429; cleared only by re-login, not by waiting | `corp8_session.py:182-191` |
| Session lifetime | 30 minutes (`SESSION_MAX_AGE_SEC`) | `corp8_session.py:54` |
| Login rate floor | 4s minimum between logins (`LOGIN_MIN_INTERVAL_SEC`) | `corp8_session.py:60` |
| Snapshot grab latency (per camera, cold) | 7–8.5s end to end (FFmpeg + HLS decrypt) | `cameras.py:850-862` |
| Snapshot cache TTL | 20s server-side per camera | `cameras.py:864` |
| corp8 health-check batch interval | 300s | `camera_heartbeat.py:90-91` |
| corp8 circuit-breaker cooldown (on HTTP 403) | 30 minutes | `camera_heartbeat.py:94` |
| Fleet size this deployment | 30 corp8 cameras + 5-adapter multi-vendor layer (§2) | — |
| Datastore | SQLite, WAL mode, single node | `db.py`; confirmed in `MACRO_TRAFFIC_IMPLEMENTATION_PLAN.md:47,184` |

The session-pool and rate-limit numbers are not theoretical — they were measured by triggering the
actual 429/403 lockouts against the live corp8 portal earlier in this project and root-causing the
two distinct failure modes (§ engineering log in `README.md`), then designing the 6-slot pool and
staggered-polling behavior in `CameraGrid.jsx` specifically to stay under them.

### 3.2 The number that breaks the current design, and where it comes from

`MACRO_TRAFFIC_IMPLEMENTATION_PLAN.md:54-58` already computed this for the detection pipeline: at
32 cameras and 5 fps per-detection logging, raw writes reach **69M rows/day**, which it calls
"untenable on SQLite." That document's own fix — log one row per completed track instead of one row
per detection — brings it to ~460K rows/day, which is why the *current* 30-ish-camera deployment is
fine on SQLite. The same document sets the explicit threshold for when this stops being true:

> **"Revisit at ~200 cameras, or the first multi-node deployment."** (`MACRO_TRAFFIC_IMPLEMENTATION_PLAN.md:52`)

That threshold is this document's Model 4 ceiling, not a new estimate. A 26-department statewide
rollout is, by definition, past "the first multi-node deployment" — Central VMS at that scope is
out of range for the present single-node SQLite design on capacity grounds alone, independent of any
other consideration.

### 3.3 What changes at statewide scale — roadmap, explicitly not built

Labeled roadmap because none of it is deployed in this repo; each item is the direct, mechanical
consequence of one constraint above, not a wish list:

- **Datastore: SQLite (WAL) → Postgres/TimescaleDB.** Forced by §3.2's row-rate ceiling, not by a
  general preference for "bigger" infrastructure — the current choice is the *correct* one below
  that ceiling (`MACRO_TRAFFIC_IMPLEMENTATION_PLAN.md:47-52` argues this explicitly: Kafka/Flink are
  deferred for the same reason, over-engineering below the threshold that justifies them).
- **corp8-style session pooling → one connector per department VMS.** The 6-session pool in
  §3.1 is sized for one portal account under this hackathon's constraints. A 26-department rollout
  means 26 independent source systems (or more, per §2's Model 3 multi-vendor path), each with its
  own auth and rate characteristics — the pooling *pattern* generalizes (it is already
  vendor-agnostic in `CameraAdapterFactory`), but a shared 6-slot pool sized for one portal does not.
- **Static `ADAPTER_REGISTRY` → dynamic vendor plugin registration.** Fine at 5 hand-written
  adapters (§2); a statewide rollout onboarding departments over time benefits from registering a
  new vendor without a code deploy. The interface (`BaseCameraAdapter`) does not need to change for
  this — only how subclasses get registered.
- **Single-node health monitoring → horizontally sharded heartbeat workers.** The 300s batch
  interval and 30-minute circuit-breaker cooldown (§3.1) are sized for checking dozens of cameras
  from one process; thousands of cameras across 26 departments need the check work partitioned
  across workers, not a single sequential batch.

None of these four are a redesign — each is the existing pattern (adapter interface, session
pooling, batched health checks, row-per-track logging) scaled past the point where a single-node
implementation of that same pattern stops being sufficient. That is what makes the statewide
migration credible: it is "run the same design on more nodes with a bigger datastore," not
"replace the design."

### 3.4 Central-VMS fleet management already built (Model 1 overlap, Model 4 depends on it)

Central VMS at any scale needs to onboard, monitor, and audit its fleet — that tooling is not new
scope, it was built for Model 1 and Model 4 needs the same operations at higher volume, not
different ones:

- `POST /cameras/bulk`, `POST /cameras/bulk/csv` (`cameras.py:469-552`) — batch onboarding with
  per-row validation and isolated rollback, so one bad row in a 500-camera CSV does not sink the
  batch.
- `GET /cameras/gap-analysis` (`cameras.py:623-655`) — missing GPS, missing department, offline,
  never-health-checked, stale-health, per zone — the fleet-health report a central operator needs,
  scoped per-department for non-admin callers.
- `camera_heartbeat.py` — real health checks (not the previous `DEMO_MODE` no-op), fail-closed:
  a camera absent from a batch result is left untouched rather than wrongly marked healthy.

---

## 4. What to actually show on hackathon day

Given the "no mockups" constraint, the honest split is:

- **Demo live:** Model 1 registry/onboarding/gap-analysis, Model 2 direct viewing (corp8 feeds,
  real snapshots in `CameraModal`/`CameraGrid`), and Model 3's negotiation path — if the organizers'
  test cameras arrive as vendor+IP+credentials, `GET /adapters/supported` and a bulk-onboarding call
  showing the fallback chain actually resolving a URL is a genuine, unscripted demonstration, not a
  mockup.
- **Present as architecture, not demo:** Model 4 statewide scale (§3.3) and Hikvision/Dahua/ONVIF
  against real hardware (§2) — both are honestly labeled here as designed-and-justified rather than
  measured, and should be presented that way in the Technical Proposal/HLD rather than implied to be
  running, which is exactly the gap organizers are positioned to catch.
