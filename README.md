# Sentinel Gujarat — CCTV Intelligence Platform

> **Hackathon build log** | Day 8 of N
> **Day 1:** ✅ YOLOv8 + BoT-SORT detection pipeline
> **Day 2:** ✅ FastAPI backend + WebSocket + React dashboard
> **Day 3:** ✅ SQLite schema + FAISS ReID index + Watchlist
> **Day 4:** ✅ ANPR pipeline — vehicle detection + plate OCR + stolen-vehicle alert
> **Day 5:** ✅ Auth (JWT), audit log, soft-delete, evidence hashing
> **Day 6:** ✅ Person ReID engine — OSNet-IBN + FAISS, review queue, journeys
> **Day 7:** ✅ Face watchlist matching (InsightFace) + camera heartbeat monitoring
> **Day 8:** ✅ Behavior engine (loitering + crowd anomaly) hardened for production —
>            Redis-backed shared state, real-world camera calibration, alert
>            lifecycle, observability, measured eval harness

---

## Day 8: Behavior Engine Hardening (Loitering + Crowd Anomaly)

Takes the loitering/crowd detectors from a single-process demo to something
closer to production, WITHOUT changing the core detection math (centroid/
radius logic for loitering, rolling-baseline logic for crowd) — this pass
changes WHERE state lives, HOW thresholds are derived, and WHAT happens
after an alert fires.

### New files

| File | Purpose |
|------|---------|
| `backend/services/state.py` | Redis-backed shared state wrapper, skip-and-log on outage |
| `backend/services/loitering_detector.py` | Per-track rolling-window loitering detection |
| `backend/services/crowd_detector.py` | Per-camera rolling-baseline crowd anomaly detection |
| `backend/services/camera_calibration.py` | Per-camera px-per-meter lookup + uncalibrated fallback |
| `backend/services/metrics.py` | In-memory counters + latency tracking for `/debug/stats` |
| `backend/scripts/calibrate_camera.py` | One-time CLI: two known-distance points → px_per_meter |
| `backend/routers/v1/debug.py` | `GET /api/v1/debug/stats` |
| `tests/behavior_eval/` | Synthetic labeled clips + `eval_runner.py` — measured precision/recall/FPR |

### 1. Shared state — Redis backend

Loitering windows (`loiter:{camera_id}:{track_id}`) and crowd baselines
(`crowd:baseline:{camera_id}`) live in Redis sorted sets (ZADD/
ZREMRANGEBYSCORE), not in-process deques — they survive a backend restart
and would be correct if the pipeline ever scaled across multiple
camera-worker processes. Debounce state (`behavior:active:LOITERING:...`)
and crowd cooldown (`crowd:cooldown:{camera_id}`) are simple TTL'd keys —
no manual state-machine bookkeeping, they expire naturally when the
condition stops being met.

**Identity key**: loitering is keyed on `{camera_id}:{track_id}` (the
pipeline's own local track id), not `global_id`. `global_id` is only
populated once Day 6's ReID resolves asynchronously and isn't guaranteed
present on every event in this build — see `loitering_detector.py`'s module
docstring for the full reasoning, including why this is actually the
semantically correct granularity for loitering specifically (continuous
presence, not cross-camera identity).

**Failure mode**: if Redis is unreachable, every detector tick is skipped
and logged (rate-limited to avoid log spam), never buffered in memory. See
`state.py`'s module docstring for why "skip" was chosen over "buffer and
resync" for this pass.

**Bug found and fixed during hardening**: the spec's literal sorted-set
member format (`"cx,cy"`) collides whenever a stationary (i.e. loitering)
person reports the same pixel position twice — `ZADD` with an existing
member overwrites its score instead of adding a new point, so the window's
earliest timestamp never gets older than the last couple of ticks and the
loitering condition never actually fires. Verified this happens (not just a
theoretical risk) before fixing it — members are now prefixed with
`video_time` to guarantee uniqueness per sample.

### 2. Camera calibration — real-world units

New `camera_calibration` table: `camera_id`, `px_per_meter`,
`calibration_method`, `calibrated_at`, `notes`. `loitering_detector.py`
converts `LOITER_RADIUS_METERS` into pixels per-camera via
`camera_calibration.py`, instead of a flat pixel constant. A camera with no
calibration row falls back to `settings.DEFAULT_PX_PER_METER`, and the
resulting alert's `metadata.calibration_method` is tagged `'uncalibrated'`
— `AlertFeed.jsx` shows a visible "⚠ uncalibrated" tag on those alerts so
they're never silently mixed with trustworthy ones.

Run the calibration CLI once per camera:
```
python -m backend.scripts.calibrate_camera --camera-id CAM-01 \
    --point1 120,400 --point2 540,410 --real-distance-meters 5.0 \
    --notes "5m mark on north crosswalk"
```
Flat px-per-meter is an approximation valid for a roughly top-down or
moderate-angle view — a wide-angle/oblique camera needs proper homography,
explicitly deferred (see `camera_calibration.py`'s docstring).

### 3. Alert lifecycle

`alerts` gets three new columns: `lifecycle_status` (OPEN | ACKNOWLEDGED |
DISMISSED | ESCALATED, defaults OPEN), `reviewed_at`, `feedback`
(TRUE_POSITIVE | FALSE_POSITIVE | null). This is deliberately additive —
Day 5/7's `status`/`false_positive_reason` columns and the
`/mark-false-positive` endpoint are untouched; `lifecycle_status` is a
separate axis that now applies to every alert type for free.

`POST /alerts/{id}/acknowledge|dismiss|escalate` pull the reviewer identity
from the authenticated JWT (`current_user`), not a client-supplied name
string — matching the precedent Day 7's `/mark-false-positive` already set,
for the same audit-integrity reason. Every transition broadcasts
`alert_status_update` over the existing dashboard WebSocket, so a second
operator's tab updates live — this is what stops two people independently
acting on the same alert.

Feedback (TRUE_POSITIVE/FALSE_POSITIVE on dismiss) is captured and
queryable but NOT fed into any auto-tuning loop yet — that's explicitly a
later pass, per the Day 8 scope.

### 4. Observability

Structured JSON logging (via the existing `backend.core.logging` — no new
logging infrastructure needed) for every detector tick:
`{camera_id, global_id, detector, video_time, decision, latency_ms}`.

`GET /api/v1/debug/stats` exposes `frames_processed_total{camera_id}`,
`alerts_fired_total{type,camera_id}`, `behavior_detector_latency_ms_avg`,
`redis_errors_total` — a plain-JSON placeholder for a real Prometheus
`/metrics` endpoint later (explicitly out of scope for this pass).

Drift check: `crowd_detector.py` logs a WARNING if a camera's rolling crowd
baseline changes by `CROWD_DRIFT_WARN_MULTIPLIER` (default 2x) between
consecutive computations — usually a sign of a degraded upstream detector,
not an actual crowd change.

### 5. Validation — measured, not eyeballed

`tests/behavior_eval/` — 7 synthetic labeled clips (loitering: true
positive, steady-walkthrough negative, enter/exit/re-enter edge case,
calibrated-camera true positive; crowd: true positive, brief-group-photo
negative, gradual-organic-growth negative) plus `eval_runner.py`, which
runs the real detector classes against each clip and reports measured
precision/recall/false-positive-rate:

```
python -m tests.behavior_eval.eval_runner
```

Clips are synthetic `(cx, cy, video_time)` / `(count, video_time)`
sequences, not real video — the detection math consumes exactly that tuple
stream regardless of how it was produced (see `eval_runner.py`'s module
docstring for why, and what a real-footage adapter would need to add).
Current baseline: **7/7 clips pass** — zero false positives on every
negative clip, zero missed detections on every positive clip, and the
calibrated clip's alert is confirmed tagged with the real
`calibration_method`, not the uncalibrated fallback.

### Explicitly out of scope for this pass

- Full homography/perspective calibration (flat px-per-meter is enough for now)
- Automated threshold retuning from feedback data (captured, not acted on)
- Prometheus/Grafana (`/debug/stats` is a placeholder for that)
- Multi-tenant access control on alert acknowledgment
- Data retention / audit-of-who-queried-what for behavior data — a
  legal/compliance decision, not an engineering one, same caveat as Day 6's
  `REID_RETENTION_DAYS` placeholder.

---

## Day 4: ANPR Pipeline

### New files

| File | Purpose |
|------|---------|
| `backend/plate_utils.py` | `normalize_plate_text()`, `correct_confusable_chars()`, `is_valid_plate()` |
| `backend/anpr.py` | OCR ensemble — EasyOCR + PaddleOCR, takes pre-loaded readers as args |
| `backend/anpr_worker.py` | OCR background thread — loads singletons once, drains job queue |
| `backend/anpr_track_aggregator.py` | Per-track scheduling, majority-vote finalization, alert writing |

### Key architecture decisions

#### 1. Single YOLOv8 pass per frame

Day 4 extends `detection.class_filter` to `[0, 2, 3, 5, 7]` (person + car + motorcycle + bus + truck). **One `model.track()` call per frame** returns all classes. `split_by_class()` in `detector.py` then routes detections to two separate `TrackStateManager` instances — one for persons, one for vehicles. Running two separate `model.track()` calls would double inference cost for no benefit.

```python
raw_tracks = detector.track_frame(frame, frame_number)  # ONE call
person_raw, vehicle_raw = split_by_class(raw_tracks, person_classes, vehicle_classes)
```

#### 2. OCR on a separate thread — never on the detection loop

EasyOCR and PaddleOCR calls take 100–300ms each. Calling them synchronously inside the detection loop would stall person track updates to the dashboard every time a vehicle is on screen. The detection loop **only enqueues a job** (non-blocking, <1µs):

```python
ocr_worker.enqueue_job(track_id, crop, frame_number, timestamp)  # returns immediately
# detection loop continues to next frame, OCR runs in background
```

Verified: with a 2s simulated OCR call, 20 detection iterations still complete in **4.01s** (purely from 200ms frame spacing).

#### 3. OCR singletons — loaded once at worker startup

Both EasyOCR and PaddleOCR models load from disk once at worker thread startup. Load time is logged at INFO level (visible in startup logs, not repeated). `read_plate()` takes the pre-loaded instances as arguments and never constructs a new reader.

#### 4. Per-track OCR scheduling (NOT global frame modulo)

Each vehicle track maintains its own `frames_since_last_read` counter, incremented when that specific track appears in a processed frame. OCR is scheduled every `anpr.read_interval_frames` track-frames. This avoids the silent bug where a track confirmed at an odd frame offset would have its scheduled reads coincidentally skipped by a global modulo.

#### 5. Finalization rules

Finalization triggers when **any** of these conditions is met:
- Track reaches `min_readings_before_finalize` (default 3) valid readings
- Track reaches `max_readings_per_track` (default 5) total OCR attempts
- Track expires (no update within `server.track_expiry_seconds`)

Finalization decision:
1. **Majority vote** among `format_valid=True` readings
2. Tie → highest-confidence valid reading wins
3. Zero valid readings → `anpr_uncertain` alert
4. **Zero OCR attempts** (vehicle too brief) → **no alert created at all** (DEBUG log only)

#### 6. Alert outcomes

| Condition | Action |
|-----------|--------|
| Exact watchlist match (active=1) | `insert_alert(type='watchlist_vehicle', severity='critical', score=7.0)` |
| Valid plate, conf ≥ 0.70, no match | Log only — ordinary vehicle, no noise in alerts table |
| Invalid format OR conf < 0.70 | `insert_alert(type='anpr_uncertain', severity=None)` |
| Zero OCR attempts (too brief) | No DB row created at all |

#### 7. Normalization contract

The same `normalize_plate_text()` is used at seed time (`seed_data.py`) and read time (`anpr_track_aggregator.py`). `plates_are_equal()` double-normalizes both sides before comparison. Verified: `'GJ05AB1234'` == `normalize_plate_text('GJ05AB1234')` → `'GJ05AB1234'`.

#### 8. Confusable-character correction

Positional correction based on Gujarat plate structure `GJ<DD><L|LL><DDDD>`:
- Digit zones: `O→0`, `I→1`, `S→5`, `B→8`
- Letter zones: `0→O`, `1→I`, `5→S`, `8→B`

Example: `GJOSAB1234` → `GJ05AB1234` (two `O`s at digit positions corrected to `0`).

---

## Day 3: Database + Watchlist + ReID Reference Set

---

## Day 3: Database + Watchlist + ReID Reference Set

### New files

| File | Purpose |
|------|---------|
| `backend/db.py` | All SQL in one place — connection factory, WAL mode, CRUD for all 6 tables |
| `backend/embedding_utils.py` | **Single** encoding convention: `encode_embedding` / `decode_embedding` / `normalize_l2` |
| `backend/faiss_index.py` | FAISS IndexFlatIP wrapper with id_map tracking, load/save, search |
| `backend/seed_data.py` | Idempotent seeding: schema, cameras, watchlist, 15 FAISS vectors |
| `output/sentinel.db` | SQLite database (WAL mode, created by seed_data.py) |
| `output/faiss/reid_reference.index` | FAISS index file |
| `output/faiss/id_map.json` | Parallel metadata for each FAISS vector |
| `data/prep/*.npy` | Pre-computed embeddings (synthetic if Day 0 files absent) |

### Seed the database

```bash
python -m backend.seed_data
```

Idempotent — run as many times as needed. Logs:
```
Cameras:            6
Watchlist persons:  1
Watchlist vehicles: 1
FAISS vectors:      15
ALL CHECKS PASSED ✓
```

### SQLite schema

6 tables with locked column names (do not rename without updating all consumers):

| Table | Purpose |
|-------|---------|
| `cameras` | Camera registry with GPS, zone, status |
| `tracks` | Per-camera confirmed tracks. `global_id` NULL until Day 6 ReID assigns it. `body_embedding` NULL until Day 6. |
| `journeys` | Cross-camera journey steps per `global_id` |
| `watchlist_persons` | InsightFace **face** embeddings only. Not OSNet. |
| `watchlist_vehicles` | Plate numbers for ANPR matching |
| `alerts` | All alert types with flexible `metadata` JSON field |

**`alerts.metadata` design decision:** Rather than adding nullable columns for every new alert type (loitering duration, ANPR confidence, ReID score, feedback flags), these are stored as a JSON string in `metadata`. Read with `json.loads(row["metadata"])`. Do NOT "fix" this into six nullable columns.

### Embedding convention (use everywhere, no exceptions)

```python
from backend.embedding_utils import encode_embedding, decode_embedding, normalize_l2

# Store in DB:
blob = encode_embedding(vector)           # float32 → bytes

# Read from DB:
vec = decode_embedding(blob, dim=512)     # bytes → float32 ndarray

# Before FAISS insert or query (MANDATORY):
vec = normalize_l2(vec)                   # unit norm → inner product = cosine sim
```

### ⚠️ Critical distinction: Watchlist matching vs. Cross-camera ReID

These are **two completely separate matching systems**:

| | Watchlist Matching | Cross-Camera ReID |
|--|-------------------|------------------|
| **Model** | InsightFace (face recognition) | OSNet-IBN (body appearance) |
| **Target** | Known wanted persons | Unknown persons across cameras |
| **Storage** | `watchlist_persons.face_embedding` (SQLite) | FAISS index (`reid_reference.index`) |
| **Day built** | Day 3 (schema) + Day 7 (engine) | Day 3 (index seed) + Day 6 (engine) |
| **Search type** | Exact lookup (small fixed set) | Approximate nearest-neighbor (scale) |

**Do NOT** put OSNet body embeddings into `watchlist_persons`. **Do NOT** use InsightFace embeddings in FAISS. They are different models solving different problems.

### FAISS index details

- Type: `faiss.IndexFlatIP` (inner product = cosine similarity when normalized)
- Dim: 512 (OSNet-IBN default; change `faiss.embedding_dim` in config.yaml if different)
- Seeded with: 15 pre-computed OSNet body reference embeddings
- Thresholds (Day 6): `≥ 0.85` high-confidence match | `0.65–0.85` uncertain (review queue)
- **id_map.json must always travel with reid_reference.index** — one without the other is useless

### Concurrency

- WAL mode: multiple readers never block each other; writer doesn't block readers
- `threading.Lock()` in `db.py` serializes write operations
- All Day 2+ threads can safely call `db.upsert_track()` etc. concurrently

---

## Day 2: Backend + Live Dashboard

---

## Day 2: Backend + Live Dashboard

### Project structure

```
sentinel gujarat/
├── config.yaml              # Single config for all days
├── main.py                  # Detection pipeline (CLI + importable)
├── detector.py              # YOLOv8 wrapper
├── tracker.py               # 3-frame confirmation state machine
├── event_emitter.py         # JSONL + queue output
├── visualizer.py            # Frame annotation
│
├── backend/                 # FastAPI server (Day 2)
│   ├── main.py              # App, routes, lifespan
│   ├── connection_manager.py # WebSocket state + broadcast
│   ├── pipeline_bridge.py   # Thread → async bridge
│   └── config.py            # Config loader
│
├── frontend/                # React dashboard (Day 2)
│   └── src/
│       ├── App.jsx
│       ├── hooks/useWebSocket.js
│       └── components/
│           ├── DetectionCanvas.jsx
│           ├── StatusBar.jsx
│           └── TrackList.jsx
│
└── output/                  # Pipeline output
    ├── events.jsonl
    ├── annotated.mp4
    └── crops/track_{id}/
```

### How to run (Day 2)

**Terminal 1 — Backend:**
```bash
# From project root
$env:SENTINEL_SOURCE = "path/to/your/video.mp4"   # PowerShell
# OR: set SENTINEL_SOURCE=path/to/your/video.mp4  # CMD

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

### WebSocket message schema (contract — Day 5+ extends, does not replace)

```jsonc
// Sent once on connect — initialise client state:
{ "type": "snapshot", "camera_id": "CAM-01",
  "tracks": [{"track_id": 1, "bbox": [x1,y1,x2,y2], "confidence": 0.87, "frame_number": 45}] }

// Sent on every confirmed detection:
{ "type": "detection_event", "camera_id": "CAM-01", "track_id": 1,
  "frame_number": 60, "timestamp": 2.0, "bbox": [x1,y1,x2,y2], "confidence": 0.91 }

// Sent when a track hasn't updated within server.track_expiry_seconds:
{ "type": "track_expired", "camera_id": "CAM-01", "track_id": 1 }
```

### REST endpoints

| Method | Path | Response |
|--------|------|----------|
| GET | `/api/health` | `{"status": "ok", "pipeline_running": bool}` |
| GET | `/api/cameras` | `[{"camera_id": "CAM-01", "status": "online"}]` |
| WS | `/ws/detections` | Live event stream |

### Thread architecture

```
Video file (looped)
  └── detection thread (blocking — run_detection_pipeline)
          │  queue.Queue
          ├── _BridgedQueue callback
          │       └── loop.call_soon_threadsafe → asyncio.Queue
          │                   └── drain task → ConnectionManager.handle_detection_event
          │                                         └── broadcast to all WebSocket clients
          └── expiry task (every 1s) → check_and_expire_tracks → broadcast track_expired
```

### Video loop & monotonic timestamps

When `camera.loop_video: true`, the video restarts from frame 0 on EOF.
Timestamps are monotonic across restarts:
```
timestamp = loop_count × video_duration_seconds + frame_number / source_fps
```

### Dashboard scope boundary

The React canvas renders **bounding boxes only** — no video pixels.
Real video streaming (HLS) is Day 16. Do not add video playback to the
Day 2 frontend — it will conflict with the HLS architecture.

---

## Day 1: Detection + Tracking Foundation

---

## What this module does

Reads a video file (or live webcam stream) and:

1. **Samples frames** at a configurable processing rate (default: 5 fps processed, regardless of source FPS)
2. **Detects persons** using YOLOv8n — persons only, COCO class 0
3. **Tracks** each person with BoT-SORT (IoU-only, no ReID weights), assigning a stable `track_id`
4. **Confirms** tracks only after 3 *consecutive* processed frames — no gaps allowed
5. **Emits** one JSON line per confirmed track per processed frame to `output/events.jsonl`
6. **Saves throttled crops** to `output/crops/track_{id}/frame_{N}.jpg` for Day 3
7. **Annotates** frames with bounding boxes, IDs, and confidence for visual verification

---

## Quick start

```bash
# Install dependencies (torch + ultralytics + opencv-python)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python

# Run on a video file (saves annotated video + shows live window)
python main.py --source path/to/video.mp4 --config config.yaml --save-video --display

# Headless (no display) — useful on servers
python main.py --source path/to/video.mp4 --config config.yaml --save-video

# Webcam (index 0)
python main.py --source 0 --config config.yaml --display
```

---

## CLI flags

| Flag | Description |
|------|-------------|
| `--source PATH` | **Required.** Path to video file, or `0`/`1` for webcam index |
| `--config PATH` | Path to config YAML (default: `config.yaml` in CWD) |
| `--save-video` | Write annotated output to `output/annotated.mp4` |
| `--display` | Show live `cv2.imshow` window (press `Q` or `ESC` to quit) |

---

## Output folder structure

```
output/
├── events.jsonl              # One JSON line per confirmed track per frame
├── annotated.mp4             # Annotated video (if --save-video)
└── crops/
    ├── track_1/
    │   ├── frame_15.jpg      # First crop (saved immediately on confirmation)
    │   ├── frame_30.jpg      # Subsequent crops every crop_save_interval_frames
    │   └── ...
    ├── track_2/
    └── ...
```

### events.jsonl schema

```json
{"event_type": "person_track", "track_id": 1, "frame_number": 15, "timestamp": 0.5, "bbox": [120.5, 80.2, 300.1, 420.9], "confidence": 0.87}
```

| Field | Type | Notes |
|-------|------|-------|
| `event_type` | `"person_track"` | Always this value for Day 1 |
| `track_id` | `int` | Stable ID assigned by BoT-SORT |
| `frame_number` | `int` | Raw source frame index |
| `timestamp` | `float` | `frame_number / source_fps` (video time, not wall-clock) |
| `bbox` | `[x1, y1, x2, y2]` | Pixel coordinates |
| `confidence` | `float` | YOLOv8 detection confidence |

> **No `confirmed` field:** Events are only written for confirmed tracks.
> Its presence in this file is implicit. Do not re-add a redundant `confirmed` field.

---

## Configuration (config.yaml)

All magic numbers live in `config.yaml` — nothing is hardcoded in source files.

Key knobs:

| Key | Default | Purpose |
|-----|---------|---------|
| `model.path` | `yolov8n.pt` | Swap to `yolov8s.pt` / `yolov8m.pt` for better accuracy |
| `model.device` | `cpu` | Change to `cuda:0` for GPU — no code changes needed |
| `detection.confidence_threshold` | `0.65` | Detections below this are discarded |
| `tracking.min_confirmation_frames` | `3` | Consecutive frames required to confirm a track |
| `processing.fps` | `5` | Processed frames per second (not source FPS) |
| `output.crop_save_interval_frames` | `15` | Crop saved every N *processed* frames per track |
| `logging.level` | `INFO` | Set to `DEBUG` for per-frame noise |

---

## Architecture

```
main.py          CLI + frame sampling loop + orchestration
  ↓ sampled frames
detector.py      YOLOv8 + BoT-SORT wrapper → RawTrackResult[]
  ↓ raw tracks
tracker.py       TrackStateManager (3-frame confirmation) → ConfirmedTrack[]
  ↓ confirmed tracks only
event_emitter.py JSONL writer + throttled crop saver
visualizer.py    Annotated frame renderer (called by main.py)
```

### Module responsibilities

- **`detector.py`** — Loads YOLOv8 once. Calls `model.track()` per frame. Returns `RawTrackResult[]`. Knows nothing about confirmation.
- **`tracker.py`** — The *only* file with custom tracking logic. Owns `TrackStateManager`. Returns `ConfirmedTrack[]`. No other module touches the confirmation state machine.
- **`event_emitter.py`** — Receives only `ConfirmedTrack[]`. Writes JSONL and crops. Knows nothing about how confirmation works.
- **`main.py`** — Wires everything together. Owns frame sampling (`frame_number % frame_interval`). Owns video I/O.
- **`visualizer.py`** — Pure rendering. Only draws confirmed tracks.

---

## ⚠️ Critical: BoT-SORT ReID vs Day 3 Cross-Camera ReID

There are **two distinct ReID concepts** in this project that must never be confused:

### 1. BoT-SORT internal appearance ReID (`botsort.yaml → reid_weights`)
- **Scope:** Single camera feed only
- **Purpose:** Helps BoT-SORT re-associate a track after brief occlusion *within the same camera*
- **Status in Day 1:** **DISABLED** (`reid_weights: null`). IoU-only association is sufficient.
- **If you enable it:** Set `reid_weights: osnet_x0_25_msmt17.pt` in `botsort.yaml`. This does NOT affect Day 3.

### 2. Day 3 cross-camera appearance ReID (separate module, uses OSNet)
- **Scope:** Multiple camera feeds — matches the *same person* across *different cameras*
- **Purpose:** Resolves identity across camera handoffs — a completely different problem
- **Input:** Reads crops from `output/crops/track_{id}/` (written by this module's `event_emitter.py`)
- **Status in Day 1:** Not built yet. Lives in a separate codebase.
- **It does NOT touch `botsort.yaml`** — they are architecturally independent.

**TL;DR:** Setting `reid_weights` in `botsort.yaml` to a model path improves *single-camera* track continuity. Day 3 ReID matches persons *across cameras* by comparing saved crops. They solve different problems at different layers.

---

## 3-Frame Confirmation State Machine

Implemented in `tracker.py → TrackStateManager`:

- Each track has a `consecutive_frame_count` counter
- Counter **increments** when the track appears in a processed frame
- Counter **resets to 0** if the track is **absent for even 1 processed frame** (deliberate — prevents false confirmation from a flickering detection)
- Track is **confirmed** when `consecutive_frame_count >= min_confirmation_frames` (default: 3) reached **without interruption**
- Once confirmed, **confirmation is not revoked** — BoT-SORT maintains the ID across brief occlusion via Kalman prediction

---

## Day 2 / Day 3 integration notes

| Downstream | What it needs from Day 1 | Where to find it |
|------------|--------------------------|------------------|
| Day 2 (FastAPI) | `events.jsonl` field names: `event_type`, `track_id`, `frame_number`, `timestamp`, `bbox`, `confidence` | `output/events.jsonl` |
| Day 3 (ReID) | Crop folder: `output/crops/track_{id}/frame_{N}.jpg` | `output/crops/` |

**Do not rename these fields or change the crop path convention** without updating all downstream consumers.

---

## Known Day 1 limitations (to address in later days)

- Single camera only — cross-camera matching is Day 3
- No vehicle detection (ANPR) — later sprint
- No watchlist matching / alerts — later sprint
- CPU inference is ~2–5× slower than GPU — swap `model.device: cuda:0` when available
- Low-light / crowded scenes may produce more ID switches — ReID in Day 3 will compensate
- BoT-SORT IoU-only association can lose tracks in dense crowds — enabling `reid_weights` in `botsort.yaml` will improve this without touching Day 3
