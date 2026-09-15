# Macro Traffic Analytics — Implementation Plan (Module 3)

**Status:** planned, not built. Phase 0 is the blocker for everything else.
**Verified against `output/sentinel.db` on 2026-08-30.**

The original plan (Kafka + Flink + TimescaleDB + vector tiles + ST-GNN) is
technically sound but assumes a system this project does not have. Nothing
downstream — speed, congestion, routes, forecasting — is computable until
cameras are calibrated and tracks are persisted.

---

## 0. Ground truth (measured, not assumed)

| Fact | Value |
|---|---|
| Cameras configured | 32 |
| Cameras with any sightings | 13 |
| `detections` rows | **0** — and nothing writes it |
| `tracks` rows | **0** |
| `camera_calibration` (px_per_meter) rows | **0** |
| `camera_calibrations` (homography) rows | **0** |
| `journey_events` rows | 914 (858 vehicles, ~4 observed hours of 2026-06-14) |
| Vehicles seen on 2+ cameras | 52, across 25 camera pairs |

The `Detection` class in `backend/core/gpu_inference_engine.py` is an in-memory
dataclass, **not** the ORM model in `backend/db/models.py`. No code path
persists detection rows.

Consequence: congestion index, vehicle density, OD matrices and predicted
bottlenecks cannot be computed from current data. Any such figure would be
invented. See `docs/ANPR_FEASIBILITY.md` for the same discipline applied to
plate recognition.

---

## 1. What changes from the original plan, and why

Three load-bearing choices are wrong *for this deployment* — not wrong in
general, wrong at 32 cameras on one box.

### 1.1 Kafka / Flink → defer

At 32 cameras averaging ~10 vehicles/camera/minute the network produces
**~5 track events per second**. Kafka's value is durability, replay and fan-out
*across machines*; Flink's is distributed windowing. Neither constraint binds
here. An in-process asyncio queue with batched SQLite (WAL) writes gives the
same ordering and durability guarantees with no cluster to operate. The replay
capability that actually matters — reprocessing history when aggregation logic
improves — comes free from keeping the raw track table.

**Revisit at ~200 cameras, or the first multi-node deployment.**

### 1.2 Store tracks, not detections (~150x fewer rows)

The original event schema is per-observation. Persisting every detection at
5 fps across 32 cameras with ~5 vehicles in frame is ~800 rows/sec, or
**69M rows/day** — untenable on SQLite and wasteful anywhere.

A vehicle crossing one camera is *one measurement*, not fifty. Collapsing each
track to a single summary row gives **~460K rows/day**. The per-frame path
still exists in memory for the speed fit; only redundancy is discarded.

### 1.3 Point-and-corridor, not a road-segment graph

OSM segment graphs, TAZs and IPF assume dense city coverage where a camera sits
on most links. This network spans **four cities** (Ahmedabad, Surat, Junagadh,
Patan). Only 52 vehicles were ever seen on two cameras, across 25 pairs — one
pair is 70 km apart.

Fitting a segment graph to that yields a model with almost no observed links.
The honest structure is **cameras as measurement points, well-sampled camera
pairs as corridors** — which is what the shipped dashboard already uses.

---

## 2. Architecture decisions

| Element | Ruling | Reasoning |
|---|---|---|
| Homography calibration | **Adopt** | The one true prerequisite. `junction_topology.py` already computes in world metres. |
| Pixel → WGS-84 transform | **Adopt** | Homography gives camera-local metres only; GPS anchor + bearing completes it. |
| Track-summary persistence | **Adopt** | ~150x fewer rows, no information lost. |
| Confidence-weighted aggregation | **Adopt** | A degraded feed must not corrupt a network figure. |
| Ingestion anomaly filtering | **Shipped** | Speed-band rejection running in `backend/services/network_activity.py`. |
| `free_flow_speed` bootstrap | **Adopt** | 85th percentile off-peak. CI stays hidden until it exists. |
| Seasonal baseline alerting | **Adopt** | Time-of-day x day-of-week beats a static threshold, needs no GNN. |
| Kafka / Redpanda | Defer | ~5 events/sec. Revisit at ~200 cameras. |
| Apache Flink | Defer | Distributed windowing for a single-box workload. |
| TimescaleDB + PostGIS | Defer | Camera→road mapping is precomputed offline; no runtime geospatial join needed. |
| Vector tiles (Martin) | Defer | 32 points, ~25 corridors. Leaflet is already a dependency. |
| PTP clock sync | Defer | Only needed for adjacent-camera appearance correlation, out of scope. |
| ST-GNN (STAEformer/DCRNN) | **Reject** | Benchmarked on 200–900 dense sensors. 32 sparse cameras form no meaningful graph; needs months of history. |
| TAZ + IPF origin–destination | **Reject** | Cannot converge on 25 pairs with n=1–15. Corridor flow with sample counts is the substitute. |
| YOLO26 (AGPL-3.0) | **Reject** | AGPL would reach the aggregation API and dashboard. Project already runs its own ONNX detector. |

---

## 3. Data model

Two new tables, two columns on an existing one.

```sql
-- Extend the EXISTING (empty) camera_calibrations table.
-- homography_matrix, reprojection_error_m and camera_angle_deg already exist.
ALTER TABLE camera_calibrations ADD COLUMN gps_anchor_lat REAL;
ALTER TABLE camera_calibrations ADD COLUMN gps_anchor_lon REAL;

-- One row per vehicle per camera. NOT one row per frame.
CREATE TABLE vehicle_track (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id          TEXT    NOT NULL,
    track_id           INTEGER NOT NULL,
    vehicle_class      TEXT,
    first_seen         DATETIME NOT NULL,
    last_seen          DATETIME NOT NULL,
    entry_lat          REAL,   entry_lon    REAL,
    exit_lat           REAL,   exit_lon     REAL,
    heading_deg        REAL,
    speed_kmh          REAL,   -- robust fit; NULL if uncalibrated
    speed_ci_kmh       REAL,   -- half-width of the estimate
    path_length_m      REAL,
    n_frames           INTEGER,
    detector_conf      REAL,
    calibration_err_m  REAL,   -- carried from calibration, for weighting
    quality            TEXT    -- good | degraded | speed_unavailable
);
CREATE INDEX idx_vt_cam_time ON vehicle_track(camera_id, first_seen);

-- Pre-aggregated rollup. The dashboard reads this, never vehicle_track.
CREATE TABLE camera_metrics_1m (
    bucket_start     DATETIME NOT NULL,
    camera_id        TEXT     NOT NULL,
    vehicle_count    INTEGER,
    count_by_class   TEXT,     -- JSON
    median_speed_kmh REAL,
    p15_speed_kmh    REAL,
    p85_speed_kmh    REAL,
    speed_samples    INTEGER,  -- n, surfaced in the UI
    confidence       REAL,
    congestion_index REAL,     -- NULL until free_flow_speed bootstraps
    PRIMARY KEY (bucket_start, camera_id)
);
```

**Why speed is nullable:** `speed_kmh` is `NULL` — never zero, never estimated —
when a camera has no calibration or fails its quality gate. Zero is a
measurement; NULL is an absence.

---

## 4. Phased build

Numbering is a hard dependency chain. Phase *n* produces the input Phase *n+1*
consumes; skipping ahead produces invented numbers.

### Phase 0 — Calibration & geo-referencing (3–5 days)

Blocks everything. Speed, mapping and corridors all silently depend on it.

- Add the two GPS-anchor columns (matrix, reprojection error, bearing already exist).
- Calibration tool: operator marks >= 4 coplanar road points on a still frame
  and supplies real-world offsets; `cv2.findHomography` stores the matrix plus
  RMS reprojection error in metres.
- **Quality gate** — accept < 0.50 m, flag 0.50–1.00 m, reject > 1.00 m. A
  rejected camera is excluded from every speed product, not silently included.
- Implement `pixel_to_world_m()` and `world_to_wgs84()` (anchor + bearing +
  matrix), with unit tests against synthetic homographies of known ground truth.

**Done when:** >= 6 pilot cameras calibrated; a held-out known distance in each
scene reproduces within 5%; pixel → world → pixel round-trip error < 1 px.

### Phase 1 — Track persistence (4–6 days)

Creates the measurement stream the module has been missing.

- Wire the existing detector + tracker to emit one `vehicle_track` row per
  completed track.
- **Robust speed** — fit over the track's world-coordinate path (Theil–Sen, or
  median of per-interval speeds) rather than differencing endpoints, which is
  dominated by bounding-box jitter. Emit a confidence interval alongside.
- Reject at write time: speed outside [0, 150] km/h, tracks under a minimum
  frame count or path length, any camera failing the Phase 0 gate.
- Batch inserts; SQLite in WAL mode.

**Done when:** tracks persist on pilot cameras; write throughput measured under
live load; speed histogram inspected for physical plausibility; zero speed rows
originate from uncalibrated cameras.

### Phase 2 — Rollups & camera health (2–3 days)

- Populate `camera_metrics_1m` on a rolling schedule.
- Weight each camera's contribution by calibration error and detector confidence.
- Integrate the existing `last_heartbeat_at` health signal: a degraded or
  offline camera is **flagged and excluded**, never imputed as zero traffic.

**Done when:** rollup query latency measured; killing a live feed surfaces
"degraded" on the dashboard rather than a zero reading.

### Phase 3 — Free-flow speed & Congestion Index (1–2 days + 2–3 week wait)

The only phase gated on wall-clock time rather than engineering effort.

- `free_flow_speed` per camera = 85th percentile of observed off-peak speeds,
  requiring a minimum sample count across >= 7 days. Refresh monthly.
- `CI = 1 - clamp(v / v_free, 0, 1)`, exposed **only** once a camera satisfies
  the bootstrap threshold.
- Until then the UI shows bootstrap progress ("12 of 200 samples"), not a
  provisional number.

**Done when:** CI is absent for every camera below threshold and begins
reporting automatically as each crosses it.

### Phase 4 — Live dashboard (3–4 days)

Extends the panel already shipped rather than replacing it.

- Add live speed and CI layers to `frontend/src/components/analytics/NetworkActivityPanel.jsx`.
- Geographic layer on Leaflet (already in `package.json` — no new dependency),
  keeping the current SVG rendering as the offline fallback.
- Push updates over the existing `connection_manager` WebSocket.

**Done when:** end-to-end camera → dashboard latency measured and published;
offline fallback verified by blocking tile requests.

### Phase 5 — Baselines & corridor flow (4–6 days)

Delivers the plan's intent — early warning and popular routes — without a GNN.

- Per-camera seasonal baseline over time-of-day x day-of-week; alert on
  sustained deviation (3+ minutes), not instantaneous breach.
- Corridor flow matrix between camera catchments, every cell carrying its
  sample count — the supportable substitute for TAZ/IPF.

**Done when:** baseline alerting beats a static threshold on a hand-labelled
week, measured by precision and recall.

### Critical path

~**3–4 weeks of engineering**, plus a mandatory **2–3 week data accumulation
window** running concurrently once Phase 2 lands. Congestion Index cannot appear
before that window closes — no amount of engineering shortens it.

---

## 5. What makes this better than the typical build

Not the stack. Every demo in this space has a map with coloured lines; most
cannot tell you where a single number came from.

- **Provenance on every figure.** Sample count, confidence and measurement basis
  travel in the API payload, not just the UI — an exported CSV or a screenshot
  keeps its caveats.
- **A calibration quality gate that actually excludes.** Cameras over 1.0 m
  reprojection error contribute no speed at all. The common failure is including
  them silently and reporting a citywide average that is 30–50% wrong.
- **Published error, not asserted accuracy.** Speed MAPE measured against ground
  truth and shown; "+/-4 km/h at 95%" is a stronger claim than an unqualified
  number because it can be checked.
- **Absence encoded as absence.** A dead camera reads "degraded", never "zero
  traffic" — the failure mode that makes a broken feed look like good news.
- **Robust estimators over convenient ones.** Theil–Sen over endpoint
  differencing; median over mean; p15/p85 alongside the centre.
- **Junction semantics most systems lack.** `backend/services/junction_topology.py`
  already models clockwise roundabouts, BRTS corridors, median U-turn grace and
  Frenet-frame slipways. Calibration is what switches it on.

---

## 6. The honesty contract

| We will not | Because |
|---|---|
| Show a Congestion Index before `free_flow_speed` bootstraps | CI against a guessed free-flow speed is a colour, not a measurement. |
| Report speed from an uncalibrated camera | Pixel-motion speed carries 30–50% error varying by camera angle. |
| Render a camera with no data as zero traffic | It is a coverage gap. Zero makes a broken feed look like an empty road. |
| Publish an OD matrix from 25 sparse camera pairs | IPF will not converge; output would be structure invented by the prior. |
| Describe corridor speed as road speed | It derives from straight-line GPS distance, so it is a lower bound. |
| Interpolate a missing segment to fill the map | A smooth map is not the goal. A trustworthy one is. |

---

## 7. Validation

### Speed ground truth, without a radar gun

1. **Known in-scene distance.** Hold out a surveyed distance not used in the
   homography fit and measure it back. Target within 5%.
2. **Manual frame-count.** For ~20 vehicles, count frames between two marked
   ground points and compute speed by hand. Closest thing to a reference
   standard available here.
3. **Cross-camera consistency.** For a well-sampled pair, compare in-camera
   speed against corridor transit speed over a known road distance.

### Regression and failure testing

- Synthetic homography fixtures with analytically known answers — catches sign
  errors and axis swaps that visual inspection misses.
- Replay harness feeding recorded tracks through aggregation, asserting exact
  expected rollups.
- Chaos test: kill a feed mid-stream, assert the dashboard shows degraded rather
  than zero.
- Speed MAPE and baseline precision/recall tracked weekly as product metrics,
  not one-time validation.

---

## 8. Start here

Phase 0 is the whole unlock and is smaller than it looks: two columns, a
calibration tool, two transform functions, and their tests.

One decision before any code: **which cameras form the pilot set.** Six is
enough. Choose them where a real-world distance can actually be surveyed and
where two share a corridor, so Phase 5 has something to measure.

On current data, **`CAM_11` → `CAM_08`** (Junagadh Dolatpara → Majewadi Gate) is
the strongest candidate pair — 15 observed transits over 2.36 km, already
producing a credible >= 27.8 km/h.

---

## Appendix — what is already shipped

`GET /api/v1/analytics/network` (in `backend/routers/v1/analytics.py`, computed
by `backend/services/network_activity.py`) and the "Macro Traffic" panel on the
Analytics page. All figures derived from `journey_events`:

- Per-camera sighting volume, distinct vehicles, share of network, peak hour
- Hourly temporal profile
- Coverage gaps (19 cameras with no indexed sightings, flagged as gaps)
- Corridor transits with sample counts, confidence labels and lower-bound speeds
- `measurement_basis` block stating what is *not* measured and why
