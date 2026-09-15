# Sarvanetra AI — One Pipeline, Nine Stages

**This document exists because the system was being *described* as two pipelines
("a 7-stage ANPR pipeline" and "a 6-stage CCTV pipeline") when the code is, and
always was, one continuous pipeline.** ANPR is not a second system running
alongside the first: it is a function call inside the main frame loop
(`live_24x7_pipeline._process_frame_batch` calls
`anpr_engine.process_vehicle_track` at line 1855). Describing it as two systems
made the architecture look more complicated than it is and invited the question
"how do the two pipelines stay in sync?" — which has no answer, because there is
only one.

Nothing in the code changed to produce this document. The stage numbering below
is a naming change only.

---

## The whole system in one sentence

A camera sees traffic → the AI finds vehicles → follows them → reads their
plates → recognises the same vehicle on another camera → works out speed and
traffic → stores it → shows it → raises alerts.

```
                    30 CITY CCTV CAMERAS
                            |
   STAGE 1  INGESTION          per-camera threads, authenticated HLS
                            |
   STAGE 2  VEHICLE DETECTION  YOLOv8s + fine-tuned detector
                            |
   STAGE 3  TRACKING           BoT-SORT, camera-isolated track ids
                            |
   STAGE 4  ANPR RECOGNITION   7 sub-steps, see below
                            |
   STAGE 5  IDENTITY FUSION    plate key + OSNet appearance
                            |
   STAGE 6  TRAJECTORY + SPEED homography, Theil-Sen, GIS route
                            |
   STAGE 7  TRAFFIC ANALYTICS  volume heatmap, average speed
                            |
   STAGE 8  ALERTS             watchlist + clone-plate physics
                            |
   STAGE 9  PERSISTENCE + UI   SQLite (WAL), FastAPI, React
```

---

## Stage-to-code map

Every row is a real location in this repository.

| Stage | Implementation | Status |
|---|---|---|
| **1 — Ingestion** | `live_24x7_pipeline.py:240` `CameraReaderThread`; `:1402` `_discover_cameras`; `:559` `_read_live_network_frame`; `hls_ffmpeg_capture.py` for AES-128 HLS | Shipped |
| **2 — Vehicle detection** | `live_24x7_pipeline.py:1570` GPU consumer loop; `:1626` `_process_frame_batch` | Shipped |
| **3 — Tracking** | BoT-SORT inside `_process_frame_batch`; `:1331` `_global_track_id` namespaces ids per camera | Shipped |
| **4 — ANPR** | `anpr_engine.py:987` `process_vehicle_track`, called from `live_24x7_pipeline.py:1855` | Shipped |
| **5 — Identity fusion** | `live_24x7_pipeline.py:2321` (`reid_id` = confirmed plate); OSNet embedder at `:68` | **Partial** — see below |
| **6 — Trajectory + speed** | `speed_estimator.estimate_track_speed` at `:2193`; `calibration.pixel_to_world_m`; `routers/v1/journeys.py` for cross-camera routes | Shipped |
| **7 — Traffic analytics** | `:2612` one-minute rollups; `congestion_engine.CongestionEngine` at `:968`; `/analytics/traffic-density` | **Partial** — see below |
| **8 — Alerts** | `anpr_engine.py:1187` `check_watchlist`; `plate_clone_detector.py` `find_clones` | Shipped |
| **9 — Persistence + UI** | `:2156` `_persist_vehicle_track`; FastAPI routers; React dashboard | Shipped |

### Stage 4 internals (the old "7-stage ANPR pipeline")

| Sub-step | Function |
|---|---|
| 4.1 Adaptive enhancement | `anpr_engine.py:495` `_enhance_frame` |
| 4.2 Plate detection | `:505` `detect_plate_bbox` (plate_v3_ft) |
| 4.3 Rectification | `:708` `_rectified_variants` (present, default OFF — median tilt here is 0.00 deg) |
| 4.4 Quality gate | `:576` `extract_plate_candidate` |
| 4.5 OCR | `:793` `_crnn_infer`, `:834` `recognize_plate` (3-model CRNN + EasyOCR fallback) |
| 4.6 Grammar decode | `decode_plate()` — Indian plate syntax |
| 4.7 Temporal voting | `:1205` `_get_voter`, `:1213` `finalize_track` |

---

## Efficiency: already implemented, do not "add" it

A common review suggestion is to stop running OCR on every frame and to score
frames for quality first. **Both are already in `process_vehicle_track`:**

| Guard | Line | Effect |
|---|---|---|
| Class filter | `anpr_engine.py:999` | persons and bicycles never reach OCR |
| **Confirmed-plate lock** | `:1002-1005` | once a plate is locked the function returns immediately — no enhancement, no detection, no OCR |
| Quality gate | `:1012` | a blurred or undersized crop returns before OCR |
| **Best-N view selection** | `:1019-1024` | crops are sorted by Laplacian sharpness and only the sharpest `FUSION_MAX_VIEWS` are kept |
| Minimum width | `:1033` `SINGLE_FRAME_FLOOR_PX` | too-small plates skip OCR unless a fused view exists |

Re-implementing these would add risk and gain nothing.

---

## What is genuinely partial — state it, do not hide it

**Stage 5 (identity fusion).** The global identity key is currently the
*confirmed plate string*. The OSNet appearance embedding is computed and stored
but is not yet combined with plate, time, direction and camera adjacency into a
single weighted match score. Cross-camera journey search nevertheless measures
**91.1% precision / 85.0% recall**. A weighted multi-signal score is the right
next step, and the weights must be tuned on validation data rather than assumed.

**Stage 7 (traffic analytics).** What exists and is honest: traffic **volume**
heatmap, average vehicle speed on calibrated cameras, direction of travel per
leg, congestion index. What is **not** yet computable honestly: a full
origin-destination matrix and true occupancy-based density. Present those as
roadmap.

---

## Corrections to keep out of any deck

| Claim seen in review notes | Reality here |
|---|---|
| "27-29 cameras" | **30** (`is_deleted = 0 AND NOT is_test_camera()`) |
| "PostgreSQL / PostGIS" | **SQLite in WAL mode.** Correct choice at this scale; do not claim otherwise |
| "Tune the plate detector's confidence" | Measured and settled: `conf=0.75`, `imgsz=320`. Lowering conf restores a **50% false-positive rate**; raising imgsz **loses 7-11% of detections** (crops are 129 px median, so 320 already upscales) |
| "Active Learning Vault gives a retraining loop" | The vault **harvests** (real). The retraining loop is **not** proven — the label pool is exhausted and self-training was measured and rejected (0 of 300 accepted). Say "harvesting for future retraining" |
| "Route anomaly detection" | What exists is a **clone-plate physics check** (haversine distance vs a speed ceiling): 10/10 known-answer tests, **0 false alarms on 727 real sightings**. It is explainable, which is better for evidence than a learned model |
| ">90% accuracy" | **93.1% character accuracy** on 83 held-out vehicles; **80.4% exact-plate** at the committed operating point. Always state the unit |

---

## The line to say out loud

> "It is one pipeline with nine stages. ANPR is stage four, and it has seven
> sub-steps of its own. There is no second system to keep in sync."
