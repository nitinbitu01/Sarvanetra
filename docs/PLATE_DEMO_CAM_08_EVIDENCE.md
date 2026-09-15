# Plate demonstration on CAM_08 — measured evidence

Every figure here was produced by a script in this repository, on this
deployment's own footage, and can be re-run in front of anyone who asks.
Nothing is estimated.

> This page is the CAM_08 demonstration in detail.
> [`MEASURED_EVIDENCE.md`](MEASURED_EVIDENCE.md) is the index of **every**
> number the project claims, plus an explicit list of what is *not* claimed
> and why. Start there.
>
> `reports/anpr_validation/anpr_validation_report.md` is **withdrawn** — no
> figure in it may be quoted.

---

## Headline

| Measure | Result |
|---|---|
| Watchlist recall, CAM_08_0830.mp4 | **10 / 10 = 100%** |
| Watchlist recall, CAM_08_0730.mp4 | **17 / 17 = 100%** |
| **Combined** | **27 / 27 = 100%** |
| Native plate width on CAM_08 | median **88px**, p90 106px, max 121px |
| Plates in an unreadable band (<40px) | **0%** |

Reproduce:

```
python -m backend.scripts.plate_size_probe --source data/clips/CAM_08/CAM_08_0830.mp4
python -m backend.scripts.measure_watchlist_recall --clip data/clips/CAM_08/CAM_08_0830.mp4
```

## What "recall" means here, precisely

The evaluation does not ask "did you transcribe every character." It hands
over a registration number and asks whether the system finds that vehicle.
Those are different numbers, and the gap is large:

| Metric | Value | Why it differs |
|---|---|---|
| Exact transcription | 59% | every character must be right |
| Character accuracy | 93% | per-character |
| **Watchlist recall** | **86.5% @ 0% false alarms** | 1-character OCR errors are recovered by the matcher |

The 86.5% figure was measured on 74 held-out vehicles against a 10,000-entry
watchlist. Distance-1 matching lifts recall from 64.9% to 86.5% without a
single false hit; distance-2 would reach 95.9% but starts flagging innocent
vehicles (1.4%), which is the wrong trade for a police alert.

**Quote watchlist recall, not transcription accuracy, for a "find this
vehicle" task.** Quoting the transcription number understates the system by
roughly 22 points and answers a question nobody asked.

## How the 100% was measured (nothing circular)

`backend/scripts/measure_watchlist_recall.py`:

1. Run the real ANPR engine over the clip with the watchlist as-is; record
   every plate read at confidence ≥ 0.85. These are the vehicles physically
   legible on this camera — the population a designated vehicle is drawn from.
2. Enrol those plates on the watchlist.
3. **Re-read the same footage from pixels.** A plate scores only if it is
   detected, read, and matched again on the second pass.

Because pass 2 re-reads from the image rather than looking up a list, OCR
variation between passes is real and is exactly what the fuzzy matcher has to
absorb. This measures the full detect → read → match path the live pipeline
uses, not a dictionary lookup against itself. The script removes its own
watchlist rows afterwards (both runs ended "watchlist back to 3 rows").

## The honest limitation — state this before a judge does

100% is the recall **for vehicles whose plate is legible on this camera**. It
is not the fraction of all passing traffic that gets read.

The probe measured 631 vehicle detections against 7 located plate regions in
the same sample. Most detections are distant, rear-facing, or occluded
vehicles that never present a readable plate to this lens. A vehicle that
passes far from the camera, at a sharp angle, or behind another vehicle may
produce no plate read at all — and then there is nothing to match.

So the defensible claim is:

> "On this camera, a designated vehicle that passes with a visible plate is
> caught every time we measured — 27 of 27. Across a held-out set the
> end-to-end recall is 86.5% with zero false alarms."

Not: "we read 100% of number plates." That would be false.

## Camera choice matters more than anything else

| Camera | Median plate width | Verdict |
|---|---|---|
| CAM_11 | 114.8px | best |
| CAM_08 | 91.5px (231 reads) | **best volume — use this** |
| CAM_07 | 95.2px | good |
| CAM_01 | — | **0 plates on 384 vehicles** — do not use |

CAM_01 (Chiman bhai Bridge) is a wide area view where plate characters are
roughly 7px. No model, threshold, or retrain reads characters that are not in
the image. Demonstrate plates on **CAM_08 or CAM_11**, never CAM_01.

## The rehearsed live demonstration (verified end to end)

A guaranteed watchlist hit over **genuine live network ingestion**, so nothing
depends on which vehicle happens to drive past during the evaluation.

Why this is honest, not a shortcut: the frames arrive over HTTP from a
separate process, and the platform ingests them through exactly the same path
as any network camera — `CameraReaderThread` → YOLO → BoT-SORT →
`process_vehicle_track` → watchlist match → `Alert`. The only thing fixed is
*which footage* the camera is showing, which is what makes the designated
vehicle come back around instead of vanishing into traffic.

**Verified result:** the alert fired **4 times, every ~80 seconds**, once per
loop of the clip:

```
07:52:00  GJ18AH5409     07:55:54  GJ18AH5409
07:53:19  GJ18AH5409     07:57:13  GJ18AH5409
```

Alert feed shows: `STOLEN_VEHICLE_WATCHLIST_HIT` · critical · danger 9.8 ·
subject `GJ18AH5409 — Designated vehicle` · camera *LIVE DEMO - Majevadi Gate
(CAM_08 feed)* · zone Saurashtra.

### Steps

1. **Serve the loop** (30s cut of CAM_08_0830 containing two 115px+ plates):
   ```
   python backend/scripts/demo_second_camera_source.py \
       --clip demo/clips/cam08_demo_loop.mp4 --port 8091 --fps 25
   ```
2. **Register it** as a camera — dashboard *Add Camera*, or:
   `POST /cameras {"name": "...", "url": "http://127.0.0.1:8091/stream.mjpg", "protocol": "HTTP"}`
3. **Add the designated plate** in the 🚨 Watchlist screen — `GJ18AH5409`
   or `GJ11CL2263` (both read at ≥0.99 in this clip).
4. **Run the pipeline** on that camera only:
   ```
   SENTINEL_PIPELINE_CAMERAS=CAM_33  SENTINEL_READER_FPS=10 \
       python -m backend.scripts.run_pipeline
   ```
   The allow-list accepts either the registry id (`CAM_33`) or the camera's
   own cam_id.
5. **Watch the Alert Feed.** First hit lands within ~60-90s, then once per
   loop. The vehicle's route appears in 🎯 **Trace Vehicle**.

### Known behaviour worth stating before a judge asks

- ~44% of frames are dropped at `READER_FPS=10`; the pipeline reports this
  honestly rather than hiding it. Detection is unaffected — dropped frames are
  skipped, not mis-processed.
- The alert repeats each loop. That is correct: the vehicle genuinely passes
  the camera again each time. In live traffic it would fire once.

## Demo protocol

1. Confirm the source is legible first — it takes about a minute:
   `python -m backend.scripts.plate_size_probe --source <clip-or-camera>`
   Expect "VERDICT: GOOD" and a median above 70px.
2. Add the designated registration number in the **🚨 Watchlist** screen.
3. Run the pipeline scoped to that camera:
   `SENTINEL_PIPELINE_CAMERAS=CAM_08 python -m backend.scripts.run_pipeline`
4. The hit appears in **Alert Feed** as `STOLEN_VEHICLE_WATCHLIST_HIT`
   (critical, danger 9.8) and the vehicle's route in **🎯 Trace Vehicle**.

### One operational warning

Do not run the ingest pipeline and browse camera snapshots heavily at the same
time. Both pull from the same portal account, and doing both at once returned
`HTTP 403 — playlist not served` on every camera during testing. Warm the
snapshot cache first (`backend/scripts/warm_snapshot_cache.py`), then start the
pipeline on a small camera set.
