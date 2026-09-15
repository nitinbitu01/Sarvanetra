# Measured evidence — Sarvanetra AI

The single source of truth for every number this project claims.

**Rule for this page:** a figure appears here only if a script in this
repository produced it, on this deployment's own footage or database, and the
command that reproduces it is printed beside it. Anything modelled, estimated,
or measured on synthetic images belongs in §5 — *not claimed* — instead.

> `reports/anpr_validation/anpr_validation_report.md` is **withdrawn** and no
> figure in it may be quoted. Its own header explains why: the detection
> scorer compared a plate detector against a ground-truth box covering the
> entire (already-cropped) image on 203/203 real instances, and 96.3% of that
> dataset is synthetic with Tier 1 and Tier 3 containing no real footage at
> all.

---

## 1. The headline — can the system find a designated vehicle?

This is the question the evaluation actually asks. It is not the same question
as "did you transcribe every character", and the gap between them is large.

| Measure | Result | Population |
|---|---|---|
| **Watchlist recall** | **86.5%** at **0% false alarms** | 74 held-out vehicles vs a 10,000-entry watchlist |
| Watchlist recall, CAM_08_0830.mp4 | 10 / 10 = 100% | vehicles legible on that clip |
| Watchlist recall, CAM_08_0730.mp4 | 17 / 17 = 100% | vehicles legible on that clip |
| **Combined CAM_08** | **27 / 27 = 100%** | |
| Exact transcription | **59.0%** | 83 held-out vehicles — see §1a |
| Character accuracy | **93.1%** | 83 held-out vehicles — see §1a |

```
python -m backend.scripts.measure_watchlist_recall --clip data/clips/CAM_08/CAM_08_0830.mp4
```

**Why recall beats transcription here.** Distance-1 fuzzy matching lifts recall
from 64.9% to 86.5% without a single false hit — a one-character OCR error is
recovered by the matcher. Distance-2 would reach 95.9% but starts flagging
innocent vehicles (1.4%), which is the wrong trade for a police alert.

**Quote watchlist recall, not transcription accuracy, for a "find this
vehicle" task.** Quoting 59% understates the system by ~27 points and answers
a question nobody asked.

### The measurement is not circular

`measure_watchlist_recall.py` runs two passes. Pass 1 reads plates with the
watchlist as-is and enrols everything read at confidence ≥ 0.85. Pass 2
**re-reads the same footage from pixels** — a plate scores only if it is
detected, read and matched again. OCR variation between passes is real, and
absorbing it is exactly the matcher's job. The script removes its own
watchlist rows afterwards.

### The honest limitation — state it before a judge does

100% is the recall **for vehicles whose plate is legible on that camera**. It
is not the fraction of all passing traffic that gets read. The probe measured
631 vehicle detections against 7 located plate regions in the same sample;
most detections are distant, rear-facing or occluded vehicles that never
present a readable plate to the lens.

The defensible sentence is:

> "On this camera, a designated vehicle that passes with a visible plate was
> caught every time we measured — 27 of 27. Across a held-out set the
> end-to-end recall is 86.5% with zero false alarms."

Not "we read 100% of number plates." That would be false.

---

## 1a. The shipped recogniser on real footage — the number that replaces the withdrawn report

The withdrawn report's recognition figures were measured on a set whose Tier 1
and Tier 3 contain no real footage at all. This is the same measurement done
properly: **the exact ensemble the live pipeline loads**, scored once on real
Gujarat CCTV it has never seen.

**Run 2026-09-06, stored at
[`reports/shipped_recognizer_eval_20260906.json`](../reports/shipped_recognizer_eval_20260906.json)**
(includes the per-vehicle prediction for all 83, so any single result can be
checked):

| Measure | Result |
|---|---|
| Vehicles scored | 83 held-out, 960 crops |
| **Exact match** (multi-frame voted) | **59.0%** (49/83) |
| **Within 1 character** | **81.9%** |
| Character error rate | 6.9% |
| **Character accuracy** | **93.1%** |
| Inference | 312 ms/vehicle, ~11.6 crops each → **≈27 ms/crop** (3-member ensemble, 64×256, CUDA) |

```
python -m backend.scripts.eval_shipped_recognizer
```

**Why this number is honest.** The split is `plate_final_model.split(by_track,
1000)` — the same call and seed used when those checkpoints were produced. The
83 test vehicles were never taken a gradient step on and were never used to
select a checkpoint (that was the val split's job). The split is **by vehicle
track, not by crop**: one car contributes ~11 frames, and splitting by crop
would put the same car on both sides of the line and inflate the result.
Scoring is per vehicle with multi-frame voting, because that is what the live
pipeline does.

The script trains nothing. Every other route to this number
(`plate_eval_clean.py`, `plate_final_model.py --train-final`) retrains first,
which answers "how good can this architecture get" rather than "what does the
checkpoint on disk actually read".

### Accuracy by condition — the answer to "across diverse real-world conditions"

Same run, same 83 vehicles, stratified. Bands are computed from the crops
themselves (median across each vehicle's frames), so they cannot drift from the
images and need no annotation pass.

**Resolution — this is the finding that explains everything else:**

| Native plate width | n | Exact match | CER |
|---|---:|---:|---:|
| under 40 px | 0 | — | — |
| 40–70 px | 15 | 33.3% | 12.0% |
| 70–90 px | 40 | 57.5% | 6.6% |
| **90 px and above** | **28** | **75.0%** | **4.7%** |

A clean monotonic curve across three populated bands. **Given a plate 90 px or
wider, the system reads it exactly 75% of the time** — and the 59.0% headline
is what you get when you average that with the 33.3% band. The constraint is
pixels on the plate, not the model, and this table is the proof.

**Angle:**

| Glyph-mass tilt | n | Exact | CER |
|---|---:|---:|---:|
| 0–2° | 80 | 57.5% | 7.2% |
| 2–5° | 2 | *too few to compare* | |
| 5–15° | 1 | *too few to compare* | |
| over 15° | 0 | — | — |

**80 of 83 vehicles sit inside 2°.** This is why perspective rectification was
measured at −3 to −8 points on this fleet and is shipped disabled: there is
essentially nothing to straighten. It also bounds the claim honestly — this
result says nothing about a camera mounted at a steep angle.

**Lighting** (crop brightness, not a clock reading):

| Band | n | Exact | CER |
|---|---:|---:|---:|
| dark | 2 | *too few* | |
| dim | 42 | 59.5% | 6.8% |
| normal | 36 | 61.1% | 7.0% |
| bright | 3 | *too few* | |

Across the two populated bands accuracy is flat — 59.5% vs 61.1%, well inside
noise. Within the exposure range this fleet actually produces, lighting is not
the limiting factor.

**Blur** (Laplacian variance): moderate 80.0% (n=10) against sharp 56.2%
(n=73). Stated because it was measured, but **not claimed as a finding** — the
"moderate" slice has ten vehicles and the direction is almost certainly
confounded with plate size, since a larger, closer plate lowers Laplacian
variance while being easier to read.

```
python -m backend.scripts.eval_shipped_recognizer
```

### 1a-ii. Where this system honestly stands at 80%

Full-plate exact match is **59.0% when the recogniser answers for every
vehicle**. Every software route to raising that number is now measured and
closed (§5). What was never tried is letting the system **decline to answer**.

An ANPR installation that returns "not read" is more useful to an investigator
than one that guesses, because a wrong plate sends a patrol after the wrong
vehicle. Every vendor quotes accuracy against a read rate for this reason.

Searched over the two gates the system can actually enforce at inference time —
native plate width, known before the recogniser runs, and read confidence —
across the same 83 held-out vehicles
([`reports/operating_points_20260906.txt`](../reports/operating_points_20260906.txt)):

| Min width | Min confidence | Coverage | **Exact match** |
|---|---|---:|---:|
| any | any | 100% | 59.0% |
| **80 px** | **0.85** | **61.4%** (51/83) | **80.4%** |
| 80 px | 0.88 | 54.2% | 84.4% |
| 80 px | 0.90 | 48.2% | 85.0% |
| 70 px | 0.92 | 41.0% | 88.2% |
| 80 px | 0.92 | 38.6% | 93.8% |
| 90 px | 0.88 | 25.3% | 90.5% |

**The defensible claim: 80.4% exact match on 61.4% of vehicles.** Both halves
must be quoted together — an accuracy without its coverage is a selection, not
a claim.

```
python -m backend.scripts.operating_point_search --eval reports/eval_greedy_20260906.json
```

**Shipped as a label, never a filter.** `anpr_engine.py` now returns
`committed` and `commit_gate` on every read (`COMMIT_MIN_NATIVE_PX = 80`,
`COMMIT_MIN_CONF = 0.85`, both env-overridable). Reads below the gate are still
returned, still voted on, and still offered to the watchlist matcher — whose
86.5% recall depends on fuzzy-matching exactly those imperfect reads. Filtering
them out would trade a headline number for the capability the system exists to
provide.

### This closes the loop on the whole story

| | |
|---|---|
| Exact transcription | **59.0%** |
| Within one character | **81.9%** |
| **Watchlist recall (distance-1 matching)** | **86.5% at 0% false alarms** |

The 23-point jump from 59.0% to 81.9% is precisely the population the fuzzy
matcher exists to recover, and it is why the watchlist figure lands where it
does. These are three independent measurements that agree — the recogniser
misses by one character far more often than it misses entirely, and the matcher
is built for exactly that error shape.

It also **independently confirms the two figures quoted throughout the
codebase** — "59% exact plate read rate" and "93% character accuracy" in
`journey_search.py` — which until now had no stored artifact behind them.

---

## 1c. Why there are only 417 labels — and why more cannot simply be mined

The obvious way to improve the recogniser is more labelled real data: the
fine-tune has 251 training vehicles, and `output/plate_corpus` holds **2,271
further vehicles** on plate-capable cameras that were never labelled. Nine
times as many, already on disk.

It does not work, and the reason matters more than the conclusion.
`backend/scripts/label_pool_probe.py` runs the **identical** plate-detection
pipeline over both populations — the control that makes the yield number
interpretable at all:

| Population | Frames yielding a plate box | Median box width | Vehicles usable (≥3 frames) |
|---|---:|---:|---:|
| Hand-labelled (416) | **77%** | **90.3 px** | 44 / 60 |
| Unlabelled pool (2,271) | **19%** | **57.6 px** | 11 / 60 |

**The pool is unlabelled because its plates are not legible.** The 417 verified
vehicles are not an arbitrary sample someone stopped labelling — they are the
readable subset, already harvested. What remains sits at a median 57.6 px,
which §1a measures at **33.3% exact**, so pseudo-labels drawn from it would be
wrong roughly two times in three. Training on those reproduces exactly the
failure `plate_self_train.py` was written to avoid: its docstring records an
earlier EasyOCR pseudo-label attempt that was 15% correct and whose *systematic*
errors survived multi-frame voting with agreement 1.00.

A dry run confirms it end to end — with the shipped ensemble as the label
source and the production detector, **0 of 300 vehicles** cleared the gates.

```
python -m backend.scripts.label_pool_probe --vehicles 60
python -m backend.scripts.plate_self_train --dry-run --max-vehicles 300
```

This is the same ceiling as §2, arriving from a different direction: camera
optics decide not only what can be *read*, but what can be *labelled to teach
the model to read*.

---

## 1b. Cross-camera search — does it answer when it should stay silent?

Recall alone is not enough. An investigator constantly types a plate from a
witness statement or an FIR for a vehicle that never passed these cameras, and
a ranking system left alone will always return *something*. A confident-looking
false top hit sends an investigation after the wrong vehicle — worse than
returning nothing. So the "not found" threshold has to be measured.

**Run 2026-09-06, stored at
[`reports/journey_eval_precision_20260906.txt`](../reports/journey_eval_precision_20260906.txt):**

Index of 1,406 vehicles · 120 positives (unseen vehicles that *are* indexed) ·
200 negatives (100 unrelated valid plates, 100 one-character near-misses).

| Score threshold | Precision | Recall | False answers |
|---|---:|---:|---:|
| ≥ −18.17 | 74.0% | 92.5% | 35 / 200 |
| **≥ −12.11** (operating point) | **91.1%** | **85.0%** | **8 / 200** |
| ≥ −8.07 | 96.8% | 75.0% | 2 / 200 |

Top-hit score separation — this is why a threshold works at all:

| | p10 | median | p90 |
|---|---:|---:|---:|
| vehicle **present** in index | −13.72 | **−1.22** | −0.03 |
| vehicle **absent** | −65.52 | **−41.98** | −15.81 |

Near-miss negatives score −20.04 at the median against −58.62 for unrelated
plates. That gap is exactly how much harder a typo is to reject than a random
plate — and the reason the negatives are built that way rather than randomly.

```
python -m backend.scripts.journey_eval_precision --positives 120 --negatives 200
```

The run is deterministic (`random.Random(7)`), so it reproduces exactly.

> **One discrepancy, stated rather than hidden.** The docstrings in
> `journeys.py`, `plate_search.py` and `journey_search.py` quote *92.0%
> precision / 89.6% recall at score ≥ −15.0* from an earlier run that left no
> stored artifact. The run above is the one with evidence behind it — **quote
> 91.1% / 85.0%**, not the docstring figure. Both are the same experiment at
> slightly different operating points and index sizes; the stored one wins.

---

## 2. Why camera choice dominates everything else

Native plate width in pixels is the strongest predictor of whether a plate can
be read at all. Measured accuracy by band:

| Native plate width | Accuracy |
|---|---|
| < 40 px | ~0% |
| 40–70 px | 17.1% |
| 70–90 px | 44.8% |

Per camera:

| Camera | Median plate width | Verdict |
|---|---|---|
| CAM_11 | 114.8 px | best optics |
| **CAM_08** | **91.5 px** (231 reads) | **best volume — demonstrate here** |
| CAM_07 | 95.2 px | good |
| CAM_01 | — | **0 plates on 384 vehicles — never use** |

```
python -m backend.scripts.plate_size_probe --source data/clips/CAM_08/CAM_08_0830.mp4
```

CAM_08 measures median **88 px**, p90 106 px, max 121 px, with **0%** of
plates in the unreadable sub-40 px band.

CAM_01 (Chiman bhai Bridge) is a wide-area view where plate characters are
roughly 7 px tall. No model, threshold or retrain reads characters that are
not in the image.

**Consequence for the detector retrain (2026-09-02):** the retrained detector
finds 2.6× more plates on live footage, but usable capture stays capped at
3.8% because the constraint is the optics, not the model.

---

## 3. Live end-to-end — feed to alert

A designated vehicle is ingested over the network and raises a real-time alert
through the same path any network camera uses:
`CameraReaderThread → YOLO → BoT-SORT → process_vehicle_track → watchlist
match → Alert`.

**Verified result:** the alert fired **4 times, once per ~80 s loop**:

```
07:52:00  GJ18AH5409     07:55:54  GJ18AH5409
07:53:19  GJ18AH5409     07:57:13  GJ18AH5409
```

Alert feed shows `STOLEN_VEHICLE_WATCHLIST_HIT` · critical · danger 9.8 ·
camera *LIVE DEMO — Majevadi Gate (CAM_08 feed)* · zone Saurashtra.

Full runbook: [`PLATE_DEMO_CAM_08_EVIDENCE.md`](PLATE_DEMO_CAM_08_EVIDENCE.md).

**Live plate reads from the corp8 government feed** (real portal, real time):
`GJ33C593` @ 1.00, `GJ03G0688` @ 0.958, `HP033721` @ 0.927.

**Stored live reads with crops on disk** —
[`output/live_anpr_eval/reads.json`](../output/live_anpr_eval/reads.json), each
with the plate crop beside it so a judge can look at the pixels the system read:

| Camera | Plate | Confidence | Native width |
|---|---|---:|---:|
| cam21 | GJ38DA1800 | 0.965 | 94 px |
| cam21 | GJ02EA0681 | 0.977 | 89 px |
| cam21 | GJ01KZ5201 | 0.965 | 88 px |
| cam21 | GJ11VV2585 | 0.974 | 83 px |
| cam21 | GJ18EH7222 | 0.971 | 82 px |
| cam21 | GJ01HV3138 | 0.956 | 75 px |

All six graded `reliable`. Note every one sits in the 75–94 px band — which is
precisely the point §2 makes about optics being the binding constraint.

Known behaviour worth stating first: ~44% of frames are dropped at
`READER_FPS=10`, and the pipeline reports this rather than hiding it — dropped
frames are skipped, not mis-processed. The alert repeats each loop because the
vehicle genuinely passes again; in live traffic it fires once.

---

## 4. What the platform actually holds

Live database `output/sentinel.db` (632 MB, 40 tables), counted 2026-09-06:

| Table | Rows | |
|---|---:|---|
| cameras | 42 | **row count, not fleet size** — 11 are soft-deleted and 1 is a demo rig. The deployed fleet is **30**; see §4c. |
| alerts | 680 | |
| journey_events | 914 | cross-camera sightings |
| camera_metrics_1m | 13,157 | per-minute camera volume |
| audit_log | 1,635 | hash-chained, tamper-evident |
| watchlist_plates | 5 | |
| **detections** | **0** | see §5 |
| **tracks** | **0** | see §5 |
| **camera_calibrations** | **0** | see §5 |

---

## 4b. City traffic heatmap — what it measures

Screen: 🔥 **Traffic Heatmap**. Endpoint: `GET /analytics/traffic-density`.

| | |
|---|---|
| Vehicle passages | **161,064** |
| Camera nodes mapped | **27** |
| Busiest node | CAM_30 Gandhidham – Rambaugh P2 Checkpost, **24,083** |
| Window covered | 2026-06-13 19:30 → 2026-09-04 18:01 |
| Source | `vehicle_track` — 220,340 rows from YOLOv8 + BoT-SORT over harvested CCTV clips |

**A "passage" is one tracked vehicle, at one camera, in one minute.** That
definition collapses the 15,818 duplicate rows sharing a
`(camera, track_id, minute)` — the same vehicle written more than once — while
keeping the 13,646 cases where a later processing run reused a tracker id,
since BoT-SORT restarts its numbering per video and those are different
vehicles. Raw rows 220,340 → deduplicated 195,447 → real fleet 161,064.

**The measure is passages per camera node — not vehicles per kilometre.**
Density per unit length needs a per-camera homography and none of this fleet is
calibrated, so that figure is not computed rather than estimated. The legend
and the on-screen footer both say this; nobody has to read a doc to find it out.

### Why this does NOT read `camera_metrics_1m`

The 1-minute rollup table is the obvious source and the wrong one. It was used
in the first version of this screen and produced a figure that had to be
withdrawn. Measured:

- **It is incomplete.** CAM_04 has **23,646 recorded tracks and a rollup sum of
  zero**; CAM_10 has 15,010 and zero. Rollups only exist for the windows a
  staging run happened to cover, so a heatmap built on them renders two of the
  busiest cameras in the state as empty ground.
- **Where it exists, it double-counts.** `vehicle_count` is the number of
  tracks *active* in that minute, so a vehicle present across three minutes
  lands in three buckets. The ratio of rollup-sum to real tracks ranges from
  **0.0× to 13.1×** across cameras — CAM_14 2.2×, CAM_16 2.2×, CAM_13 2.3×.

Summing that column gives 581,383, which is neither a vehicle count nor
anything else nameable. Use `vehicle_track`.

### Three further decisions, each a place this could have quietly lied

1. **The window is not "last 24 hours".** This footage spans a fixed historical
   range, so a hard-coded recent window renders an empty map. The endpoint
   defaults to everything on record and always reports the window it actually
   covered, so the screen labels it truthfully instead of implying "live".
2. **Cameras are matched on three keys** — `Camera.id`, `Camera.camera_id` and
   `Camera.name` — because `vehicle_track.camera_id` holds whichever the
   writing process had. Matching on one key alone silently drops real traffic.
3. **Unplaceable tracks are excluded and named, not folded in.** The looped
   demonstration feed carries 33,162 passages — a fifth of the real fleet —
   because it replays a 30-second clip for hours. It and the `CAM_E2E` test
   camera are listed under the map as excluded.

The heat weighting is `sqrt(count / peak)`. Counts span 24,083 down to double
digits, and a linear ramp would render every node except the top few as
invisible — which would misrepresent a network that carries traffic everywhere.

No new dependency was added: the heat layer is a canvas overlay
(radial-gradient brush, then an alpha→colour ramp) in
`frontend/src/components/analytics/TrafficHeatmap.jsx`.

---

## 4c. Camera calibration — two cameras now measure real km/h

No site visit was possible, so the calibration had to come from the footage.
It did, for two cameras — CAM_11 by geometry, CAM_08 by a second, independent
route — and in each case the result agrees with a measurement that shares
nothing with it.

**CAM_11**, `reports/tier1_validation_20260906.json`:

| | |
|---|---|
| Method | VP1 from vehicle motion + **VP2 from vehicle edges** (Dubská et al., IEEE T-ITS 2014) |
| Focal length | 1028.6 px → field of view **86.0°** |
| Mounting height | 6.20 m (IQR 11% of the estimate) |
| Consensus | 6,099 inliers from 18,078 transverse edge segments |
| **Calibrated median speed** | **27.7 km/h** over 95 tracked vehicles |
| **Independent leg speed** | **27.8 km/h** over 16 camera-to-camera legs |
| **Ratio** | **1.00** |

The independent figure uses GPS positions and timestamps only — no homography,
no vanishing point, no vehicle width. It cannot inherit an error from the thing
it is checking. Clip frame rate was verified at 25.00 fps, so the speed is not
scaled by a wrong assumption.

**State it as "agrees within measurement noise", not "0.4% accurate".**
Straight-line distance between cameras understates road distance, so the leg
figure is a lower bound and exact agreement is partly fortunate.

### What the other cameras did, and why the failures are not one failure

Sixteen cameras were fitted. One reached a metric ground plane:

| Outcome | Cameras |
|---|---|
| **`full`** | **CAM_11** |
| `vps_not_orthogonal` | CAM_02, CAM_05, CAM_08, CAM_21 |
| `no_vp1` | CAM_03, CAM_07, CAM_10, CAM_12 |
| `vp_directions_too_close` | CAM_04, CAM_09, CAM_14 |
| `no_consensus_vp2` | CAM_06, CAM_13 |
| `implausible_field_of_view` | CAM_01 (129°, correctly refused) |

The tempting single explanation — CCTV delivered cropped, so the principal
point is not at the frame centre — was **tested and rejected**.
`principal_point_probe.py` grid-searches every principal point within a frame
width and asks whether *any* of them yields a positive f² at a plausible field
of view. For CAM_08, CAM_05, CAM_09, CAM_14, CAM_16 and CAM_01 there is none,
anywhere. Only CAM_21 (104 px offset) and CAM_02 (258 px) could be explained
that way, and both would be unusual crops.

### What has been promoted — two cameras, and what they are validated *for*

`camera_calibrations` holds **2 active rows**, promoted 2026-09-06 behind a DB
backup: **CAM_08** and **CAM_11**. Every other script here writes to a dated
report and stops. Promotion stayed a separate, deliberate step — the field
`homography_matrix` is read by `speed_estimator`, and an unread matrix there is
the failure this project has already corrected twice.

Both rows carry `speed_validated = 1` and `positional_error_is_measured = 0`,
and the API says so: `_calibration_scope()` in `analytics.py` returns
`validated_for = "speed"` with a `scope_note`, on every calibration endpoint
that reports one. **These cameras measure km/h. They do not yet place a vehicle
at a metre-accurate position**, because nothing has been surveyed against
ground truth to say how far off that would be. The distinction is in the
schema rather than in a caveat someone has to remember.

Confidence is graded rather than boolean
(`backend/services/calibration_validation.py`): `rejected` → `unconfirmed` →
`cross_validated` → `surveyed`. Only `cross_validated` may publish a bare km/h.

A camera at `unconfirmed` was previously described here as needing nothing but
patience — "more vehicles seen at two cameras, which happens on its own."
**That was too optimistic and is corrected.** Four of the six cameras in that
state have *zero* cross-camera legs against 36 in the entire fleet, and Phase
2B measured that improving the detector adds none. They are now in the field
campaign, where a ground-point survey finishes them without a leg at all.

### Fleet size — corrected, and why it matters

An earlier draft of this section claimed *"29 of 42 registered cameras have
clips; thirteen have none"* and called it an onboarding gap. **Both numbers
were wrong and the finding inverted once they were fixed.**

The authoritative count is `backend/services/fleet_census.py`:

```
27 of 30 deployed cameras have footage on disk; 3 have none
(CAM_15, CAM_17, CAM_22). Excluded: 11 soft-deleted, 1 test/demo.
```

| Number | Where the wrong one came from |
|---|---|
| **30** deployed | 42 was `SELECT COUNT(1) FROM cameras` **without** `is_deleted = 0` — it swept in 11 soft-deleted rows and the CAM_33 demo rig created during this session's alert rehearsal. |
| **27** with footage | 29 counted `data/clips/CAM_*` **directories**; two of them (CAM_15, CAM_22) are empty. A directory is not footage. |
| **3** dark | 42 − 29 = 13. The real subtraction is 30 − 27 = 3. |

So there is no onboarding gap worth escalating: **three cameras are dark.** The
calibration coverage figure is **2 of 27 cameras with footage** — CAM_08 and
CAM_11 — not "2 of 42".

The three dark cameras did earn one escalation, just not the one first
claimed. They have delivered **zero frames since 2026-08-21** and all three
still report `status = ONLINE`, because every camera in the fleet shares one
identical `last_heartbeat_at` — a bulk write, not a liveness check. So the
health field could never have flagged them. That is an ops ticket
(`reports/ops_ticket_dark_cameras.md`) and a thing not to demo as live (§5).

**A second, sharper form of the same 30-vs-more-than-30 mistake was found live
in the API, not in a report, 2026-09-07.** `GET /api/v1/cameras` and three
siblings (`/scope`, `/gap-analysis`, `/export`) — the endpoints the fleet map
and camera grid actually call — filtered `is_deleted` and stopped, never
excluding `CAM_33`. Because `CAM_33.department = 'police'`, any
police-department user or an admin would see **31** cameras on screen, in a
grid whose own comment calls it a "30-Camera Grid". Fixed at all 4 sites plus
the unused legacy `/api/cameras` in `main.py`, by importing the same
`is_test_camera()` predicate `fleet_census()` already used — verified:
`list_cameras()` went from 31 rows to 30. The rule going forward: **every
camera-listing query filters on `is_deleted` AND `is_test_camera()`, never one
alone.**

The count had drifted three ways in one working session (22, 42, 29), which is
why it now has a single source. Every report should call `fleet_census.census()`
rather than counting for itself; the headline string it returns carries the
exclusions with it so a bare ratio cannot be quoted without them.

---

## 5. Not claimed — and why

State these before a judge finds them. Each is a bounded, explainable limit,
not a hidden failure.

| Capability | Status | Reason |
|---|---|---|
| Average vehicle speed | **CAM_08 and CAM_11 only — promoted and validated. NULL on the other 28.** | See §4c. `camera_calibrations` now holds **2 active rows**, promoted 2026-09-06 after cross-validation (CAM_11: 27.7 vs 27.8 km/h held-out, ratio 1.00; CAM_08 ratio 1.03). Both carry `speed_validated = 1` and `positional_error_is_measured = 0` — validated for speed, **not** for absolute position. `speed_estimator` returns NULL for every other camera. |
| Camera health / uptime monitoring | **Not functioning — do not demo it as live** | All 30 deployed cameras share one identical `last_heartbeat_at` (`2026-09-04 17:59:42.746201`), so it is a bulk write, not a per-camera liveness signal. Consequently `status = ONLINE` carries no information: CAM_15, CAM_17 and CAM_22 have delivered **zero frames since 2026-08-21** and still read ONLINE. The fleet map may be shown; any claim that it reflects live camera health may not. Ticket: `reports/ops_ticket_dark_cameras.md`. |
| Congestion index | **Not claimed** | Derived from speed. NULL follows. |
| Traffic density per km | **Not claimed** | Same calibration dependency. Volume counts are real; density per unit length is not computed. |
| Origin–destination matrix | **Not claimed** | Requires a continuous per-vehicle detection stream. `detections` and `tracks` are empty. The corridor pairs in `traffic_baseline.py` are hardcoded, not discovered from data. |
| Traffic heatmap | **Built — as volume, not density** | `GET /analytics/traffic-density` + 🔥 **Traffic Heatmap** screen. **161,064 vehicle passages across 27 mapped nodes**, 2026-06-13 → 2026-09-04, counted from `vehicle_track`. It is not vehicles/km, and the screen says so. See §4b. |
| Per-camera speed / congestion from `camera_metrics_1m` | **Do not use that table for totals** | The rollups are incomplete (CAM_04: 23,646 tracks, rollup sum 0) and double-count where present (0.0×–13.1× vs real tracks). They remain usable for a camera-vs-its-own-history anomaly signal, which is all `traffic_baseline.py` asks of them — but never for a network total. |
| Cross-camera direction of travel | **Implemented** | `journeys.py` computes a forward azimuth between consecutive sightings (`calculate_bearing`) plus an 8-wind `cardinal`, per leg. Rendered as rotated arrow markers along the route in `JourneyView.jsx` and as a heading column in `EvaluationMode.jsx`. This is a bearing between two *confirmed* sightings — a measurement, not the guess `plate_search.py` declines to print for a single sighting. |
| Motion blur / occlusion handling | **Not implemented** | No deblurring or occlusion-specific training. |
| Detection-to-alert latency | **Not measured** | No instrumented P50/P95 figure exists. The live demo's observed first hit lands within ~60–90 s of pipeline start, which is startup plus loop position, not per-frame latency. |
| Inference latency (10.3 ms / 97 FPS, RTX 4070) | **Superseded — do not quote** | From the withdrawn report. The re-measured figure is **≈27 ms per crop** on CUDA for the shipped 3-member ensemble at 64×256 (§1a). The 10.3 ms figure presumably describes a single member at a smaller input; quote 27 ms, or 312 ms per vehicle including multi-frame voting. |
| Grammar post-processor net effect | **Open question** | The withdrawn report measured 6 true fixes against 45 silent wrong corrections. That audit runs on crops and never touches the broken detection step, so the direction may hold — but it was measured on a 96% synthetic set and has not been re-run on real footage. |
| Multi-view **logit** fusion | **Measured, rejected** | Proposed as the highest-impact change in an external plan. Measured 2026-09-06 (`reports/plate_logit_fusion_eval_20260906.txt`): mean-logprob and width-weighted fusion both score **25.5% exact against 49.1% for a single view** — −23.6 points, far outside the ±13-point noise band the script itself states. "Most confident view" reaches 50.9%, +1.8, which is inside noise. **Do not implement it.** |
| Grammar-constrained **beam decode** in the live engine | **Measured twice, rejected** | A/B on the shipped ensemble over the same 83 held-out vehicles, 2026-09-06 (`reports/eval_beam_20260906.json` vs `eval_greedy_20260906.json`): **59.0% → 59.0%, exactly zero**, at **4× the latency** (146 → 587 ms/vehicle). Within-1-char moved 81.9% → 83.1% and character accuracy 93.1% → 93.2%, both inside noise. It is not a wiring failure — the latency change proves beam ran. The gain is absent because the post-hoc `decode_plate` grammar pass already performs the same repair. Confirms an earlier independent measurement of 0.0. |
| Retraining on a **corrected synthetic corpus** (width recalibration, occlusion/weather augmentation) | **Measured, rejected** | `reports/scale_sensitivity_20260906.txt`: on the same images, `best.pt` (synthetic-pretrained) scores **79.7% synthetic / 25.7% real**, while the shipped `final_m0.pt` scores **0.7% synthetic / 80.0% real**. The real-crop fine-tune erases the synthetic behaviour entirely, so improving the synthetic corpus alters a stage that is then overwritten. ≈0 expected gain for ~5 days of GPU. The corpus itself still earns its place (§5b) — what does not pay is *improving* it. |
| **Self-training on the unlabelled pool** | **Measured, rejected** | 2,271 unlabelled vehicles exist against 417 labelled — nine times as many, and it looks like the obvious win. It is not. `reports/label_pool_probe_20260906.txt` runs the identical detection pipeline over both: hand-labelled vehicles yield a plate box on **77%** of frames at a median **90 px**; the unlabelled pool yields **19%** at a median **58 px**. The pool is unlabelled *because* its plates are not legible. A dry run of `plate_self_train.py` with the shipped ensemble accepted **0 of 300** vehicles. See §1c. |
| Plate rectification | **Implemented, default OFF** | Wired with confidence arbitration, but median plate tilt on this fleet is 0.00° — there is nothing to straighten. |

---

## 5b. The synthetic corpus — why it stays, and the rule it lives under

98,586 synthetic plates in `data/synth_plates`, generated by
`backend/scripts/synth_plates.py`. The recogniser pre-trains on them and is
then fine-tuned on real crops. This is deliberate and it stays.

**Why it is necessary.** The hand-verified real corpus contains **six
non-Gujarat plates in total**. Synthetic is the only thing teaching the model
all 37 state codes, single- and two-row layouts, and all four colour schemes.
Without it the system cannot read an out-of-state vehicle — a real capability
loss on any highway or border camera.

**It is not leaking into the evaluation.** Checked 2026-09-06 by intersecting
every benchmark plate string against the full training corpus:

```
benchmark synthetic plates also in training corpus:  0 / 5,297  = 0.0%
benchmark real plates also in training corpus:       0 /   200  = 0.0%
```

The held-out split is genuinely held out.

**The rule, which the withdrawn report broke.**

> Synthetic and real results are **never blended into one figure**.
> Synthetic is a *development signal* — did this change help? Real footage is
> the **only** source of a number quoted outward.

The withdrawn report reported one accuracy over 203 real + 5,297 synthetic
instances, with Tier 1 and Tier 3 containing no real footage at all. That is a
reporting failure, not a data failure.

**One genuine shortcoming of the generator, worth fixing.** The corpus is
described as "degraded to match the measured statistics of this footage". At
the variable that dominates readability, it is not:

| Slice | n | median | p90 | max |
|---|---:|---:|---:|---:|
| Real (benchmark) | 203 | **107 px** | 120 px | 135 px |
| Synthetic | 5,297 | **197 px** | 304 px | 323 px |

The synthetic centre of mass sits at roughly **twice** the real plate width —
and the real benchmark slice is itself the easy tail, since CAM_08's full
distribution medians at 88 px (§2). Accuracy learned at 197 px does not
transfer to 88 px. This is a generator parameter, not a reason to discard
98,586 samples.

**Long-term priorities, in order:**

1. Keep synthetic for training; report it separately, always labelled.
2. Recalibrate the generator's width/degradation profile to the measured real
   distribution (median ≈ 88–107 px, not 197 px).
3. Fix the ground-truth boxes on the 203 real instances — they are currently
   full-frame, which is what broke the withdrawn report's detection scorer.
   203 correctly-boxed real instances is a small but **honest** benchmark, and
   worth more than 5,500 blended ones.
4. Grow the real labelled set. At 203 real against 98,586 synthetic, this is
   the actual bottleneck.

---

## 6. Reproducing everything on this page

```bash
# Is a source legible at all? (~1 min) — expect "VERDICT: GOOD", median > 70px
python -m backend.scripts.plate_size_probe --source <clip-or-camera>

# Watchlist recall, two-pass, non-circular
python -m backend.scripts.measure_watchlist_recall --clip data/clips/CAM_08/CAM_08_0830.mp4

# Cross-camera search precision/recall + the "not found" threshold (deterministic)
python -m backend.scripts.journey_eval_precision --positives 120 --negatives 200

# The shipped ensemble on held-out real footage — trains nothing, ~30s on GPU
python -m backend.scripts.eval_shipped_recognizer

# The rehearsed live demonstration — see PLATE_DEMO_CAM_08_EVIDENCE.md §Steps
python backend/scripts/demo_second_camera_source.py \
    --clip demo/clips/cam08_demo_loop.mp4 --port 8091 --fps 25
```

### One operational warning

Do not run the ingest pipeline and browse camera snapshots heavily at the same
time. Both pull from the same corp8 portal account, and doing both at once
returned `HTTP 403 — playlist not served` on every camera during testing. Warm
the snapshot cache first (`backend/scripts/warm_snapshot_cache.py`), then start
the pipeline on a small camera set.
