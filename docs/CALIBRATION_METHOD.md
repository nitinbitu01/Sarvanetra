# Camera calibration without a site visit — method, and why it is built this way

This deployment has no survey team, no ground control points, and its cameras
are hundreds of kilometres away. Calibration therefore has to come from the
footage and from the network's own behaviour. It can, for some cameras.

This document is the **why**. The scripts say what they do; the reasoning
below is what stops someone — including whoever wrote it, three months on —
from re-introducing a defect that has already been found and removed once.

---

## What a calibration is for

`speed_estimator` turns pixel motion into km/h using a ground-plane homography
stored in `camera_calibrations.homography_matrix`. Without it, speed is NULL,
and with it wrong, speed is confidently wrong. Congestion index, traffic
density and wrong-way detection all sit downstream of the same number.

That is the whole reason for the discipline that follows.

---

## The two routes, and where each one breaks

### Route A — geometry only (`fit_vp2_from_edges.py`)

A vanishing point from vehicle motion (VP1) plus a second, orthogonal one
(VP2) determines focal length through `f² = −(vp1−c)·(vp2−c)`. VP2 classically
comes from a second stream of traffic crossing the first. **Most of this fleet
watches one-directional roads, so there is no second stream.**

Dubská, Herout & Sochor (IEEE T-ITS 2014) take VP2 from the vehicles instead:
a car is a box, and its bumper line, roofline and window frames run *across*
the direction of travel. Accumulated over thousands of vehicles those edges
converge on the transverse vanishing point.

**Result: 1 of 16 cameras (CAM_11).** The rest fail, and they fail in six
distinguishable ways — see the failure taxonomy below.

### Route B — the network calibrates itself (`fit_focal_from_legs.py`)

With VP1 and a camera height:

```
X = h (u − vpx) / dv      lateral    — no f in it
Y = f h / dv              along road — one unknown
```

`f` is a single scalar. **One independent speed measurement determines it**, and
the network already produces those: a plate read at two cameras gives distance
over time from GPS and clocks alone, touching no camera model. A geometric
constraint is traded for a data-driven one — and unlike a survey, it improves
on its own as traffic accumulates.

**Result: CAM_08 and CAM_11.** And on CAM_11, where both routes work, they
agree: **974.4 px from legs against 1028.6 px from geometry, ratio 0.947**,
with the geometric value inside the leg method's confidence interval. Two
methods sharing no input arriving at the same answer is the strongest evidence
available here.

---

## Fourteen things that are non-obvious, and why each is there

### 1. Speed is **not** linear in `f`

`d(f) = √(dX² + f²·dY1²)` — that is `√(a + b f²)`, not a line. It is monotonic,
so it is solved by **bisection**, not fitted by regression. On a camera aimed
along the road `dX` is small and the relation looks nearly linear; "nearly" is
not a reason to use a model that is wrong at the edges.

### 2. Bootstrap the solve. Never report a point estimate.

Two legs at 141 and 147 km/h — two different vehicles read as one plate —
were, on their own, moving CAM_08's focal length from 1087 px to 1190 px and
widening its interval from 13% of the estimate to **153%**.

A point estimate would have reported 1190 px with nothing to suggest that two
rows were carrying it. The confidence interval is what made the corruption
visible. **This is the single most valuable guard in the pipeline.**

### 3. A leg over a long link is not a proxy for speed at the camera

Keying legs by corridor immediately exposed two that per-camera pooling had
hidden:

| corridor | distance | implied speed |
|---|---:|---:|
| CAM_08 → CAM_07 | 69,896 m | 65 km/h |
| CAM_21 → CAM_08 | **311,456 m** | 20 km/h (15.5 hours) |

The second is not a journey — it is an OCR collision 311 km apart. The first is
a real trip and still useless: over 70 km the average is dominated by whatever
happened in between, not by the speed at CAM_08.

`MAX_LEG_METRES = 5000`. The useful CAM_11 → CAM_08 corridor is 2.4 km.

### 4. Corridor is the primary key, camera is a view

"Median leg speed at CAM_08" is not a property of CAM_08 — it is an artefact of
which corridor contributed more legs that week. Before the distance cap,
CAM_08's legs split into ~26 km/h (urban) and ~65 km/h (highway) and its
interval blew out accordingly.

`corridor_legs.py` keys on the **directed** camera pair. Direction is kept
because A→B and B→A occasionally differ — a gradient, a signal on one
approach — and collapsing them hides exactly that.

`fit_camera` refuses any camera whose well-sampled corridors disagree by more
than 1.5× (`corridors_disagree`) rather than averaging them silently.

### 5. Removing data must be justified by held-out evidence, not by a nicer number

When the long legs were removed, CAM_08's **held-out** ratio went 0.87 → 1.03.
The held-out legs are not influenced by the ones removed, so their agreement
improving is independent evidence the removal was correct rather than
convenient. **If a cleanup only tightens the fit and does nothing for the
held-out set, be suspicious of it.**

The confidence interval also widened slightly as legs fell — the honest
trade-off of cleaner data with less of it.

### 6. Fit and validate on different legs

Otherwise the method confirms itself. Split is 65/35 with at least three held
out.

### 7. `f` is pinned by tracks at *different* distances

If every track sits at a similar `dv`, the fit is ill-conditioned however many
legs exist. The `dv` p10/p50/p90 spread is measured and a camera below 1.5× is
refused. Held-out legs clustered at one extreme are extrapolation, not
validation.

### 8. Rejecting motion-aligned edges must be **angular**, not a pixel distance

The first version asked whether a segment's line passed within 60 px of VP1.
Correct question, wrong threshold: CAM_21's VP1 sits 434 px *outside* the
frame, so an edge aligned to within two degrees still misses it by hundreds of
pixels. The filter rejected 4% of segments, the road-aligned majority survived,
and RANSAC found *them* — returning a "second" vanishing point 1.2° from the
first.

Measuring the angle between the segment and the direction to VP1 removes the
dependence on how far away VP1 happens to be. **Separation went 1.2° → 63°.**

### 9. VP2 must be constrained to VP1's horizon row

Two ground-plane directions both vanish on the horizon; for a camera with
negligible roll — the assumption the height estimator already makes — that is
one image row. Unconstrained, RANSAC found 3,192 inliers on a point 494 px
above the centre while VP1 sat 974 px above it: two different rows, so not two
ground directions. It had locked onto **vertical** structure (vehicle sides,
poles), whose vanishing point is the third one.

**Separation went 63° → 85.9°** once constrained.

### 10. `cv2.HoughLinesP` returns `(N, 4)` on this build, not `(N, 1, 4)`

Silent corruption if assumed. Reshape to `(-1, 4)`.

### 11. The cropped-sensor explanation was tested and falsified

It was tempting: the orthogonality formula assumes the principal point is at
the image centre, and NVR-delivered CCTV is often cropped. `principal_point_probe.py`
grid-searches every principal point within a frame width and asks whether *any*
yields a positive `f²` at a plausible field of view.

For CAM_08, CAM_05, CAM_09, CAM_14, CAM_16 and CAM_01 there is **none,
anywhere**. Only CAM_21 (104 px offset) and CAM_02 (258 px) could be explained
that way, and both would be unusual crops. **One elegant explanation does not
get to absorb six different failures.**

### 12. Stale model paths hide in old scripts

Both `plate_self_train.py` and `journey_build_index.py` pointed at
`runs/detect/runs/plate_v3/recovered/weights/best.pt`. Production is
`models/plate_detector/plate_v4_small.pt`, retrained 2026-09-02, which finds
**2.6× more plates**. The journey index is the sole source of legs, so a
detector that misses plates there starves calibration *and* its validation.
Check the weights path before trusting any script that has not been run
recently.

### 13. Never count the fleet inline

The fleet size was stated three ways in one session — 22, 42, 29 — and a
finding was built on the wrong one and had to be withdrawn ("13 cameras have no
footage" collapsed to **3**). `SELECT COUNT(1) FROM cameras` returns 42, which
includes 11 soft-deleted rows and a demo rig. Counting `data/clips/CAM_*`
directories returns 29, two of which are empty.

Call `backend/services/fleet_census.py::census()` and quote its `headline()`,
which carries the exclusions with it. **The fleet is 30; 27 have footage.**

### 14. Nothing is promoted without being read

Every fitter writes a dated report and stops. Promotion is a separate script
(`promote_calibrations.py`) that defaults to a dry run and takes a database
backup before writing. Confidence is graded — `rejected` → `unconfirmed` →
`cross_validated` → `surveyed` — and only `cross_validated` may publish a bare
km/h; `unconfirmed` carries its uncertainty. A camera at `unconfirmed` needs no
rework, only more traffic, and upgrades on the next run.

**Promoted 2026-09-07 after sign-off: CAM_08 and CAM_11.** Every other camera
still returns NULL, which remains correct.

On the `reprojection_error_m` column, which is NOT NULL and means positional
error in metres: there are no ground control points on this fleet, so a
reprojection error against surveyed points **does not exist**. What is written
is the focal interval's relative width applied at the camera's own median
observed range — ~6 m at ~39 m — recorded in `notes` as *inferred*, not
measured. `quality_gate` is `degraded` and never `good`, because that gate
means sub-metre against real points.

The distinction matters: these calibrations are **validated for speed**
(held-out ratios 1.03 and 0.94) and **not established for position**. Anything
that uses them to place an object on a map rather than to measure how fast it
was going is outside what was tested.

### 15. Improving the detector does **not** produce more legs — measured

The obvious lever for the leg shortage is better plate detection. It was tried:
`journey_build_index.py` was pointed at the production detector
(`plate_v4_small.pt`, which finds 2.6× more plates on live footage) instead of
the stale `plate_v3` checkpoint, and re-run over all 5,596 corpus vehicles.

| | before | after |
|---|---:|---:|
| index records | 1,406 | 1,452 (+3.3%) |
| journey_events | 914 | **914** |
| usable legs | 19 | **19** |

**No change.** 4,144 of 5,596 vehicles (74%) have no readable plate at all. The
corpus crops are already the ones where a plate was locatable, so a better
detector has nothing left to find. A leg needs a plate readable at *two*
cameras — the product of two already-low probabilities.

**Do not spend more effort on detection to unblock calibration.** The
constraint is camera optics and coverage, and it is the same ceiling that caps
recognition accuracy. See `docs/MEASURED_EVIDENCE.md` §2.

### 16. A rescue method must be regression-checked before its wins are counted

`fit_vp1_robust.py` removes the 3%-straightness pre-filter — which was
discarding 85% of tracks and leaving four cameras with no vanishing point — and
makes the estimator robust instead. The first version produced VP1 for 17
cameras and appeared to rescue four.

Checking it against the cameras whose VP1 was already known good showed it was
partly wrong:

| camera | inlier frac | direction spread | shift vs known-good |
|---|---:|---:|---:|
| CAM_06 | 0.55 | 21.7° | 20 px ✓ |
| CAM_09 | 0.66 | 28.8° | 57 px ✓ |
| CAM_08 | **0.20** | 109.7° | **329 px** |
| CAM_02 | 0.47 | 164.5° | **1,261 px** |
| CAM_13 | 0.32 | 63.6° | **3,878 px** |

Removing a pre-filter admits junk as well as usable data, and **RANSAC
tolerating outliers is not the same as RANSAC being right about a set that is
mostly outliers.** A direction spread above a right angle means the tracks are
not on one road — several flows plus association errors — and no single
vanishing point describes them however confident the consensus looks.

Gate: `MIN_INLIER_FRAC = 0.50`, `MAX_DIRECTION_SPREAD_DEG = 60`. After it,
three cameras solve and **one is newly reachable** (CAM_25 — 12 of 13 lines
agree at 10.9° spread, where only 2 would have passed the old gate). CAM_08 and
CAM_11 are now correctly *rejected* by the robust method, so it cannot
overwrite the VP1s their working calibrations rest on.

Without that check, two cross-validated cameras would have been silently
degraded and it would have been reported as progress.

**A VP1 is not a calibration.** All three cameras that gained one are still
stuck: CAM_06 has 0 legs, CAM_09 has 2, CAM_25 has no height estimate.

### 17. The strategy for the cameras this method cannot reach

Three options were on the table for the ~20 cameras still uncalibrated:

1. **VP2 from lane markings** — rejected. It is the same geometric route that
   already failed on 15 of 16 cameras, with a different edge source. The
   principal-point falsification (#11) showed the failures are not one fixable
   thing.
2. **Known-object scale as an alternative speed source** — rejected. This is
   what `estimate_height` already does, using regulated vehicle widths. It
   produced heights for six cameras and stopped; there is no second scale
   source hiding in the same footage.
3. **Mark them explicitly, with a reason code and a remedy** — adopted.

`calibration_status.py` classifies every deployed camera into one of six codes
and, more usefully, one of four **remedies**:

| remedy | meaning | cameras |
|---|---|---:|
| `done` | calibrated, speed is live | 2 |
| `wait` | clears by itself as vehicles are seen at two cameras | 6 |
| `person` | mounting height cannot come from this footage | 7 |
| `method` | this technique does not fit this camera | 12 |
| `deployment` | camera produces no clips at all | 3 |

The distinction is the point. A flat "not calibrated" list invites someone to
spend a week on the twelve that cannot move. `--check-upgrades` compares
against the previous run and reports only what changed, so the six `wait`
cameras are a scheduled re-run rather than a diary entry. Closest today:
CAM_09, six legs short.

`method` is not a permanent no — it is "not with this technique". If a
different data source ever appears, the grading in
`calibration_validation.py` lets those cameras re-enter at `unconfirmed`
without any rework.

### 18. A latent inconsistency in `quality_gate` — read this before re-running any calibration writer

`evaluate_calibration_quality()` maps a reprojection error to a gate:
good < 0.5 m, degraded < 1.0 m, **rejected above**. Those thresholds are for
positional accuracy measured against surveyed ground control points.

The promoted calibrations store `reprojection_error_m` of **6.13 m and 6.24 m**
— an inferred positional uncertainty (#14) — with `quality_gate = "degraded"`.
Passing those same numbers through `evaluate_calibration_quality()` returns
**`rejected`**.

Both are defensible and they mean different things: *by a positional standard
these are rejected; by the speed standard they were actually validated against
(held-out ratios 1.03 and 0.94) they are usable.* The single `quality_gate`
column cannot express that, and that is the real defect.

**What works today:** every consumer reads the stored `db_cal.quality_gate`,
and `estimate_track_speed` blocks only `rejected` and `uncalibrated`, so
`degraded` passes and speed is emitted.

**What would silently break it:** `analytics.py` (the calibration studio save
path) and `calibrate_pilot_cameras.py` both *recompute* the gate from the
error at write time. Re-running either against these rows would flip them to
`rejected` and speed would go NULL again with nothing in the logs to say why.

**RESOLVED 2026-09-07.** The column is now split. `camera_calibrations` carries
three additional fields, added through the same self-healing `ALTER TABLE`
pattern in `db/session.py` that the other late columns use:

| column | meaning |
|---|---|
| `positional_error_is_measured` | true only for a calibration solved against surveyed ground control points |
| `speed_validated` | true when held-out cross-camera legs agreed |
| `calibration_method` | how it was derived, so nothing has to be inferred from `calibrated_by` |

The two promoted rows are backfilled as
`positional_error_is_measured = 0, speed_validated = 1,
calibration_method = 'vp1_motion+focal_from_legs'`, and `_calibration_scope()`
now reads these flags, falling back to the `calibrated_by` check only for rows
written before the columns existed.

`quality_gate` remains as it was for compatibility, and remains a *positional*
statement. It is no longer the only thing standing between a speed-validated
camera and a writer that recomputes it — but recomputing it on these rows is
still wrong, and still turns a working camera off.

### 19. The GCP route shares none of the automatic routes' blockers — verified

This is load-bearing, because the site-measurement campaign asks for ground
points at twenty-five cameras and twelve of those were classed as unreachable
by the automatic routes. If the studio's solve secretly rested on the same
assumptions, the campaign would be asking for work that could not help them.

It does not. Checked at both levels:

```python
# backend/services/calibration.py
H, _ = cv2.findHomography(src, dst, method=0)
```

and the endpoint (`analytics.py`) calls exactly that, then
`validate_homography` and `evaluate_calibration_quality`. **No vanishing point,
no orthogonality constraint, no straight-road assumption, no camera height.**
It is a pure point correspondence between image pixels and world metres.

So every blocker that stops the automatic routes is irrelevant to it:

| blocker | why it stops VP/leg routes | does it stop GCP? |
|---|---|---|
| `no_vp1` | no consistent direction of travel to fit | **no** |
| `not_one_road` | tracks span too many directions for one VP | **no** |
| `needs_height` | width observations cannot give a pole height | **no** — height is not an input |
| `vps_not_orthogonal` | the two vanishing directions are inconsistent | **no** |
| too few legs | no independent speed to solve `f` | **no** — `f` is not solved separately |

**The limitation it does have is different: coplanarity.** A homography maps
one *plane* to another, so the chosen ground points must lie on a flat road
surface. A bend in the road is fine — plan curvature is irrelevant. A crest, a
dip, or a strongly cambered carriageway is not, because those points are not
on one plane. That is the thing to watch when choosing points, and it is not
what the automatic routes were failing on.

`method=0` also means a plain least-squares fit rather than RANSAC, which is
why the studio requires **five** points: four determine an 8-DOF homography
exactly and the residual is zero by construction, telling you nothing.

**Consequence for the status report:** those twelve cameras were labelled
`method`, which read as permanent. They are not — they are waiting on the same
site visit as the seven `person` ones. The bucket is now called `survey`.

### 20. `wait` was the wrong remedy for six more cameras — the same mistake, once removed

Finding #19 fixed the `method` bucket and stopped one step short. Six cameras
sat in a bucket called `wait`, with the note *"clears on its own as vehicles
are seen at two cameras."* The reasoning felt safe: doing nothing is free, and
the state would resolve itself. Both halves turn out to be wrong.

**It does not clear on its own.** Four of the six have **zero** legs, and the
whole fleet has produced 36. Phase 2B measured that a better detector adds
none (+3.3% index, legs unchanged). "Eventually" had no date attached to it and
no mechanism behind it — it was a hope written in the same column as a plan.

**And they were never blocked by the site.** A `needs_legs` camera has VP1 and
a height already; what it lacks is a focal length, and legs are how the *leg
route* solves for one. #19 established that the GCP route solves for no focal
length at all. So the leg count is irrelevant to it, exactly as `no_vp1` is.
The blocker belonged to the method, not to the camera — which is the identical
error as #19, one bucket over.

The cost of including them turned out to be near zero. All six are in districts
already being visited, each within **7.4 km** of a camera already on the list
(CAM_16: 0.7 km, CAM_09: 0.8 km, CAM_02: 1.4 km, CAM_21: 1.5 km, CAM_06:
2.3 km, CAM_04: 7.4 km). They add travel time, not a trip. The campaign goes
from 19 cameras to **25**, and the best next trip changes from Navsari (6) to
**Ahmedabad (8)**.

Their *ask* is smaller and the request now says so per camera: height and VP1
are on record, so **ground points only**. Sending someone up a pole for a
number already in the database is how a campaign loses the goodwill it runs on.

**The general lesson, and the reason this is written down:** a remedy of "wait"
deserves the same scrutiny as a remedy of "impossible". Both end the
conversation, and neither leaves anything to check later. The question that
catches it is *what exactly has to happen, and what is the evidence it will?*
Here the answer was 36 legs fleet-wide and a measured null result — which is an
answer, just not the one the label implied.

### 21. Speed on a surveyed camera does not need a leg either — traced end to end, and a stated claim corrected

A claim made earlier in this document's own history — and repeated verbally to
whoever was planning the traffic-analytics dashboard — was that the GCP route
gives a camera `positional_error_is_measured = true` but leaves
`speed_validated = false` until an independent cross-camera leg passes through
it. **That claim does not survive tracing the actual code path, and was
withdrawn 2026-09-07.**

Followed the chain from the endpoint to the number on screen:

```
POST /macro/calibrate                    operator submits ground points
  -> compute_homography(img_pts, world_pts)     fits H on the submitted points
  -> validate_homography(H, held_img, held_world)  tests a HELD-OUT point, not
                                                    one used in the fit
  -> evaluate_calibration_quality(err)           < 0.50 m -> "good": accepted,
                                                    by this function's own
                                                    docstring, "for full speed
                                                    & congestion products"
  -> track_recorder.py reads quality_gate directly, calls
       pixel_to_world_m(H, px, py)                same H, any pixel
  -> speed_estimator.estimate_track_speed(world_points, timestamps)
                                                    Theil-Sen over world-plane
                                                    points; needs no focal
                                                    length, no height, no legs
```

**There is no separate "speed scale" unknown for this route.** A VP/leg
calibration jointly estimates height and focal length from vehicle statistics
with no ground truth at all — that is exactly why §19–20's independent leg
check exists, as the *only* available cross-check. A GCP homography's scale
comes directly from metres someone measured in the field; a held-out point not
used in the fit is already an independent test of that same transform, and
speed is computed by feeding two world-plane points from it into a time
difference — not a second, unchecked quantity riding along on a validated
position.

**What was actually wrong, concretely:** `submit_camera_calibration()` never
wrote `positional_error_is_measured` or `speed_validated` at all — both
columns default to `False` in the model, not `NULL`, so `_calibration_scope()`
fell through its "prefer stored flags" branch and hit a pre-existing fallback
that already returned `validated_for: "position_and_speed"` for any
non-leg-derived row. The fallback and the stated claim disagreed; checking
against the running code is what caught it, not re-reading the stated claim
more carefully.

**Fix, not a reversal:** kept the behavior — a good held-out error still
grants speed immediately, no legs required — because tracing the maths says it
should. What changed is the *label*. `_calibration_scope()` now returns
`speed_confidence`: `"surveyed"` for a GCP-only camera, `"cross_validated"`
for a leg-checked one (CAM_08/CAM_11 — verified unaffected by this change).
Both may show a bare km/h; only the stronger claim gets the stronger word,
because a held-out ground point tests spatial accuracy and a cross-camera leg
additionally tests the time axis — genuinely different evidence, not a formality.

**A second, smaller gap found while tracing this:** a `rejected` gate
(held-out error > 1.00 m) still reached `is_active = True` — nothing stopped
it. `speed_estimator` happened to null the *speed* for such a camera via its
own `quality_gate` check, but nothing protected a position-only consumer from
reading a homography already known to be wrong. Closed: `rejected` now returns
unwritten, matching the rule §1 of `calibration_validation.py` already states
for the other route — nothing reaches `camera_calibrations` without passing
its own gate.

**Consequence for the dashboard plan:** the 25-camera field campaign, once
ingested with a well-spread 5+ points and a good held-out error, unlocks
position AND speed together — not position now, speed later on some cameras
and never on others. Density and flow, which read the same calibrated
coordinates, follow the same timing. The only camera-scoped upgrade path still
waiting on legs is the *confidence label* moving from `surveyed` to
`cross_validated`, not the number's existence.

### 22. The fix above landed on dead code first — the real endpoint had two further bugs, both found by actually calling it

§21's fix was applied to `submit_camera_calibration()` (`POST /macro/calibrate`)
before checking who calls it. **Nobody does** — not the frontend, not a
script, not a test. The calibration studio the field campaign will actually
use (`LiveCalibrationStudio.jsx`) posts to a completely different endpoint,
`save_camera_calibration()` (`POST /calibration/save`). Re-applied the same
`calibration_method` / `positional_error_is_measured` / `speed_validated`
write there — but calling the real endpoint end to end, rather than trusting
that the fix transferred, surfaced two more bugs the dead-code copy never
would have shown:

**A crash that turned every successful save into a shown failure.**
`save_camera_calibration()`'s response built `"quality_gate": payload.quality_gate`
— but `SaveCalibrationRequest` deliberately has no `quality_gate` field (an
earlier fix, documented in its own docstring, removed it so a client could no
longer assert its own gate). Accessing an undeclared field on a Pydantic model
raises `AttributeError`, confirmed directly:

```
>>> payload.quality_gate
AttributeError: 'SaveCalibrationRequest' object has no attribute 'quality_gate'
```

The crash happens *after* the database commit — so the calibration was
correctly written, the exception fires only while building the response,
FastAPI returns 500, and the studio's `runSave()` shows the operator a red
error notice for a save that had, in fact, succeeded. For a campaign whose
entire model is "trust the person standing at the pole to report back," this
is close to the worst possible failure mode: it does not lose data, it makes
the operator not trust the tool that has none. Fixed by returning
`solved.quality_gate` / `solved.held_out_error_m` — the server's own computed
values, which is what the surrounding code already claims to do.

**A silent BOM bug that meant the visual GCP overlay never had anything to
draw.** The same endpoint also writes each save to
`ground_control_points.json`, read back by `live_24x7_pipeline.py` to draw
green marker circles and the held-out point on the live annotated feed —
cosmetic only, no real number depends on it (those come from the DB via
`TrackRecorder`), but it is exactly what a judge watching the demo would see.
The file on disk carries a UTF-8 BOM; both the write-side read (merge with
existing entries) and the pipeline's load used plain `"utf-8"`, which raises
on a BOM — caught by a bare `except: pass` on the read side, logged-but-easy-
to-miss on the write side. Net effect, confirmed end to end: the JSON file
was never actually updated by any save, silently, on both ends. Fixed with
`encoding="utf-8-sig"` on both reads (transparently handles a BOM or its
absence) and gave the pipeline's swallow a `logger.warning` so a repeat of
this specific failure mode would at least appear in a log next time.

**None of this was found by re-reading the code more carefully — it was found
by actually calling the function**, with real geometrically-valid points, a
real DB session, and asserting on the response and the row it wrote. §19–21
each corrected a *stated claim* against the *running code*; this one corrects
a fix already believed complete against the *actual code path a user takes*.
The general lesson: verifying a fix means calling the function real users
call, not the one with the matching name.

---

## Failure taxonomy (16 cameras fitted)

| Outcome | Cameras | Meaning |
|---|---|---|
| **`full`** | CAM_11 | metric ground plane |
| `vps_not_orthogonal` | CAM_02, 05, 08, 21 | VP2 found, angle inconsistent with a centred principal point |
| `no_vp1` | CAM_03, 07, 10, 12 | no motion vanishing point at all — the straightness gate starves them |
| `vp_directions_too_close` | CAM_04, 09, 14 | VP1 and VP2 nearly parallel; `f` ill-conditioned |
| `no_consensus_vp2` | CAM_06, 13 | too few transverse edges to agree on anything |
| `implausible_field_of_view` | CAM_01 | 129° — correctly refused; a lens that wide also breaks the pinhole model |

**These are six different problems.** Treating them as one is how a wrong fix
gets applied to five cameras that did not have that fault.

---

## Pipeline

```
collect_road_geometry.py     tracks + vehicle widths, ALL clips per camera
        │                    (reading one clip per camera starved everything;
        │                     CAM_08 went 8 → 30 straight tracks when fixed)
        ├─► fit_camera_calibration.py    VP1 + pole height
        │
collect_vehicle_edges.py     line segments inside vehicle boxes
        └─► fit_vp2_from_edges.py        VP2 → focal length      [Route A]

journey_build_index.py       plate reads → journey_events
        └─► corridor_legs.py             legs, keyed by corridor
                └─► fit_focal_from_legs.py   focal from legs     [Route B]

                        ▼
        calibration_validation.py   graded gate, shared by both routes
                        ▼
        validate_tier1_speeds.py    against independent leg speeds
                        ▼
              (manual sign-off)  →  camera_calibrations
```

## The bottleneck, stated plainly

**Legs.** Only 19 survive filtering, across 5 corridors; a camera needs ≥8.
Every additional leg helps two cameras, for both calibration and validation, so
the return on more journey-index coverage is superlinear. That is why
`journey_build_index.py` running with the correct detector matters more than
any further tuning of the geometric route.
