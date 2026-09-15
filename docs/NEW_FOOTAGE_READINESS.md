# Will this work on footage it has never seen?

The evaluation runs on the judges' CCTV footage, not ours. This document answers, with
measurements rather than assurance, what survives that change and what does not — and
records the hardcoded assumptions found and removed while checking.

The honest summary in one line: **everything except plate legibility is deployment-agnostic
and now verified so; plate legibility is set by the judges' camera optics, not by this
code, and the probe in §4 tells you which case you are in within about a minute.**

---

## 1. What is genuinely not tied to our deployment (verified, not assumed)

| Concern | Finding |
|---|---|
| Feed type | `CameraReaderThread._detect_source_type` (`live_24x7_pipeline.py:275-289`) handles `rtsp://`, `rtsps://`, `http(s)://`, `onvif://`, local files (`.mp4/.m3u8/.ts/.avi/.mkv/.mov/.flv/.webm`), and folders of clips. An arbitrary RTSP URL or a dropped MP4 both work. |
| Camera discovery | `_discover_cameras()` (`live_24x7_pipeline.py:1256`) reads the **database**, not `config.yaml`. A camera added through `POST /cameras` is picked up on pipeline start — it is not limited to a baked-in list. |
| Plate format | `backend/scripts/indian_plate_grammar.py` carries **all** Indian state codes plus the special series (BH/CD/CC/UN). The Gujarat bias is a *prior* (`PRIOR_TIE_WEIGHT = 0.06`, "additive nudge to optical cost"), a tiebreaker between visually-similar reads — not a filter. A clean `MH`/`DL`/`RJ` plate stays what it is. |
| Watchlist | Now DB-backed and editable at runtime (`docs/WATCHLIST_ALERTING_ARCHITECTURE.md`). A judge's target plate is added through the UI and reaches the live matcher within ~5s. No file editing, no restart. |
| Camera count | The `count=30` / `CAM_{i:02d}` loop in `backend/cache/redis_cache.py:82` is **dead code** — no callers anywhere in the backend. Harmless. |

## 2. Hardcoded assumptions found and fixed in this pass

**Fabricated success on connecting a new camera** — `POST /analytics/live/connect-stream`
is precisely the endpoint used to attach a new feed. It returned
`"success": true`, `"is_connected": true`, `"resolution": "1920x1080"`,
`"sample_fps": 5.0` and *"Successfully connected!"* **whenever the pipeline returned no
reader at all** — because `stats` was `{}` and every field fell through to a hardcoded
default. Connecting a camera that did not work reported that it worked, with invented
specs. Now every field is measured or explicitly `null`, `success` means *attached and
receiving frames*, and the response distinguishes the three real cases: camera saved but
pipeline not running in this process / attached but not yet connected / actually
receiving. (`backend/routers/v1/analytics.py`)

**Footage ignored because of its folder name** — local-clip discovery skipped any
directory not named `CAM_*` (`live_24x7_pipeline.py:1292`). A folder of evaluator
footage dropped in under any other name was silently invisible. Now any directory
containing video files is discovered, with the extension list widened beyond `.mp4`.

**ANPR quality gates were unchangeable** — `PLATE_DET_CONF`, `SINGLE_FRAME_FLOOR_PX`,
`FUSION_FLOOR_PX`, `FUSION_MAX_NATIVE_PX` were fixed constants tuned to *our* optics.
Native plate width is a property of sensor resolution, focal length and camera distance,
so a different site legitimately lands in a different pixel band — and there was no way
to adapt without editing source. They are now environment-overridable
(`SENTINEL_PLATE_DET_CONF`, `SENTINEL_SINGLE_FRAME_FLOOR_PX`,
`SENTINEL_FUSION_FLOOR_PX`, `SENTINEL_FUSION_MAX_NATIVE_PX`), **with the measured values
unchanged as defaults**. Verified: defaults read 0.75 / 70 / 40; with the variables set
they read 0.20 / 45 / 30.

Lowering the floors is not free and should not be done casually: the defaults exist
because exact-match measures ~0% below 40px, and a wrong read is worse than a missing
one — it points police at an innocent vehicle.

## 3. The one thing that is physics, not code

ANPR accuracy here is dominated by **native plate width in pixels**. Measured on 960
holdout reads: ~0% exact below 40px, 17.1% at 40–70px, 44.8% at 70–90px, higher above.

On *our* corp8 cameras this is the known, documented ceiling — wide-area bridge and
junction views where number-plate characters are roughly 7px tall. Re-confirmed directly
while writing this document, on `demo/clips/cam_01_0700.mp4`:

```
Frames sampled        : 40
Vehicles detected     : 384
Plate regions located : 0
```

Zero — and still zero with the detector threshold dropped to 0.20, which proves the
constraint is the optics and not the threshold. No model swap, retrain, or threshold
change reads characters that are not present in the image. (The detector *was* retrained
for small plates and finds 2.6× more candidates on live feeds; usable capture still caps
at 3.8% on this fleet, for the same reason.)

**What this means for the evaluation.** If the judges' footage is shot like ours — wide
area, plates a handful of pixels — no system reads it, ours included. If their footage is
framed for ANPR, or simply closer to the lane, plates land in the 70px+ band and the
measured 44.8%+ exact / **86.5% watchlist recall at 0% false alarms** applies. Watchlist
recall is the number that matters for a "find this plate" test, because 1-character fuzzy
matching recovers most single-character OCR errors; do not quote the raw transcription
figure for that task.

The correct posture on stage is therefore to *know which case you are in before you are
asked* — hence §4.

## 4. Run this the moment you get their footage

```
python -m backend.scripts.plate_size_probe --source <file | rtsp-url | camera-id>
```

It runs the real pipeline path — same vehicle detector, same
`extract_plate_candidate()` quality gate the live pipeline uses — and reports the native
plate-width distribution against the measured accuracy bands, an expected read rate, and
a plain verdict (GOOD / WORKABLE / MOSTLY UNREADABLE / UNREADABLE), plus which
environment variables to consider if the site has genuinely been measured into a
different band. It also separates "no vehicles detected at all" (a feed/decode problem)
from "vehicles but no plates" (an optics problem), so a failure on the day is
diagnosable in seconds instead of guessed at.

## 5. Known gap, stated rather than hidden

Person/face watchlist matching is fully wired end to end
(`backend/services/face_watchlist_matcher.py`) but the seeded demo persons carry
`np.random.rand(512)` as their face embedding (`backend/db/seed_data.py:110-111`) —
random noise, not a face. A person-watchlist match therefore cannot fire against seeded
demo data until those rows are re-embedded from real reference photos. The vehicle-plate
path does not share this problem: it matches operator-entered plate text, not a computed
embedding.
