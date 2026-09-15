# ANPR implementation plan — detect and read Indian plates on real CCTV

**Written 2026-08-27.** Supersedes the "not feasible" verdict in
`ANPR_FEASIBILITY.md`, which used a threshold ~40% stricter than any vendor
specifies and therefore understated what this fleet can do. Keep that document
for the physics; this one is the build.

## What the measurement actually says

`backend/scripts/plate_camera_survey.py`, all 29 cameras, daylight (`0830`),
scored against vendor-standard PLATE width (Plate Recognizer recommends ~100px,
reads down to ~30px in controlled conditions):

> **"29" here is a directory count, not the fleet size — traced 2026-09-07.**
> `plate_camera_survey.py` iterates `data/clips/<id>/` folders directly
> (`clip_root.iterdir()`), never touching the `cameras` table, so it counts 27
> cameras that actually have clips plus 2 (CAM_15, CAM_22) whose directory
> exists but is empty — 29 folders, not 29 working cameras. The authoritative
> fleet size is **30** (`backend/services/fleet_census.py`); footage is on
> **27** of them (`docs/MEASURED_EVIDENCE.md`, "Fleet size — corrected").
> This doesn't change the per-camera numbers below — those are real
> measurements of the cameras they name — only what "29" means in the
> sentence introducing them.

| camera | median plate | ≥100px | ≥70px | ≥50px |
|--------|-------------:|-------:|------:|------:|
| CAM_06 | 96px         | 45.0%  | 80.0% | 80.0% |
| CAM_07 | 58px         | 17.5%  | 30.0% | 62.5% |
| CAM_02 | 26px         | 10.0%  | 21.4% | 31.0% |
| CAM_04 | 27px         |  4.5%  |  7.9% | 12.2% |
| CAM_01 | 21px         |  2.8%  |  6.5% | 10.4% |

CAM_06 is an ANPR-grade camera by accident of placement. CAM_07 is workable.
The rest are chokepoint-only or hopeless. **Scope ANPR to the cameras that
support it** — that is what every commercial deployment does.

## The three constraints that shape every decision below

1. **Pixels.** Enhancement reveals and combines information; it never invents
   it. The 100px/70px/50px columns are the real budget.
2. **Motion blur.** The close vehicles that finally have enough pixels are also
   the ones with the highest angular velocity. Fusion and deblur mitigate;
   physics caps.
3. **Night is 40% of the footage** (23 clips @0100, 24 @2200). Every number
   above is daylight. Commercial ANPR handles night with 850/940nm IR
   illuminators reading a plate's retroreflective coating — wide-area cameras
   have no such thing. Assume near-zero at night until measured.

---

## Phase 0 · Capture — upstream of everything else

No software stage below recovers a bad capture, so record what the fleet would
need to be a real ANPR deployment. This is not work to do; it is the
specification to hand the department, and the honest answer to "why not every
camera".

| requirement | why | this fleet |
|---|---|---|
| plate ≥25-30px per character row | recognition floor | met on ~8.6% of vehicles |
| shutter ≥1/500s | freezes motion at capture, so deblur is a fallback not a crutch | unknown, likely slower |
| dedicated IR illuminator (850/940nm) | reads the plate's retroreflective coating; generic CCTV IR is too weak | absent — this is why night fails |
| <~30° off-axis | limits the distortion pushed onto rectification | varies by camera |
| per-lane ANPR camera | wide-angle cameras were not installed to read plates | none deployed |

The motion-blur tension is structural: the close vehicles that finally have
enough pixels are also those with the highest angular velocity. Shutter speed
fixes this at capture; nothing downstream fully fixes it after.

## Phase 1 · Mine the corpus (2 days)

Sweep the clips end to end, not by sampling. This is the step the harvested
clips exist for, and it cannot be done against a live stream.

- Track with BoT-SORT (`ultralytics`, already the pipeline's tracker) so every
  vehicle carries a stable `track_id`.
- Keep every vehicle crop whose implied plate width ≥ 50px, cropped from the
  **source-resolution frame**, never the 1280 inference downscale.
- **Group crops by `track_id`.** A vehicle seen across 20-40 frames is one
  training sample and one fusion group. This grouping IS the deliverable —
  single frames are worth far less.
- Record per-crop: camera, frame index, vehicle width, sharpness
  (variance-of-Laplacian), time-of-day.

Target: 2,000-5,000 tracks on CAM_06/07/02/04, each with 10-40 frames.

## Phase 2 · Plate detection (2 days)

- **Model:** YOLOv8n at `imgsz=320` on the vehicle crop. Tiny and fast — it
  only ever sees a pre-cropped vehicle, never a full frame.
- **Bootstrap labels without hand-labelling 2,000 plates:** run a pretrained
  license-plate detector over the mined crops to propose boxes, then hand-verify
  a few hundred. Correcting proposals is roughly 5x faster than drawing boxes.
- **Sanity gate:** a plate box must sit in the lower half of the vehicle and
  have aspect ratio 2:1-5:1. Cheap, and it removes most false positives.

## Phase 3 · Enhancement (3 days)

Order matters and one ordering mistake destroys the main gain: **align and fuse
on RAW pixels first.** CLAHE, deblur and binarisation are nonlinear — applied
per-frame they distort each frame differently, break sub-pixel alignment, and
throw away the fusion benefit. Enhance the fused result once.

**Select frames before fusing — do NOT fuse the whole track.** Score every crop
in the track by variance-of-Laplacian and fuse only the sharpest 5-8. Averaging
blurry frames into sharp ones actively degrades the result; more frames is not
better, sharper frames are. (An earlier revision of this plan said "fuse 20-40
frames" — that was wrong.)

**Gate each stage on measured quality; do not run all of them on every crop.**
Deconvolution is not free — `richardson_lucy` on an already-sharp plate adds
ringing and makes it worse. CLAHE on a well-exposed plate can crush glyph
edges. Gate on:
  - `findTransformECC` refinement  -> only when distortion is high
  - `richardson_lucy` deblur       -> only when blur score is poor
  - denoise + CLAHE                -> only on low-quality crops
Per-crop gating beats picking one global configuration by ablation.

| step | tool | installed |
|------|------|-----------|
| 1. sub-pixel align | `skimage.registration.phase_cross_correlation(upsample_factor=10)`, `cv2.findTransformECC` | yes |
| 2. **multi-frame fusion** | shift-and-add onto a 3-4x grid (NumPy) | yes |
| 3. perspective rectify | `cv2.minAreaRect` → `getPerspectiveTransform` → `warpPerspective` | yes |
| 4. motion deblur | `skimage.restoration.richardson_lucy` / `unsupervised_wiener` | yes |
| 5. denoise | `cv2.fastNlMeansDenoisingColored` (fusion already gives √N) | yes |
| 6. contrast | `cv2.createCLAHE(clipLimit=2.0)` on LAB L-channel | yes |
| 7. upscale | `cv2.INTER_LANCZOS4`, or `cv2.dnn_superres` ESPCN/FSRCNN | yes |

**Step 2 is the only step that adds real information.** Sub-pixel jitter means
each frame samples the plate on a different grid; combining 20-40 frames
genuinely recovers detail and cuts noise by √N. Expect 2-3x effective
resolution. Everything else preserves or reveals.

**Build this as an ablation harness** — every stage independently toggleable,
all measured on the same plates. It is how the stack gets tuned, and it is the
table to show judges when they ask why it is shaped this way.

**Excluded:** GAN-based SR (Real-ESRGAN, SD-upscale). Trained adversarially to
look sharp, so it fails *confidently* — renders a crisp, wrong character. L1/L2
models (ESPCN/EDSR) fail conservatively (blurry but faithful) and are allowed.
Any enhancement must beat its own ablation on ground truth before it ships.

## Phase 4 · Recognition (3 days)

**Synthetic pretraining is the unlock** — it removes the label bottleneck that
has blocked this work from the start.

1. **Generate ~100k synthetic Indian plates.** Correct font, correct `GJ 01 AB
   1234` grammar, valid RTO codes. Then degrade them to match *measured*
   statistics from Phase 1 — the actual blur, noise, JPEG QP, perspective and
   illumination distribution of this footage, not generic augmentation.
   Degrade for **night and IR** as well as daylight — synthetic is the only
   way to get night training data without hand-labelling footage where the
   plates are barely visible, and night is 40% of the corpus.
2. **Pretrain** a recognizer on synthetic. Candidates, in order of preference:
   - **PARSeq** — current SOTA scene-text recognition, permissive licence
   - **LPRNet** — lightweight, plate-specific, ONNX-exportable (`onnxruntime` present)
   - **PaddleOCR** / **TrOCR** — strong, both need installing
   - EasyOCR — currently used, weakest; keep only as an ensemble vote

   **Cascade, do not always ensemble.** Run PARSeq first; escalate to
   PaddleOCR/EasyOCR only when its confidence is low. A permanent 3-way
   ensemble costs ~3x the compute for marginal accuracy, and this fleet is
   already at 178% of one GPU. The format decoder (below) runs always — it is
   pure logic and effectively free.
3. **Fine-tune on ~500 hand-labelled real plates** from the mined corpus.
   Synthetic teaches glyph shapes; real data teaches this fleet's degradation.
4. **Format-constrained decoding** — highest value per line of code, no model
   required. Positions 0-1 letters, 2-3 digits, 4-5 letters, 6-9 digits
   resolves 8↔B, 0↔O/D, 1↔I, 5↔S *by position*. Validate the RTO code against
   real Gujarat districts (GJ-01…GJ-38).

## Phase 5 · Consensus and confidence (1 day)

- Levenshtein-weighted vote across the track's 20-40 per-frame reads.
- Weight each frame's vote by its sharpness and plate width.
- Emit a calibrated confidence; above threshold auto-accept, below route to
  human review. **Reporting "unread" honestly is a feature, not a shortfall.**

## Phase 6 · Evaluation (2 days)

- Hand-label a **held-out** test set of ~200 plates. This is unavoidable —
  without ground truth there is no accuracy number, only a vibe.
- Report **CER**, **plate-level exact-match**, and **coverage** (fraction of
  vehicles yielding any confident read) separately. Coverage × accuracy is the
  honest headline; either alone is misleading.
- Report daylight and night separately. Never blend them.

---

## Realistic expected outcome

On CAM_06/07, daylight, four-wheelers, full stack:

- plate **detection**: 90%+
- plate **read** given ≥70px plate: 85-95%
- **coverage** (all vehicles on that camera): 50-70% on CAM_06
- fleet-wide across 29 cameras: low, and that is correct — 19 cameras
  physically cannot support it. **The "19" is not re-verified against the
  corrected 30/27 roster** (only the denominator was traced above; this
  specific count would need `plate_camera_survey.py` re-run against the
  current camera list to confirm rather than assumed unchanged)
- motorcycles (two-row plates, ~45mm characters): substantially worse; scope
  them out of accuracy claims
- night: unmeasured, assume poor

**A system that reads 60% of vehicles at 95% precision is genuinely useful and
defensible. One claiming 100% at 40% precision collapses under the first live
test a judge runs.**

## Cost

~13 working days. That is roughly half the remaining runway (28 days as of
2026-08-27), and it competes directly with the behaviour layer, which currently
emits **zero** alerts of any kind (72 alerts, all `anpr_uncertain`). Both cannot
be done well. That tradeoff is the owner's call, not the implementer's — but it
should be made deliberately rather than by drift.
