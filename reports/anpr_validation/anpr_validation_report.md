# ⛔ WITHDRAWN — do not quote any number in this document

**Withdrawn**: 2026-09-06. **Superseded by**: [`docs/MEASURED_EVIDENCE.md`](../../docs/MEASURED_EVIDENCE.md)

This report is kept, unedited below the line, so the withdrawal is auditable.
Every figure in it is unsafe to quote, for two independent reasons that were
each verified against the raw manifest that produced it
(`benchmark_manifest.jsonl`, 5,500 lines).

## Reason 1 — the detection scorer was measuring the wrong thing

The benchmark's real-CCTV images are **already-cropped plates**, and the
ground-truth box recorded for each one is the **entire image**:

```json
"image_path": ".../REAL_00001_CAM_08_CAM_08_0730_t34394_00.jpg",
"bbox": [0.0, 0.0, 110.0, 34.0],
"plate_width_px": 110.0, "plate_height_px": 34.0
```

So detection was scored by running a plate detector on a picture that is
already nothing but a plate, and requiring it to return a box covering 100% of
the frame at IoU ≥ 0.7. A detector returns a slightly inset box, so IoU lands
around 0.4–0.65 and the instance is recorded as a miss.

Counted over the manifest, the ground-truth box equals the whole image for
**203 of 203 (100%) real-CCTV instances**, and for **0 of 5,297 synthetic
instances**. The bug is confined to exactly the real footage.

The report's own failure table is the proof: it lists
`detection_miss (best IoU=0.61)` × 220, `IoU=0.62` × 188, `IoU=0.60` × 165,
and so on up to `IoU=0.70`. Those are not misses — the box was found and well
placed. This is also why §2 reports the impossible pairing of **99.40%
precision with 0.98% recall**.

Because End-to-End = detection × recognition, **every E2E figure in §1, §2 and
§7 — including the headline 0.49% and all three FAIL verdicts — is an
artefact of this scorer and says nothing about the system.**

## Reason 2 — the set is 96.3% synthetic, and the tiers are not mixed

| Slice | Instances | Real Gujarat CCTV | Synthetic |
|---|---:|---:|---:|
| Tier 1 (Ideal) | 1,800 | **0** | 1,800 |
| Tier 2 (Moderate) | 1,900 | 203 | 1,697 |
| Tier 3 (Hard) | 1,800 | **0** | 1,800 |
| **Total** | **5,500** | **203 (3.7%)** | **5,297 (96.3%)** |

Tier 1 and Tier 3 contain **no real footage at all**. So the recognition
numbers quoted per tier — including the 84.50% Tier 1 exact-match and the
2.37% Tier 1 CER — are measurements of synthetic images, not of this system's
accuracy on Gujarat CCTV. They must not be presented as field accuracy.

The synthetic set is also materially easier than the real fleet at the one
variable that dominates plate readability:

| Slice | n | median | p90 | max |
|---|---:|---:|---:|---:|
| Real | 203 | **107 px** | 120 px | 135 px |
| Synthetic | 5,297 | **197 px** | 304 px | 323 px |

The synthetic centre of mass sits at roughly twice the real width — and the
real slice is itself the easy tail, since CAM_08's full distribution medians at
88 px and several cameras are optically incapable of a legible plate at all.
Accuracy measured at 197 px does not transfer to 88 px.

The 203 real instances come from 8 cameras (CAM_09 ×99, CAM_08 ×37, CAM_07
×21, CAM_10 ×19, CAM_21 ×15, CAM_11 ×7, CAM_06 ×4, CAM_04 ×1) and are all
labelled Tier 2 — too few, and too unevenly spread, to carry a headline claim
on their own.

## What this dataset got RIGHT — stated so the withdrawal is fair

The held-out split is genuine. Intersecting every benchmark plate string
against the full 98,586-row training corpus:

```
benchmark synthetic plates also in training:  0 / 5,297  = 0.0%
benchmark real plates also in training:       0 /   200  = 0.0%
```

There is **no train/test leakage**. The synthetic corpus is also correctly
annotated — the full-frame ground-truth box defect appears on 0 of its 5,297
instances. Neither the synthetic data nor the split is the problem here; the
scorer and the blended reporting are.

## What was NOT measured, but is formatted as if it were

- **§6, NVIDIA Jetson Orin Nano (TensorRT FP16)** — no Jetson device is part of
  this deployment. That row is modelled, not measured.
- **§7, "Commercial Cloud ANPR API (Reference Spec)"** — a vendor spec sheet,
  sitting inside a table headed *"Comparative Benchmark (Identical Held-Out
  Dataset)"*. It was never run on this dataset.
- **§3, ECE 59.86%** — 5,429 of 5,500 samples fall in a single confidence bin,
  and the "actual accuracy" it is compared against is the broken E2E number.
  It measures the bug, not calibration.

## What replaces it

`docs/MEASURED_EVIDENCE.md` carries only figures produced by scripts in this
repository, on this deployment's own footage, each with the command that
reproduces it.

The recognition half has been **re-measured properly** —
`python -m backend.scripts.eval_shipped_recognizer`, stored at
`reports/shipped_recognizer_eval_20260906.json`. It loads the exact ensemble
the live pipeline uses, trains nothing, and scores it once on 83 held-out real
Gujarat vehicles (960 crops) that were never trained on and never used to
select a checkpoint:

| | withdrawn report | re-measured on real footage |
|---|---|---|
| Exact match | 84.50% (Tier 1, 0 real images) | **59.0%** (83 real vehicles) |
| Character accuracy | 97.63% (Tier 1, 0 real images) | **93.1%** |
| Within 1 character | not reported | **81.9%** |

The real figure is lower, and that is the point — it is the one that survives
being asked how it was obtained.

One item below is still worth investigating on its own terms: §5 reports the
grammar post-processor making **6 true fixes against 45 silent wrong
corrections**. That audit runs on crops and never touches the broken detection
step, so the *direction* of the finding may hold even though its rates are
synthetic-derived. It has not been re-measured on real footage and is not
quoted anywhere as fact.

---
---

# (WITHDRAWN) Production-Grade ANPR Pipeline Quantitative Validation Report

**Evaluation Date**: 2026-08-29  
**Dataset Size**: 5,500 Annotated Plate Instances (203 real / 5,297 synthetic)  
**Hardware Evaluated**: NVIDIA GeForce RTX 4070 (Host), Intel CPU, NVIDIA Jetson Orin Nano (Edge — *not present, modelled*)  
**Confidence Interval Standard**: 95% Wilson Score Interval  

---

## 1. Executive Summary & Tiered Target Verdicts

| Tier Condition | Detection Recall (IoU ≥ 0.7) | Rec Exact-Match Acc | Target Det / Rec | Latency | Single Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Tier 1 (Ideal)** | 1.28% (95% CI: [0.85%, 1.91%]) | 84.50% (95% CI: [82.75%, 86.10%]) | >= 99.5% / >= 99.0% | < 50ms | **FAIL** |
| **Tier 2 (Moderate)** | 1.00% (95% CI: [0.64%, 1.56%]) | 25.53% (95% CI: [23.62%, 27.53%]) | >= 97.0% / >= 95.0% | < 50ms | **FAIL** |
| **Tier 3 (Hard)** | 0.67% (95% CI: [0.38%, 1.16%]) | 2.39% (95% CI: [1.78%, 3.20%]) | >= 90.0% / >= 85.0% | < 50ms | **FAIL** |


## 2. Quantitative Metrics Table (All Mandatory Metrics)

### A. Detection Pipeline Metrics

| Evaluation Slice | Precision@0.7 | Recall@0.7 (95% CI) | mAP@0.5 | mAP@0.5:0.95 | Miss Rate (<20px) | Miss Rate (20–40px) | Miss Rate (>40px) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Test Set** | 99.40% | 0.98% ([0.75%, 1.28%]) | 44.42% | 22.01% | 99.01% | 99.17% | 99.21% |
| **Tier 1 (Ideal)** | 99.40% | 1.28% ([0.85%, 1.91%]) | 70.02% | 34.56% | 0.00% | 99.12% | 98.79% |
| **Tier 2 (Moderate)** | 99.40% | 1.00% ([0.64%, 1.56%]) | 45.07% | 22.33% | 0.00% | 99.30% | 99.09% |
| **Tier 3 (Hard)** | 99.40% | 0.67% ([0.38%, 1.16%]) | 18.15% | 9.12% | 99.01% | 99.02% | 100.00% |


### B. Recognition & End-to-End Pipeline Metrics

| Evaluation Slice | Sample Count | Recognition Exact-Match (95% CI) | Character Error Rate (CER) | End-to-End Pipeline Accuracy (95% CI) |
| :--- | :---: | :---: | :---: | :---: |
| **Overall Test Set** | 5,500 | 37.25% ([35.99%, 38.54%]) | 34.11% | **0.49%** ([0.34%, 0.71%]) |
| **Tier 1 (Ideal)** | 1,800 | 84.50% ([82.75%, 86.10%]) | 2.37% | **0.89%** ([0.55%, 1.44%]) |
| **Tier 2 (Moderate)** | 1,900 | 25.53% ([23.62%, 27.53%]) | 27.30% | **0.58%** ([0.32%, 1.03%]) |
| **Tier 3 (Hard)** | 1,800 | 2.39% ([1.78%, 3.20%]) | 73.05% | **0.00%** ([0.00%, 0.21%]) |


## 3. Confidence Calibration & Reliability Analysis

**Expected Calibration Error (ECE)**: **59.86%**  

| Confidence Bin | Sample Count | Average Confidence | Actual Accuracy | Calibration Gap (|Acc - Conf|) |
| :---: | :---: | :---: | :---: | :---: |
| 0.0-0.1 | 0 | 5.00% | 0.00% | 0.00% |
| 0.1-0.2 | 0 | 15.00% | 0.00% | 0.00% |
| 0.2-0.3 | 0 | 25.00% | 0.00% | 0.00% |
| 0.3-0.4 | 0 | 35.00% | 0.00% | 0.00% |
| 0.4-0.5 | 0 | 45.00% | 0.00% | 0.00% |
| 0.5-0.6 | 0 | 55.00% | 0.00% | 0.00% |
| 0.6-0.7 | 0 | 65.00% | 0.00% | 0.00% |
| 0.7-0.8 | 0 | 75.00% | 0.00% | 0.00% |
| 0.8-0.9 | 71 | 89.12% | 0.00% | 89.12% |
| 0.9-1.0 | 5,429 | 97.22% | 37.74% | 59.48% |


## 4. Confusable Pairs & Character Confusion Matrix

| Confusable Pair | Direction 1 (A → B) | Error Rate 1 | Direction 2 (B → A) | Error Rate 2 | Production Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **0/O** | 0 → O (18/3408) | 0.53% | O → 0 (5/652) | 0.77% | **MITIGATED** |
| **1/I** | 1 → I (29/3567) | 0.81% | I → 1 (13/292) | 4.45% | **ACTION_REQUIRED** |
| **8/B** | 8 → B (10/2304) | 0.43% | B → 8 (15/1384) | 1.08% | **MITIGATED** |
| **5/S** | 5 → S (18/2408) | 0.75% | S → 5 (10/622) | 1.61% | **ACTION_REQUIRED** |
| **2/Z** | 2 → Z (9/3673) | 0.25% | Z → 2 (9/291) | 3.09% | **ACTION_REQUIRED** |
| **D/0** | D → 0 (19/965) | 1.97% | 0 → D (3/3408) | 0.09% | **ACTION_REQUIRED** |


## 5. Regional Format Rules & Post-Processing Pass Audit

Audits the effect of applying `indian_plate_grammar.py` and positional disambiguation to raw OCR predictions:

- **Total Raw OCR Errors**: 3,457
- **True Fixes (Corrected to Exact GT)**: **6** (0.17% of raw errors)
- **Silent Wrong Corrections (Corrupted to another valid plate)**: **45** (1.30%)
- **Unchanged Errors**: 3,406 (98.52%)
- **Unnecessary Corruptions (Ruined valid raw reads)**: **0**

## 6. Efficiency & Hardware Latency Profiles

| Target Hardware Device | Median Latency | p95 Latency | Throughput (FPS) | Peak VRAM | Real-time Target (<50ms GPU / <150ms Edge) | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA GeForce RTX 4070 (Server/Workstation GPU)** | 10.30 ms | 12.27 ms | **97.0 FPS** | 239.6 MB | < 50 ms | **PASS** |
| **NVIDIA Jetson Orin Nano (Edge Embedded - TensorRT FP16)** | 29.37 ms | 35.58 ms | **34.1 FPS** | 155.7 MB | < 150 ms | **PASS** |
| **Intel Host CPU (CPU-only Fallback)** | 66.98 ms | 88.32 ms | **14.9 FPS** | N/A | < 300 ms | **PASS** |


## 7. Comparative Benchmark (Identical Held-Out Dataset)

| System / Pipeline | Tier 1 Accuracy | Tier 2 Accuracy | Tier 3 Accuracy | Overall End-to-End | Latency (ms) | Cost / 1,000 Plates |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentinel Gujarat Production ANPR (YOLOv8 + CRNN + Grammar)** | 84.50% | 25.53% | 2.39% | **0.49%** | 10.3 ms | $0.002 |
| **Open-Source Baseline (YOLOv8 + EasyOCR)** | 50.00% | 21.32% | 1.94% | **24.36%** | 115.0 ms | $0.015 |
| **Commercial Cloud ANPR API (Reference Spec)** | 98.50% | 92.00% | 78.00% | **88.00%** | 280.0 ms | $4.500 |


## 8. Failure Cause Clustering & Ranking Roadmap

Ranked by frequency across all failed test instances:

| Rank | Root-Cause Category | Failure Count | Share of Failures | Engineering Mitigation Action |
| :---: | :--- | :---: | :---: | :--- |
| 1 | `motion_blur` | 2,599 | 21.0% | Integrate deblurring Wiener deconvolution kernel or multi-frame temporal voting across track. |
| 2 | `occlusion_bracket_mud` | 1,699 | 13.7% | Expand fine-tuning dataset with synthetic dirt masks and broken character inpainting. |
| 3 | `low_contrast_glare` | 1,165 | 9.4% | Add multi-exposure gamma curve normalization + highlight suppression filter in preprocessing. |
| 4 | `perspective_skew_30-45deg` | 878 | 7.1% | Implement STN (Spatial Transformer Network) / Thin-Plate-Spline 4-point rectification before OCR. |
| 5 | `low_resolution_under20px` | 609 | 4.9% | Enforce minimum crop upsampling threshold (300px min-side) with bicubic interpolation. |
| 6 | `detection_miss (best IoU=0.02)` | 306 | 2.5% | Dataset expansion and specialized augmentations. |
| 7 | `detection_miss (best IoU=0.03)` | 294 | 2.4% | Dataset expansion and specialized augmentations. |
| 8 | `detection_miss (best IoU=0.61)` | 220 | 1.8% | Dataset expansion and specialized augmentations. |
| 9 | `detection_miss (best IoU=0.56)` | 212 | 1.7% | Dataset expansion and specialized augmentations. |
| 10 | `detection_miss (best IoU=0.53)` | 208 | 1.7% | Dataset expansion and specialized augmentations. |
| 11 | `detection_miss (best IoU=0.62)` | 188 | 1.5% | Dataset expansion and specialized augmentations. |
| 12 | `detection_miss (best IoU=0.52)` | 178 | 1.4% | Dataset expansion and specialized augmentations. |
| 13 | `detection_miss (best IoU=0.58)` | 176 | 1.4% | Dataset expansion and specialized augmentations. |
| 14 | `detection_miss (best IoU=0.59)` | 175 | 1.4% | Dataset expansion and specialized augmentations. |
| 15 | `detection_miss (best IoU=0.51)` | 172 | 1.4% | Dataset expansion and specialized augmentations. |
| 16 | `detection_miss (best IoU=0.55)` | 171 | 1.4% | Dataset expansion and specialized augmentations. |
| 17 | `detection_miss (best IoU=0.57)` | 169 | 1.4% | Dataset expansion and specialized augmentations. |
| 18 | `detection_miss (best IoU=0.54)` | 169 | 1.4% | Dataset expansion and specialized augmentations. |
| 19 | `detection_miss (best IoU=0.60)` | 165 | 1.3% | Dataset expansion and specialized augmentations. |
| 20 | `detection_miss (best IoU=0.49)` | 152 | 1.2% | Dataset expansion and specialized augmentations. |
| 21 | `detection_miss (best IoU=0.44)` | 140 | 1.1% | Dataset expansion and specialized augmentations. |
| 22 | `detection_miss (best IoU=0.47)` | 138 | 1.1% | Dataset expansion and specialized augmentations. |
| 23 | `detection_miss (best IoU=0.48)` | 132 | 1.1% | Dataset expansion and specialized augmentations. |
| 24 | `detection_miss (best IoU=0.43)` | 129 | 1.0% | Dataset expansion and specialized augmentations. |
| 25 | `detection_miss (best IoU=0.45)` | 119 | 1.0% | Dataset expansion and specialized augmentations. |
| 26 | `detection_miss (best IoU=0.41)` | 107 | 0.9% | Dataset expansion and specialized augmentations. |
| 27 | `detection_miss (best IoU=0.40)` | 102 | 0.8% | Dataset expansion and specialized augmentations. |
| 28 | `detection_miss (best IoU=0.50)` | 96 | 0.8% | Dataset expansion and specialized augmentations. |
| 29 | `detection_miss (best IoU=0.39)` | 91 | 0.7% | Dataset expansion and specialized augmentations. |
| 30 | `detection_miss (best IoU=0.38)` | 89 | 0.7% | Dataset expansion and specialized augmentations. |
| 31 | `detection_miss (best IoU=0.18)` | 82 | 0.7% | Dataset expansion and specialized augmentations. |
| 32 | `detection_miss (best IoU=0.36)` | 79 | 0.6% | Dataset expansion and specialized augmentations. |
| 33 | `detection_miss (best IoU=0.14)` | 73 | 0.6% | Dataset expansion and specialized augmentations. |
| 34 | `detection_miss (best IoU=0.10)` | 65 | 0.5% | Dataset expansion and specialized augmentations. |
| 35 | `detection_miss (best IoU=0.09)` | 62 | 0.5% | Dataset expansion and specialized augmentations. |
| 36 | `detection_miss (best IoU=0.12)` | 57 | 0.5% | Dataset expansion and specialized augmentations. |
| 37 | `detection_miss (best IoU=0.29)` | 54 | 0.4% | Dataset expansion and specialized augmentations. |
| 38 | `detection_miss (best IoU=0.28)` | 52 | 0.4% | Dataset expansion and specialized augmentations. |
| 39 | `detection_miss (best IoU=0.25)` | 49 | 0.4% | Dataset expansion and specialized augmentations. |
| 40 | `detection_miss (best IoU=0.21)` | 49 | 0.4% | Dataset expansion and specialized augmentations. |
| 41 | `detection_miss (best IoU=0.46)` | 47 | 0.4% | Dataset expansion and specialized augmentations. |
| 42 | `detection_miss (best IoU=0.24)` | 44 | 0.4% | Dataset expansion and specialized augmentations. |
| 43 | `detection_miss (best IoU=0.17)` | 44 | 0.4% | Dataset expansion and specialized augmentations. |
| 44 | `detection_miss (best IoU=0.26)` | 40 | 0.3% | Dataset expansion and specialized augmentations. |
| 45 | `detection_miss (best IoU=0.11)` | 40 | 0.3% | Dataset expansion and specialized augmentations. |
| 46 | `detection_miss (best IoU=0.63)` | 34 | 0.3% | Dataset expansion and specialized augmentations. |
| 47 | `detection_miss (best IoU=0.20)` | 34 | 0.3% | Dataset expansion and specialized augmentations. |
| 48 | `detection_miss (best IoU=0.23)` | 33 | 0.3% | Dataset expansion and specialized augmentations. |
| 49 | `detection_miss (best IoU=0.19)` | 33 | 0.3% | Dataset expansion and specialized augmentations. |
| 50 | `detection_miss (best IoU=0.15)` | 33 | 0.3% | Dataset expansion and specialized augmentations. |
| 51 | `detection_miss (best IoU=0.22)` | 32 | 0.3% | Dataset expansion and specialized augmentations. |
| 52 | `detection_miss (best IoU=0.27)` | 31 | 0.2% | Dataset expansion and specialized augmentations. |
| 53 | `detection_miss (best IoU=0.66)` | 30 | 0.2% | Dataset expansion and specialized augmentations. |
| 54 | `detection_miss (best IoU=0.13)` | 30 | 0.2% | Dataset expansion and specialized augmentations. |
| 55 | `detection_miss (best IoU=0.16)` | 30 | 0.2% | Dataset expansion and specialized augmentations. |
| 56 | `detection_miss (best IoU=0.64)` | 29 | 0.2% | Dataset expansion and specialized augmentations. |
| 57 | `detection_miss (best IoU=0.65)` | 28 | 0.2% | Dataset expansion and specialized augmentations. |
| 58 | `detection_miss (best IoU=0.00)` | 21 | 0.2% | Dataset expansion and specialized augmentations. |
| 59 | `detection_miss (best IoU=0.42)` | 19 | 0.2% | Dataset expansion and specialized augmentations. |
| 60 | `detection_miss (best IoU=0.67)` | 18 | 0.1% | Dataset expansion and specialized augmentations. |
| 61 | `detection_miss (best IoU=0.68)` | 12 | 0.1% | Dataset expansion and specialized augmentations. |
| 62 | `detection_miss (best IoU=0.69)` | 12 | 0.1% | Dataset expansion and specialized augmentations. |
| 63 | `detection_miss (best IoU=0.35)` | 9 | 0.1% | Dataset expansion and specialized augmentations. |
| 64 | `character_glyph_ambiguity` | 8 | 0.1% | Fine-tune CTC beam search decoder with localized font variation synthetic generator. |
| 65 | `detection_miss (best IoU=0.34)` | 8 | 0.1% | Dataset expansion and specialized augmentations. |
| 66 | `detection_miss (best IoU=0.33)` | 8 | 0.1% | Dataset expansion and specialized augmentations. |
| 67 | `detection_miss (best IoU=0.31)` | 6 | 0.0% | Dataset expansion and specialized augmentations. |
| 68 | `detection_miss (best IoU=0.70)` | 6 | 0.0% | Dataset expansion and specialized augmentations. |
| 69 | `detection_miss (best IoU=0.30)` | 6 | 0.0% | Dataset expansion and specialized augmentations. |
| 70 | `detection_miss (best IoU=0.37)` | 5 | 0.0% | Dataset expansion and specialized augmentations. |
| 71 | `detection_miss (best IoU=0.32)` | 3 | 0.0% | Dataset expansion and specialized augmentations. |
| 72 | `detection_miss (best IoU=0.06)` | 1 | 0.0% | Dataset expansion and specialized augmentations. |


## 9. Failure Gallery — Worst 20 Cases

| # | Instance ID | Ground Truth | Predicted | Raw OCR | Detected? (IoU) | Condition Tags | Root-Cause Hypothesis |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `REAL_00104` | `GJ05RN5430` | `CG3ZTTF779` | `CG3ZTTF779` | NO (0.61) | dusk_dawn | clear | moderate | mid | normal | shadow | detection_miss (best IoU=0.61), motion_blur |
| 2 | `REAL_00162` | `GJ10AS0113` | `KZ34X6777` | `K23AX6777` | NO (0.49) | daylight | clear | moderate | mid | normal | clean | detection_miss (best IoU=0.49), motion_blur |
| 3 | `TEST_02487` | `TS18C4053` | `TA30BGA9818` | `TA308G498IB` | NO (0.51) | dusk_dawn | dust | frontal | near | slow | dirt_mud | detection_miss (best IoU=0.51), occlusion_bracket_mud |
| 4 | `TEST_03701` | `GJ17UV1608` | `L39925` | `L39925` | NO (0.47) | dusk_dawn | clear | moderate | near | fast | dirt_mud | detection_miss (best IoU=0.47), motion_blur, occlusion_bracket_mud |
| 5 | `TEST_03717` | `KL45UT5897` | `AR04X223` | `AR04X223` | NO (0.18) | night_ir | fog_haze | severe | mid | normal | shadow | detection_miss (best IoU=0.18), perspective_skew_30-45deg, low_contrast_glare, motion_blur |
| 6 | `TEST_03727` | `BR11UI9455` | `A3V711` | `A3V711` | NO (0.41) | night_glare | dust | severe | near | normal | dirt_mud | detection_miss (best IoU=0.41), perspective_skew_30-45deg, low_contrast_glare, motion_blur, occlusion_bracket_mud |
| 7 | `TEST_03753` | `HR23BB0244` | `LK06H4868W` | `LK06H4868W` | NO (0.43) | dusk_dawn | rain | moderate | near | fast | dirt_mud | detection_miss (best IoU=0.43), motion_blur, occlusion_bracket_mud |
| 8 | `TEST_03770` | `MH10AU5252` | `B2C373` | `B2C373` | NO (0.18) | night_ir | fog_haze | severe | mid | fast | shadow | detection_miss (best IoU=0.18), perspective_skew_30-45deg, low_contrast_glare, motion_blur |
| 9 | `TEST_03837` | `DL35HB4924` | `UK1` | `UK1` | NO (0.02) | night_glare | rain | moderate | far | normal | dirt_mud | detection_miss (best IoU=0.02), low_contrast_glare, motion_blur, occlusion_bracket_mud, low_resolution_under20px |
| 10 | `TEST_03838` | `BR20NF2274` | `PO4M5539` | `P04M5539` | NO (0.02) | night_ir | fog_haze | moderate | far | normal | shadow | detection_miss (best IoU=0.02), low_contrast_glare, motion_blur, low_resolution_under20px |
| 11 | `TEST_03859` | `BR05AD6310` | `MH34MM4H9` | `MH34MM4H9` | NO (0.18) | night_glare | rain | moderate | mid | normal | clean | detection_miss (best IoU=0.18), low_contrast_glare, motion_blur |
| 12 | `TEST_03868` | `PB33KM6022` | `A22D533` | `A22D533` | NO (0.51) | dusk_dawn | clear | severe | near | fast | shadow | detection_miss (best IoU=0.51), perspective_skew_30-45deg, motion_blur |
| 13 | `TEST_03876` | `KA08TT8423` | `UP37M99` | `UP37M99` | NO (0.22) | night_glare | dust | moderate | mid | fast | shadow | detection_miss (best IoU=0.22), low_contrast_glare, motion_blur |
| 14 | `TEST_03878` | `UP11RE1110` | `A298029` | `A298029` | NO (0.18) | dusk_dawn | rain | severe | mid | normal | dirt_mud | detection_miss (best IoU=0.18), perspective_skew_30-45deg, motion_blur, occlusion_bracket_mud |
| 15 | `TEST_03885` | `MP06SL2765` | `L144154` | `L144154` | NO (0.15) | night_ir | fog_haze | severe | mid | fast | dirt_mud | detection_miss (best IoU=0.15), perspective_skew_30-45deg, low_contrast_glare, motion_blur, occlusion_bracket_mud |
| 16 | `TEST_03902` | `PB37CS7720` | `AP2404` | `AP2404` | NO (0.02) | night_ir | clear | severe | far | normal | partial_cover | detection_miss (best IoU=0.02), perspective_skew_30-45deg, low_contrast_glare, motion_blur, occlusion_bracket_mud, low_resolution_under20px |
| 17 | `TEST_03918` | `TS43TH9589` | `AP26X0223` | `AP2GXD223` | NO (0.57) | night_ir | fog_haze | severe | near | fast | clean | detection_miss (best IoU=0.57), perspective_skew_30-45deg, low_contrast_glare, motion_blur |
| 18 | `TEST_03924` | `GJ41GY7560` | `PB3B0333` | `PB38Q333` | NO (0.44) | night_glare | dust | severe | near | fast | clean | detection_miss (best IoU=0.44), perspective_skew_30-45deg, low_contrast_glare, motion_blur |
| 19 | `TEST_03929` | `OD20CG5690` | `A38884` | `A38884` | NO (0.11) | dusk_dawn | dust | severe | mid | fast | clean | detection_miss (best IoU=0.11), perspective_skew_30-45deg, motion_blur |
| 20 | `TEST_03941` | `AP17PZ6447` | `HJ3` | `HJ3` | NO (0.03) | night_glare | clear | severe | far | fast | dirt_mud | detection_miss (best IoU=0.03), perspective_skew_30-45deg, low_contrast_glare, motion_blur, occlusion_bracket_mud, low_resolution_under20px |


## 10. Test Matrix Completeness & Sample Verification

| Matrix Dimension | Condition Cell | Instance Count | Status (≥300 instances) |
| :--- | :--- | :---: | :---: |
| **lighting** | `dusk_dawn` | 1,572 | PASS (Sufficient Evidence) |
| **lighting** | `daylight` | 2,740 | PASS (Sufficient Evidence) |
| **lighting** | `night_ir` | 588 | PASS (Sufficient Evidence) |
| **lighting** | `night_glare` | 600 | PASS (Sufficient Evidence) |
| **weather** | `clear` | 3,011 | PASS (Sufficient Evidence) |
| **weather** | `dust` | 996 | PASS (Sufficient Evidence) |
| **weather** | `fog_haze` | 1,036 | PASS (Sufficient Evidence) |
| **weather** | `rain` | 457 | PASS (Sufficient Evidence) |
| **angle** | `moderate` | 1,977 | PASS (Sufficient Evidence) |
| **angle** | `frontal` | 2,645 | PASS (Sufficient Evidence) |
| **angle** | `severe` | 878 | PASS (Sufficient Evidence) |
| **distance_size** | `mid` | 2,518 | PASS (Sufficient Evidence) |
| **distance_size** | `near` | 2,373 | PASS (Sufficient Evidence) |
| **distance_size** | `far` | 609 | PASS (Sufficient Evidence) |
| **motion** | `normal` | 1,989 | PASS (Sufficient Evidence) |
| **motion** | `static` | 894 | PASS (Sufficient Evidence) |
| **motion** | `slow` | 1,718 | PASS (Sufficient Evidence) |
| **motion** | `fast` | 899 | PASS (Sufficient Evidence) |
| **occlusion** | `shadow` | 897 | PASS (Sufficient Evidence) |
| **occlusion** | `clean` | 2,811 | PASS (Sufficient Evidence) |
| **occlusion** | `partial_cover` | 886 | PASS (Sufficient Evidence) |
| **occlusion** | `dirt_mud` | 906 | PASS (Sufficient Evidence) |
| **plate_variety** | `standard` | 2,901 | PASS (Sufficient Evidence) |
| **plate_variety** | `commercial` | 858 | PASS (Sufficient Evidence) |
| **plate_variety** | `ev` | 861 | PASS (Sufficient Evidence) |
| **plate_variety** | `custom_font` | 274 | **FLAGGED (Insufficient Evidence)** |
| **plate_variety** | `bent_damaged` | 297 | **FLAGGED (Insufficient Evidence)** |
| **plate_variety** | `temporary` | 309 | PASS (Sufficient Evidence) |
| **multi_plate** | `single` | 2,822 | PASS (Sufficient Evidence) |
| **multi_plate** | `multi` | 2,678 | PASS (Sufficient Evidence) |
| **camera_source** | `fixed_cctv` | 1,570 | PASS (Sufficient Evidence) |
| **camera_source** | `dashcam` | 1,332 | PASS (Sufficient Evidence) |
| **camera_source** | `drone` | 1,303 | PASS (Sufficient Evidence) |
| **camera_source** | `phone` | 1,295 | PASS (Sufficient Evidence) |

---

*Report generated automatically by Sentinel Gujarat Quantitative ANPR Validation Suite.*
