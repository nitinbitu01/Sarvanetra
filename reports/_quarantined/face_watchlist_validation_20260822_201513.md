# Sentinel Gujarat — Face Watchlist Match Validation Report

Generated: 2026-08-22 20:15 UTC
InsightFace buffalo_l (ArcFace r100)


---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total pairs | 20 |
| Same-person pairs | 10 |
| Different-person pairs | 10 |
| Data source | Labeled dataset |

---

## Score Distributions

| | Mean Score | 
|--|--|
| Same-person pairs | 0.9833 |
| Different-person pairs | 0.0128 |
| Separation | 0.9705 |

---

## ROC Analysis

- **AUC**: 1.0000 ✅ Good separation

---

## Current Threshold Performance

### Watchlist Match Threshold (0.8)

| Metric | Value |
|--------|-------|
| True Positives | 10 |
| False Positives (false alerts) | 0 |
| True Negatives | 10 |
| False Negatives (missed matches) | 0 |
| Precision | 100.0% |
| Recall | 100.0% |
| F1 | 1.0000 |
| **False Alert Rate** (FP / (FP+TP)) | **0.0%** ✅ Within 1% target |
| **False Positive Rate** (FP / (FP+TN)) | **0.00%** ✅ Within 1% target |


> **Read the False Positive Rate row, not just precision.** At threshold
> `0.8`, **0.00%** of the not-on-the-watchlist people in this
> test set were auto-flagged CRITICAL. That rate scales with footfall, not
> with watchlist size: at a location seeing 10,000 people a day it implies
> roughly **0 innocent people flagged to an
> officer per day**. Precision (100.0%) is the officer's-eye
> view of the alerts that do fire; the FPR is the public's-eye view of being
> stopped without cause. Both belong in any go/no-go decision on this
> threshold.

> No review-queue band for face watchlist matching — a score below threshold simply means no alert fires. A missed match here is a worse outcome than an extra review for policing purposes, but this design does not currently surface near-misses for human review.

---

## Threshold Recommendation

**Suggested threshold for ≤1% false-positive rate: `0.9968`**

> ⚠️ This is a SUGGESTION only. Do NOT auto-apply.
> An authorized admin must review this recommendation and manually update
> `WATCHLIST_FACE_MATCH_THRESHOLD` in `.env` if they agree with it.
> Re-run validation after any threshold change — and this threshold change carries higher stakes than a ReID threshold change, since it directly controls when an automatic CRITICAL police alert fires on a real person.

---

### Fairness / Stratification Analysis

Overall accuracy at auto-merge threshold: **100.0%**

| Bucket | N | Accuracy | Δ vs Overall |
|--------|---|----------|--------------|
| angle=frontal | 10 | 100.0% | +0.0% |
| angle=off_axis | 10 | 100.0% | +0.0% |
| lighting=day | 10 | 100.0% | +0.0% |
| lighting=night | 10 | 100.0% | +0.0% |
| quality=high | 14 | 100.0% | +0.0% |
| quality=low | 6 | 100.0% | +0.0% |

> Buckets flagged ⚠️ show accuracy more than 5% below overall — investigate before deployment.


---

## Demographic Bias Disclaimer

> ⚠️ **OPEN ITEM — requires explicit legal/ethics sign-off**
>
> This system has NOT been evaluated for demographic, skin-tone, or ethnic
> bias in the ReID matching accuracy. Such evaluation requires:
> 1. A dataset design decision specifying representative demographic groups
>    relevant to Gujarat deployment contexts.
> 2. Labeled data with those demographic attributes.
> 3. Review by whoever owns legal/ethical accountability for this system.
>
> Engineering cannot close this item alone. It is flagged here so it appears
> in every validation report and cannot be overlooked during compliance review.

---

## Re-Run Requirements

This report must be re-run whenever:
- The InsightFace model is updated or fine-tuned on local footage
- `WATCHLIST_FACE_MATCH_THRESHOLD` is changed
- Camera hardware changes materially (resolution, field of view, lighting)
- After each major software release that touches the embedding pipeline

Command: `python -m backend.scripts.reid_validate --mode face --dataset-dir data/test_face_pairs/`
