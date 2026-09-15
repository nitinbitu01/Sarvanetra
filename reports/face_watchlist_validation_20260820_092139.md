# Sentinel Gujarat — Face Watchlist Match Validation Report

Generated: 2026-08-20 09:21 UTC
InsightFace buffalo_l (ArcFace r100)

> ### ⛔ DEGENERATE RUN — NO MATCHES FIRED AT ANY POINT
>
> Not one pair in this dataset scored at or above the threshold
> (0.8). True Positives = 0 and False Positives = 0, so every
> rate derived from them below is **0 divided by 0, printed as 0.0%**.
> Those are not passing scores — they are undefined, and any green
> tick next to them is an artifact of the formatting, not a result.
>
> Same-person pairs here average **0.1628**, far below the
> **0.8** threshold, so this dataset cannot exercise the
> threshold at all. AUC may still look high (the two classes ARE
> separable — just in a score range nowhere near where the system
> is configured to act), which makes this failure mode easy to miss.
>
> **Conclusion: this run validates nothing about the configured
> threshold.** Re-run against real labeled pairs whose scores land in
> the operating range before drawing any conclusion from this report.

> ⚠️ **PROXY VALIDATION — SYNTHETIC DATA**
> These results are based on synthetic random embeddings, NOT real footage.
> This report MUST be re-run on representative Gujarat CCTV footage
> with labeled same/different pairs before go-live.

> ### ⛔ THIS NUMBER IS NOT A SAFETY RESULT
>
> Read the False Positive Rate below as **0% BY CONSTRUCTION, not by
> measurement.** Synthetic 'different person' pairs here are two
> independent random 512-dimensional unit vectors, which are
> near-orthogonal by definition — their cosine similarity clusters
> around 0.0 and cannot approach the 0.80 threshold. The separation,
> the AUC, and the false-positive rate in this report therefore
> measure the random number generator, **not** ArcFace's ability to
> tell two real people apart in low-resolution CCTV stills — which is
> precisely where face recognition is known to be weakest, and where
> near-look-alikes, poor lighting, motion blur, and off-angle capture
> all live.
>
> **No face-watchlist validation has ever been run against real data
> in this project.** `data/test_face_pairs/` holds no labeled pairs —
> see its README for the layout and the data-quality requirements the
> pairs have to meet. Until it is populated and this report is
> regenerated, treat the `WATCHLIST_FACE_MATCH_THRESHOLD` default of 0.8 as an
> unvalidated engineering guess, and treat any CRITICAL alert it
> fires as unverified.


---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total pairs | 400 |
| Same-person pairs | 200 |
| Different-person pairs | 200 |
| Data source | **SYNTHETIC PROXY** — re-run on real footage before go-live |

---

## Score Distributions

| | Mean Score | 
|--|--|
| Same-person pairs | 0.1628 |
| Different-person pairs | -0.0030 |
| Separation | 0.1658 |

---

## ROC Analysis

- **AUC**: 0.9948 ✅ Good separation

---

## Current Threshold Performance

### Watchlist Match Threshold (0.8)

| Metric | Value |
|--------|-------|
| True Positives | 0 |
| False Positives (false alerts) | 0 |
| True Negatives | 200 |
| False Negatives (missed matches) | 200 |
| Precision | 0.0% |
| Recall | 0.0% |
| F1 | 0.0000 |
| **False Alert Rate** (FP / (FP+TP)) | **n/a** — 0/0, nothing fired |
| **False Positive Rate** (FP / (FP+TN)) | **n/a** — nothing fired; see degenerate-run notice above |


> No review-queue band for face watchlist matching — a score below threshold simply means no alert fires. A missed match here is a worse outcome than an extra review for policing purposes, but this design does not currently surface near-misses for human review.

---

## Threshold Recommendation

**Suggested threshold for ≤1% false-positive rate: `0.2929`**

> ⚠️ This is a SUGGESTION only. Do NOT auto-apply.
> An authorized admin must review this recommendation and manually update
> `WATCHLIST_FACE_MATCH_THRESHOLD` in `.env` if they agree with it.
> Re-run validation after any threshold change — and this threshold change carries higher stakes than a ReID threshold change, since it directly controls when an automatic CRITICAL police alert fires on a real person.

---

### Fairness / Stratification Analysis

> ⚠️ **No stratified fairness data available.**
> This is a gap that MUST be closed with representative Gujarat-condition
> footage before deployment — not a passed check.
> Lighting conditions (day/night), camera angle buckets, and image quality
> tiers should all be present in the test set.


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
