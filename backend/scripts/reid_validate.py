"""
backend/scripts/reid_validate.py — Offline validation & fairness evaluation harness.

PURPOSE:
  This script MUST be run before treating the ReID engine's auto-merge threshold
  as trustworthy in production. It computes match-score distributions, ROC/AUC,
  precision/recall at the configured thresholds, and a fairness breakdown.

USAGE:
  python -m backend.scripts.reid_validate [--dataset-dir PATH] [--output-dir PATH]

DATASET FORMAT:
  The script accepts a directory with the following layout (Market-1501 compatible):
    dataset_dir/
      same/          # pairs of crops that ARE the same person
        pair_001_a.jpg  pair_001_b.jpg
        pair_002_a.jpg  pair_002_b.jpg
        ...
      different/     # pairs of crops that are DIFFERENT persons
        pair_001_a.jpg  pair_001_b.jpg
        ...
      metadata.json  # OPTIONAL: {"pair_001": {"lighting": "day", "quality": "high"}}

  If no real dataset is available, the script generates a synthetic proxy using
  random embeddings and clearly marks the report as a PROXY VALIDATION.

RE-RUN CADENCE (from spec §4.3):
  This script must be re-run whenever:
  - The OSNet-IBN model is updated or fine-tuned
  - The REID_AUTO_MERGE_THRESHOLD or REID_REVIEW_LOWER_THRESHOLD is changed
  - Camera hardware changes significantly (resolution, angle, lighting)
  - After major deployment changes
  Never treat a single validation run as permanent approval.

NOTE ON DEMOGRAPHIC BIAS:
  This script does NOT evaluate demographic/skin-tone bias. That evaluation
  requires a dataset design decision beyond engineering scope — it must be
  commissioned with representative labeled data and reviewed by the legal/
  ethics team responsible for this deployment. This is flagged as an
  OPEN ITEM requiring explicit sign-off before production use.

OUTPUT:
  reports/reid_validation_<date>.md — markdown report with ROC/AUC, threshold
  analysis, fairness breakdown, and explicit gap statements.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ── sys.path setup so we can import backend modules when run as a script ──────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np


def _load_pairs_from_dir(
    dataset_dir: Path,
    embedder=None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[bool], dict[str, dict], list[str]]:
    """Load same/different pairs from dataset_dir.

    Args:
        dataset_dir: same/different crop-pair directory (see module docstring).
        embedder: Any object exposing .extract(crop_bgr) -> np.ndarray | None.
                   Defaults to the ReID (OSNet-IBN) embedder for backward
                   compatibility. Pass get_face_embedder() for --mode face.

    Returns:
        pairs:    list of (emb_a, emb_b) tuples
        labels:   True = same person, False = different person
        metadata: pair_id → {lighting, quality, ...} if metadata.json exists
    """
    import cv2

    if embedder is None:
        from backend.services.reid_embedder import get_embedder
        embedder = get_embedder()
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[bool] = []
    pair_ids: list[str] = []

    for is_same, subdir in [(True, "same"), (False, "different")]:
        subpath = dataset_dir / subdir
        if not subpath.exists():
            continue
        # Group by pair_id prefix (pair_001_a / pair_001_b)
        crops: dict[str, list[Path]] = {}
        for f in sorted(subpath.glob("*.jpg")) + sorted(subpath.glob("*.png")):
            stem = f.stem
            # Extract pair_id: everything except trailing _a or _b
            if stem.endswith("_a") or stem.endswith("_b"):
                pair_id = stem[:-2]
            else:
                continue
            crops.setdefault(pair_id, []).append(f)

        for pair_id, files in crops.items():
            if len(files) < 2:
                continue
            files = sorted(files)   # _a before _b
            img_a = cv2.imread(str(files[0]))
            img_b = cv2.imread(str(files[1]))
            if img_a is None or img_b is None:
                continue
            emb_a = embedder.extract(img_a)
            emb_b = embedder.extract(img_b)
            # Face embedder can return None (no face detected in the crop) —
            # the ReID (OSNet-IBN) embedder never does. Skip such pairs rather
            # than crashing the cosine-similarity step below.
            if emb_a is None or emb_b is None:
                print(f"  [skip] {pair_id}: no face detected in one or both crops")
                continue
            pairs.append((emb_a, emb_b))
            labels.append(is_same)
            pair_ids.append(pair_id)

    # Load optional metadata
    meta_path = dataset_dir / "metadata.json"
    metadata: dict[str, dict] = {}
    if meta_path.exists():
        with meta_path.open() as fh:
            metadata = json.load(fh)

    return pairs, labels, metadata, pair_ids


def _generate_synthetic_pairs(
    n_same: int = 100, n_different: int = 100
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[bool], dict, list[str]]:
    """Generate synthetic proxy pairs for testing without a real dataset.

    Same-person pairs: two embeddings from the same base vector + small noise.
    Different-person pairs: two completely independent random embeddings.
    """
    rng = np.random.default_rng(42)
    pairs, labels, pair_ids = [], [], []

    for i in range(n_same):
        base = rng.standard_normal(512).astype(np.float32)
        base /= np.linalg.norm(base)
        noise_a = rng.standard_normal(512).astype(np.float32) * 0.1
        noise_b = rng.standard_normal(512).astype(np.float32) * 0.1
        emb_a = base + noise_a
        emb_a /= np.linalg.norm(emb_a)
        emb_b = base + noise_b
        emb_b /= np.linalg.norm(emb_b)
        pairs.append((emb_a, emb_b))
        labels.append(True)
        pair_ids.append(f"syn_same_{i:04d}")

    for i in range(n_different):
        emb_a = rng.standard_normal(512).astype(np.float32)
        emb_a /= np.linalg.norm(emb_a)
        emb_b = rng.standard_normal(512).astype(np.float32)
        emb_b /= np.linalg.norm(emb_b)
        pairs.append((emb_a, emb_b))
        labels.append(False)
        pair_ids.append(f"syn_diff_{i:04d}")

    return pairs, labels, {}, pair_ids


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two unit-norm vectors."""
    return float(np.dot(a, b))


def _compute_roc(
    scores: list[float], labels: list[bool]
) -> tuple[list[float], list[float], list[float]]:
    """Compute ROC curve points (fpr, tpr, thresholds)."""
    thresholds = sorted(set(scores), reverse=True)
    fprs, tprs = [], []
    pos = sum(labels)
    neg = len(labels) - pos

    for thresh in thresholds:
        tp = sum(1 for s, l in zip(scores, labels) if s >= thresh and l)
        fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and not l)
        tpr = tp / pos if pos > 0 else 0.0
        fpr = fp / neg if neg > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)

    return fprs, tprs, thresholds


def _compute_auc(fprs: list[float], tprs: list[float]) -> float:
    """Compute AUC using trapezoidal integration."""
    auc = 0.0
    for i in range(1, len(fprs)):
        auc += (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
    return abs(auc)


def _threshold_metrics(
    scores: list[float], labels: list[bool], threshold: float
) -> dict:
    """Precision, recall, F1 at a given threshold."""
    tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l)
    fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and not l)
    fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l)

    tn = sum(1 for s, l in zip(scores, labels) if s < threshold and not l)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    false_merge_rate = fp / (fp + tp) if (fp + tp) > 0 else 0.0
    # Distinct from false_merge_rate above, and the two answer different
    # questions — both matter, and conflating them understates the risk:
    #   false_merge_rate = FP / (FP + TP)  "of the alerts that fired, what
    #                                       fraction were wrong"
    #   fpr              = FP / (FP + TN)  "of the people who are NOT on the
    #                                       watchlist, what fraction got
    #                                       auto-flagged CRITICAL anyway"
    # The second is the one that tells you how often an innocent person is
    # flagged in front of an officer, and it is the number that scales with
    # footfall: a 1% FPR at a station gate with 20,000 daily passers-by is
    # 200 wrongly-flagged people a day, regardless of how good precision looks.
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_merge_rate": round(false_merge_rate, 4),
        "fpr": round(fpr, 4),
    }


def _fairness_analysis(
    scores: list[float],
    labels: list[bool],
    pair_ids: list[str],
    metadata: dict,
    auto_threshold: float,
) -> str:
    """Compute per-bucket accuracy if stratification metadata is available."""
    if not metadata:
        return (
            "### Fairness / Stratification Analysis\n\n"
            "> ⚠️ **No stratified fairness data available.**\n"
            "> This is a gap that MUST be closed with representative Gujarat-condition\n"
            "> footage before deployment — not a passed check.\n"
            "> Lighting conditions (day/night), camera angle buckets, and image quality\n"
            "> tiers should all be present in the test set.\n"
        )

    # Group by any available metadata key
    buckets: dict[str, list[tuple[float, bool]]] = {}
    for pid, score, label in zip(pair_ids, scores, labels):
        meta = metadata.get(pid, {})
        for key, val in meta.items():
            bucket_key = f"{key}={val}"
            buckets.setdefault(bucket_key, []).append((score, label))

    if not buckets:
        return (
            "### Fairness / Stratification Analysis\n\n"
            "> ⚠️ metadata.json found but no pairs matched — check pair_id format.\n"
        )

    lines = ["### Fairness / Stratification Analysis\n"]
    overall_acc = sum(
        1 for s, l in zip(scores, labels)
        if (s >= auto_threshold) == l
    ) / len(labels)
    lines.append(f"Overall accuracy at auto-merge threshold: **{overall_acc:.1%}**\n")
    lines.append("| Bucket | N | Accuracy | Δ vs Overall |")
    lines.append("|--------|---|----------|--------------|")

    for bk, pairs_in_bucket in sorted(buckets.items()):
        n = len(pairs_in_bucket)
        acc = sum(
            1 for s, l in pairs_in_bucket if (s >= auto_threshold) == l
        ) / n
        delta = acc - overall_acc
        flag = " ⚠️" if delta < -0.05 else ""
        lines.append(f"| {bk} | {n} | {acc:.1%} | {delta:+.1%}{flag} |")

    lines.append("")
    lines.append(
        "> Buckets flagged ⚠️ show accuracy more than 5% below overall — "
        "investigate before deployment.\n"
    )
    return "\n".join(lines)


def _find_target_threshold(
    scores: list[float],
    labels: list[bool],
    max_fpr: float = 0.01,
) -> float | None:
    """Find lowest threshold where FPR ≤ max_fpr (i.e. false-merge rate ≤ 1%)."""
    thresholds = sorted(set(scores), reverse=True)
    neg = sum(1 for l in labels if not l)
    for thresh in thresholds:
        fp = sum(1 for s, l in zip(scores, labels) if s >= thresh and not l)
        fpr = fp / neg if neg > 0 else 0.0
        if fpr <= max_fpr:
            return thresh
    return None


def generate_report(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    labels: list[bool],
    pair_ids: list[str],
    metadata: dict,
    is_synthetic: bool,
    output_dir: Path,
    mode: str = "reid",
) -> Path:
    """Run full validation analysis and write the markdown report.

    Args:
        mode: "reid" (default — OSNet-IBN body ReID, two thresholds) or
              "face" (Day 7 — InsightFace watchlist matching, single
              threshold; a missed match just means no alert, there is no
              review-queue band for face matching).
    """
    from backend.core.config import settings

    is_face = mode == "face"
    if is_face:
        auto_t = settings.WATCHLIST_FACE_MATCH_THRESHOLD
        review_t = None
        model_line = "InsightFace buffalo_l (ArcFace r100)"
        title = "Sentinel Gujarat — Face Watchlist Match Validation Report"
        threshold_label = "Watchlist Match Threshold"
        config_key = "WATCHLIST_FACE_MATCH_THRESHOLD"
        run_command = "python -m backend.scripts.reid_validate --mode face --dataset-dir data/test_face_pairs/"
    else:
        auto_t = settings.REID_AUTO_MERGE_THRESHOLD
        review_t = settings.REID_REVIEW_LOWER_THRESHOLD
        model_line = "OSNet-IBN pretrained weights (MSMT17 benchmark)"
        title = "Sentinel Gujarat — ReID Validation Report"
        threshold_label = "Auto-Merge Threshold"
        config_key = "REID_AUTO_MERGE_THRESHOLD"
        run_command = "python -m backend.scripts.reid_validate --dataset-dir data/test_pairs/"

    # Compute all-pairs cosine similarities
    scores = [_cosine_similarity(a, b) for a, b in pairs]

    # ROC + AUC
    fprs, tprs, thresholds = _compute_roc(scores, labels)
    auc = _compute_auc(fprs, tprs)

    # Threshold metrics
    m_auto = _threshold_metrics(scores, labels, auto_t)
    m_review = _threshold_metrics(scores, labels, review_t) if review_t is not None else None

    # Recommended threshold — for face mode, a missed match (false negative) is
    # the worse outcome, but a false positive is what actually harms someone
    # (wrongful stop/detention), so we still target a low false-positive rate
    # here, same as ReID's false-merge-rate target. This is a starting point
    # for discussion, not a substitute for real labeled-pair validation.
    recommended = _find_target_threshold(scores, labels, max_fpr=0.01)

    # Score distribution
    same_scores = [s for s, l in zip(scores, labels) if l]
    diff_scores = [s for s, l in zip(scores, labels) if not l]
    same_mean = float(np.mean(same_scores)) if same_scores else 0.0
    diff_mean = float(np.mean(diff_scores)) if diff_scores else 0.0

    # Fairness section
    fairness_md = _fairness_analysis(scores, labels, pair_ids, metadata, auto_t)

    # ── Degenerate-run guard ────────────────────────────────────────────────
    # If NOTHING scored above the threshold, then TP = FP = 0 and every
    # rate below is 0/0 reported as 0.0% — which the "✅ Within 1% target"
    # tick then dresses up as a pass. Reports in reports/ have been carrying
    # exactly that false green tick since Day 6. A run where the matcher
    # matched nothing at all is not a pass; it is an uninterpretable run,
    # and it must say so above the numbers, not below them.
    fired_any = (m_auto["tp"] + m_auto["fp"]) > 0
    degenerate_warning = ""
    if not fired_any:
        degenerate_warning = (
            "> ### ⛔ DEGENERATE RUN — NO MATCHES FIRED AT ANY POINT\n"
            ">\n"
            f"> Not one pair in this dataset scored at or above the threshold\n"
            f"> ({auto_t}). True Positives = 0 and False Positives = 0, so every\n"
            "> rate derived from them below is **0 divided by 0, printed as 0.0%**.\n"
            "> Those are not passing scores — they are undefined, and any green\n"
            "> tick next to them is an artifact of the formatting, not a result.\n"
            ">\n"
            f"> Same-person pairs here average **{same_mean:.4f}**, far below the\n"
            f"> **{auto_t}** threshold, so this dataset cannot exercise the\n"
            "> threshold at all. AUC may still look high (the two classes ARE\n"
            "> separable — just in a score range nowhere near where the system\n"
            "> is configured to act), which makes this failure mode easy to miss.\n"
            ">\n"
            "> **Conclusion: this run validates nothing about the configured\n"
            "> threshold.** Re-run against real labeled pairs whose scores land in\n"
            "> the operating range before drawing any conclusion from this report.\n\n"
        )

    # Build markdown report
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    proxy_warning = ""
    if is_synthetic:
        proxy_warning = (
            "> ⚠️ **PROXY VALIDATION — SYNTHETIC DATA**\n"
            "> These results are based on synthetic random embeddings, NOT real footage.\n"
            "> This report MUST be re-run on representative Gujarat CCTV footage\n"
            "> with labeled same/different pairs before go-live.\n\n"
        )
        if is_face:
            # Without this, a synthetic face report is actively misleading:
            # independent random 512-d unit vectors are near-orthogonal, so
            # every "different person" pair scores near 0.0 and the FPR at
            # 0.80 comes out at a flawless 0.00%. That number is a property
            # of the random number generator, not of ArcFace on CCTV stills.
            proxy_warning += (
                "> ### ⛔ THIS NUMBER IS NOT A SAFETY RESULT\n"
                ">\n"
                "> Read the False Positive Rate below as **0% BY CONSTRUCTION, not by\n"
                "> measurement.** Synthetic 'different person' pairs here are two\n"
                "> independent random 512-dimensional unit vectors, which are\n"
                "> near-orthogonal by definition — their cosine similarity clusters\n"
                "> around 0.0 and cannot approach the 0.80 threshold. The separation,\n"
                "> the AUC, and the false-positive rate in this report therefore\n"
                "> measure the random number generator, **not** ArcFace's ability to\n"
                "> tell two real people apart in low-resolution CCTV stills — which is\n"
                "> precisely where face recognition is known to be weakest, and where\n"
                "> near-look-alikes, poor lighting, motion blur, and off-angle capture\n"
                "> all live.\n"
                ">\n"
                "> **No face-watchlist validation has ever been run against real data\n"
                "> in this project.** `data/test_face_pairs/` holds no labeled pairs —\n"
                "> see its README for the layout and the data-quality requirements the\n"
                "> pairs have to meet. Until it is populated and this report is\n"
                f"> regenerated, treat the `{config_key}` default of {auto_t} as an\n"
                "> unvalidated engineering guess, and treat any CRITICAL alert it\n"
                "> fires as unverified.\n\n"
            )

    review_section = ""
    if not is_face and m_review is not None:
        review_section = f"""### Review-Queue Lower Threshold ({review_t})

| Metric | Value |
|--------|-------|
| Precision | {m_review['precision']:.1%} |
| Recall | {m_review['recall']:.1%} |
| F1 | {m_review['f1']:.4f} |
| False Merge Rate | {m_review['false_merge_rate']:.1%} |
"""
    elif is_face:
        review_section = (
            "> No review-queue band for face watchlist matching — a score below "
            "threshold simply means no alert fires. A missed match here is a "
            "worse outcome than an extra review for policing purposes, but this "
            "design does not currently surface near-misses for human review.\n"
        )

    report = f"""# {title}

Generated: {date_str}
{model_line}

{degenerate_warning}{proxy_warning}
---

## Dataset Summary

| Metric | Value |
|--------|-------|
| Total pairs | {len(pairs)} |
| Same-person pairs | {sum(labels)} |
| Different-person pairs | {len(labels) - sum(labels)} |
| Data source | {"**SYNTHETIC PROXY** — re-run on real footage before go-live" if is_synthetic else "Labeled dataset"} |

---

## Score Distributions

| | Mean Score | 
|--|--|
| Same-person pairs | {same_mean:.4f} |
| Different-person pairs | {diff_mean:.4f} |
| Separation | {same_mean - diff_mean:.4f} |

---

## ROC Analysis

- **AUC**: {auc:.4f} {"✅ Good separation" if auc > 0.9 else "⚠️ Below 0.90 — investigate before deploying"}

---

## Current Threshold Performance

### {threshold_label} ({auto_t})

| Metric | Value |
|--------|-------|
| True Positives | {m_auto['tp']} |
| False Positives ({"false alerts" if is_face else "false merges"}) | {m_auto['fp']} |
| True Negatives | {m_auto['tn']} |
| False Negatives ({"missed matches" if is_face else "missed merges"}) | {m_auto['fn']} |
| Precision | {m_auto['precision']:.1%} |
| Recall | {m_auto['recall']:.1%} |
| F1 | {m_auto['f1']:.4f} |
| **False {"Alert" if is_face else "Merge"} Rate** (FP / (FP+TP)) | {"**n/a** — 0/0, nothing fired" if not fired_any else f"**{m_auto['false_merge_rate']:.1%}** " + ("⚠️ Exceeds 1% target" if m_auto['false_merge_rate'] > 0.01 else "✅ Within 1% target")} |
| **False Positive Rate** (FP / (FP+TN)) | {"**n/a** — nothing fired; see degenerate-run notice above" if not fired_any else f"**{m_auto['fpr']:.2%}** " + ("⚠️ Exceeds 1% target" if m_auto['fpr'] > 0.01 else "✅ Within 1% target")} |

{"" if (not is_face or not fired_any) else f'''
> **Read the False Positive Rate row, not just precision.** At threshold
> `{auto_t}`, **{m_auto['fpr']:.2%}** of the not-on-the-watchlist people in this
> test set were auto-flagged CRITICAL. That rate scales with footfall, not
> with watchlist size: at a location seeing 10,000 people a day it implies
> roughly **{int(round(m_auto['fpr'] * 10000)):,} innocent people flagged to an
> officer per day**. Precision ({m_auto['precision']:.1%}) is the officer's-eye
> view of the alerts that do fire; the FPR is the public's-eye view of being
> stopped without cause. Both belong in any go/no-go decision on this
> threshold.
'''}
{review_section}
---

## Threshold Recommendation

{f"**Suggested threshold for ≤1% false-positive rate: `{recommended:.4f}`**" if recommended else "Unable to achieve ≤1% FPR with this dataset — consider stricter threshold or more data."}

> ⚠️ This is a SUGGESTION only. Do NOT auto-apply.
> An authorized admin must review this recommendation and manually update
> `{config_key}` in `.env` if they agree with it.
> Re-run validation after any threshold change{" — and this threshold change carries higher stakes than a ReID threshold change, since it directly controls when an automatic CRITICAL police alert fires on a real person" if is_face else ""}.

---

{fairness_md}

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
- The {"InsightFace" if is_face else "OSNet-IBN"} model is updated or fine-tuned on local footage
- `{config_key}`{"" if is_face else " or `REID_REVIEW_LOWER_THRESHOLD`"} is changed
- Camera hardware changes materially (resolution, field of view, lighting)
- After each major software release that touches the embedding pipeline

Command: `{run_command}`
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "face_watchlist_validation" if is_face else "reid_validation"
    filename = f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = output_dir / filename
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[OK] Report written to: {report_path}")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ReID offline validation harness")
    parser.add_argument(
        "--dataset-dir", type=Path, default=None,
        help="Path to same/different crop pair directory. "
             "If omitted, synthetic proxy data is used."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "reports",
        help="Directory for the output report. Default: reports/"
    )
    parser.add_argument(
        "--mode", choices=["reid", "face"], default="reid",
        help="'reid' validates the Day 6 OSNet-IBN body-ReID thresholds (default). "
             "'face' validates the Day 7 InsightFace watchlist-match threshold — "
             "extends this same harness rather than duplicating it, per the Day 7 spec."
    )
    args = parser.parse_args()

    if args.mode == "face":
        from backend.services.face_embedder import get_face_embedder
        embedder = get_face_embedder()
        default_dataset_dir = _PROJECT_ROOT / "data" / "test_face_pairs"
    else:
        from backend.services.reid_embedder import get_embedder
        embedder = get_embedder()
        default_dataset_dir = _PROJECT_ROOT / "data" / "test_pairs"

    dataset_dir = args.dataset_dir or default_dataset_dir

    is_synthetic = False
    if dataset_dir.exists():
        print(f"Loading pairs from {dataset_dir} ...")
        try:
            pairs, labels, metadata, pair_ids = _load_pairs_from_dir(dataset_dir, embedder=embedder)
            if not pairs:
                print("No valid pairs found in dataset_dir — falling back to synthetic.")
                is_synthetic = True
        except Exception as exc:
            print(f"Failed to load dataset: {exc} — falling back to synthetic.")
            is_synthetic = True
    else:
        if args.dataset_dir:
            print(f"⚠️ Dataset dir not found: {dataset_dir} — using synthetic proxy.")
        else:
            print(f"No --dataset-dir provided — using synthetic proxy data "
                  f"(expected default location: {dataset_dir}).")
        is_synthetic = True

    if is_synthetic:
        print("Generating synthetic proxy pairs (N=200 same + 200 different)...")
        pairs, labels, metadata, pair_ids = _generate_synthetic_pairs(200, 200)

    print(f"Pairs loaded: {len(pairs)} ({sum(labels)} same, {len(labels)-sum(labels)} different)")
    print("Computing similarities...")

    report_path = generate_report(
        pairs, labels, pair_ids, metadata, is_synthetic, args.output_dir, mode=args.mode
    )
    print("\nReport summary (first 40 lines):")
    lines = report_path.read_text(encoding="utf-8").split("\n")
    import sys
    out = sys.stdout
    for line in lines[:40]:
        try:
            out.write(line + "\n")
        except (UnicodeEncodeError, UnicodeDecodeError):
            out.write(line.encode('ascii', errors='replace').decode('ascii') + "\n")


if __name__ == "__main__":
    main()
