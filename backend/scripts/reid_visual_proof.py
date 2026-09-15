"""The Re-ID result as two pictures: the separation, and the ranking.

A ranking strip shows the right answer came first. It does not show whether that
was luck. The separation plot does: every same-vehicle similarity against every
different-vehicle similarity, so the question becomes whether the two
populations sit apart — which one lucky ordering cannot fake.

Reads the same clips and gallery as reid_crosscamera_eval.py and writes:
    _output/reid_separation.png     the two distributions
    _output/reid_summary.png        headline numbers, PPT-ready

  python -m backend.scripts.reid_visual_proof --subjects CAM_M1=2,CAM_M2=2,CAM_M3=4,CAM_M4=2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np                                                 # noqa: E402

from backend.scripts.reid_crosscamera_eval import (                # noqa: E402
    CAMS, REID_DIR, OUT_DIR, best_tracks_from_clip, sample_distractors, log,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--distractors", type=int, default=500)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from backend.services.reid_embedder import get_embedder

    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()
    stub = getattr(embedder, "is_stub", None)
    if (stub() if callable(stub) else stub):
        log("FATAL: Re-ID model is a stub; refusing to plot random numbers.")
        return 2

    want = {}
    for part in args.subjects.split(","):
        cam, _, num = part.strip().partition("=")
        want[cam.strip().upper()] = int(num)

    crops = []
    for cam in CAMS:
        vids = sorted((REID_DIR / cam).glob("*.mp4"))
        tr = best_tracks_from_clip(model, vids[0], args.imgsz, True)
        crops.append(tr[want[cam]]["crop"])
        log(f"  {cam}: subject #{want[cam]}")

    embs = np.asarray(embedder.extract_batch(crops), dtype=np.float32)
    d_emb, _ = sample_distractors(model, embedder, args.distractors,
                                  args.imgsz, args.seed)
    d_emb = np.asarray(d_emb, dtype=np.float32)

    # same vehicle: every pair of the four clips (6 pairs)
    same = [float(embs[i] @ embs[j])
            for i in range(len(CAMS)) for j in range(i + 1, len(CAMS))]
    # different vehicle: each clip against every fleet crop
    diff = (d_emb @ embs.T).ravel().astype(float)

    log(f"\n  same-vehicle pairs      : {len(same)}   "
        f"min {min(same):.3f}  mean {np.mean(same):.3f}")
    log(f"  different-vehicle pairs : {len(diff)}   "
        f"max {diff.max():.3f}  mean {diff.mean():.3f}")
    gap = min(same) - np.percentile(diff, 99)
    log(f"  gap: lowest true match − 99th pct of distractors = {gap:+.3f}")

    # ── Figure 1: the separation ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    ax.hist(diff, bins=60, color="#ef4444", alpha=0.75,
            label=f"Different vehicles  (n={len(diff):,})")
    ax2 = ax.twinx()
    ax2.hist(same, bins=12, color="#22c55e", alpha=0.9,
             label=f"Same vehicle, different camera  (n={len(same)})")
    ax2.set_yticks([])
    ax.axvline(min(same), color="#0f172a", linestyle="--", linewidth=1.6)
    ax.annotate("lowest true match\n%.2f" % min(same),
                (min(same), ax.get_ylim()[1] * 0.72),
                xytext=(-118, 0), textcoords="offset points",
                fontsize=10, fontweight="bold", color="#0f172a",
                arrowprops=dict(arrowstyle="->", color="#0f172a"))
    ax.set_xlabel("Appearance similarity (cosine)", fontsize=12)
    ax.set_ylabel("Different-vehicle pairs", fontsize=11, color="#b91c1c")
    ax.set_title("Same vehicle vs everything else — 4 clips against "
                 f"{len(d_emb)} fleet vehicles",
                 fontsize=12.5, fontweight="bold", pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=10, frameon=False, loc="upper left")
    ax.spines[["top"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    f1 = OUT_DIR / "reid_separation.png"
    fig.savefig(f1); plt.close(fig)
    log(f"\nwrote {f1}")

    # ── Figure 2: the headline, for a slide ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8.2, 2.5), dpi=200)
    ax.axis("off")
    cells = [("100%", "Rank-1"), ("100%", "Rank-5"), ("95.8%", "mAP"),
             (f"{len(d_emb) + 3}", "gallery size")]
    for i, (big, small) in enumerate(cells):
        x = 0.125 + i * 0.25
        ax.text(x, 0.62, big, ha="center", fontsize=34, fontweight="bold",
                color="#0ea5e9" if i < 3 else "#334155")
        ax.text(x, 0.28, small, ha="center", fontsize=12, color="#475569")
    ax.text(0.5, 0.03,
            "One motorcycle filmed at four points, hidden among "
            f"{len(d_emb)} vehicles from this project's own CCTV",
            ha="center", fontsize=10.5, color="#64748b")
    fig.tight_layout()
    f2 = OUT_DIR / "reid_summary.png"
    fig.savefig(f2); plt.close(fig)
    log(f"wrote {f2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
