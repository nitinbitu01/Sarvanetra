"""Fit the two pixel rules the environment classifier still needs, and save them.

Day versus night is answered by clip_clock: OCR the camera's burnt-in clock,
convert with the camera's GPS, read the sun's elevation. That was exact on all
12 of 18 labelled clips whose clock was legible, including the indoor and
infrared views a pixel model cannot do anything with.

Two jobs are left for pixels:

  wet        whether a daylight road surface is wet. Not recoverable from a
             clock at all.
  daylight   a fallback for the 6 clips whose overlay could not be read. All
             six are bright outdoor daytime views, which is the easy end of
             the problem, but the threshold should still be fitted rather
             than guessed.

Both are fitted with leave-one-camera-out, because 26 cameras is 26
independent observations however many frames are sampled from them.

Run:  python -m backend.scripts.fit_wet_model
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "output" / "env_classifier" / "features.json"
OUT = ROOT / "output" / "env_classifier" / "pixel_models.json"

WET_FEATURES = ["specular", "road_mean", "road_std", "sky_road_ratio",
                "v_p05", "v_p50", "v_p95", "mean_s"]
DAY_FEATURES = ["sky_mean", "sky_road_ratio", "dark_frac", "bright_frac",
                "v_p50", "v_p95", "mean_s"]


def fit(rows: list[dict], feats: list[str], positive: str,
        subset: list[str] | None, name: str) -> dict:
    """Fit one binary rule and report what it scores with each camera held out."""
    data = [r for r in rows if subset is None or r["label"] in subset]
    X = np.array([[r[f] for f in feats] for r in data])
    y = np.array([1 if r["label"] == positive else 0 for r in data])
    g = np.array([r["camera"] for r in data])

    correct = 0
    per_cam: dict[str, float] = {}
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        ok = int((clf.predict((X[te] - mu) / sd) == y[te]).sum())
        per_cam[g[te][0]] = ok / len(te)
        correct += ok

    acc = correct / len(y)
    print(f"\n--- {name} ---")
    print(f"  {len(data)} frames, {len(per_cam)} cameras, "
          f"{int(y.sum())} positive")
    print(f"  leave-one-camera-out: {correct}/{len(y)} = {acc:.1%}")
    misses = {c: v for c, v in per_cam.items() if v < 0.999}
    print(f"  cameras fully correct: {len(per_cam)-len(misses)}/{len(per_cam)}")
    for c, v in sorted(misses.items(), key=lambda kv: kv[1]):
        truth = next(r["label"] for r in data if r["camera"] == c)
        print(f"    {c:<9} {truth:<6} {v:6.1%}")

    mu, sd = X.mean(0), X.std(0) + 1e-9
    final = LogisticRegression(max_iter=2000, class_weight="balanced")
    final.fit((X - mu) / sd, y)
    return {
        "features": feats,
        "mean": mu.tolist(),
        "std": sd.tolist(),
        "coef": final.coef_[0].tolist(),
        "intercept": float(final.intercept_[0]),
        "positive_class": positive,
        "held_out_accuracy": acc,
        "cameras": len(per_cam),
        "frames": len(data),
        "per_camera_accuracy": per_cam,
    }


def main() -> int:
    if not FEATURES.is_file():
        print(f"missing {FEATURES} — run fit_environment_classifier first")
        return 1
    rows = json.loads(FEATURES.read_text(encoding="utf-8"))
    print(f"{len(rows)} labelled frames")

    models = {
        # Wet is only meaningful in daylight, so night frames are excluded
        # rather than being asked to vote on road reflectance.
        "wet": fit(rows, WET_FEATURES, "WET", ["DAY", "WET"],
                   "wet vs dry (daylight frames only)"),
        # The daylight fallback treats WET as daylight, which it is.
        "daylight": fit([{**r, "label": ("DAYLIGHT" if r["label"] in
                                         ("DAY", "WET") else "NIGHT")}
                         for r in rows],
                        DAY_FEATURES, "DAYLIGHT", None,
                        "daylight vs night (pixel fallback only)"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(models, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("""
These two numbers are what the vault's lighting_condition column is worth on
this footage. Day/night reaches them only when the clip clock is unreadable;
where it is readable the answer is arithmetic, and was right on 12 of 12.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
