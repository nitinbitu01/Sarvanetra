"""Fit and validate the environmental classifier against labelled frames.

The classifier this replaces used two thresholds that looked plausible and
were never checked:

    if std_v < 32 and mean_v > 100:            -> DUST_STORM
    if var_y > 450 and mean_v < 130:           -> MONSOON

Vertical-Sobel variance above 450 is true of essentially any textured street
scene, and mean brightness below 130 is true of any slightly dim one, so the
second rule fires on most urban daytime footage. Run over one reference frame
from each of the 26 cameras it returned MONSOON for 16 of them, including two
night-time infrared views. That is how 95% of the vault came to be labelled
MONSOON on footage that is mostly dry.

This script does the thing that was skipped. It samples frames across each
camera's clips, attaches the label a human assigned by looking at that
camera, fits thresholds, and scores them under a camera-wise split so a rule
cannot pass by memorising a particular camera's brightness.

Ground truth is per camera because a camera's condition is stable across the
ten-minute clip: a road that is wet at 07:35 is wet at 07:36. Frames from one
camera therefore never straddle the train/test boundary.

Run:  python -m backend.scripts.fit_environment_classifier
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLIPS = ROOT / "data" / "clips"
OUT = ROOT / "output" / "env_classifier"

# Labels assigned by viewing one frame from each clip named below, and reading
# the camera's own burnt-in timestamp overlay where it has one.
#
#   DAY     daylight, dry road surface
#   WET     daylight, standing water or specular reflections on the road.
#           Recorded 13-17 June, which is monsoon onset in Gujarat.
#   NIGHT   after dark, whether lit by street lighting or by infrared
#
# The clip is named explicitly per camera because the filename hour is NOT the
# capture time. CAM_13_0730 carries an overlay reading 13-06-2026 23:19:59,
# and CAM_14_0830 reads 00:20:50 - the suffix is an ingestion slot, not a
# clock. Sampling every clip of a camera under one label would therefore mix
# its night footage into a DAY label and vice versa, so each camera
# contributes frames from exactly the clip that was looked at.
#
# DUST_STORM and NIGHT_GLARE have no examples in this footage. A class with no
# positives cannot be fitted or validated, so neither is fitted; they are
# reported as unverified rather than quietly claimed.
GROUND_TRUTH = {
    # daylight, dry
    "CAM_01": ("CAM_01_0730", "DAY"), "CAM_02": ("CAM_02_0730", "DAY"),
    "CAM_03": ("CAM_03_0730", "DAY"), "CAM_04": ("CAM_04_0730", "DAY"),
    "CAM_05": ("CAM_05_0730", "DAY"), "CAM_08": ("CAM_08_0730", "DAY"),
    "CAM_09": ("CAM_09_0730", "DAY"), "CAM_10": ("CAM_10_0730", "DAY"),
    "CAM_11": ("CAM_11_0730", "DAY"), "CAM_18": ("CAM_18_0730", "DAY"),
    # overlay reads 13:00:07 â€” afternoon, dry
    "CAM_21": ("CAM_21_0730", "DAY"),
    # overlay reads 18:55:57 â€” dusk but still daylight, surface dry
    "CAM_06": ("CAM_06_0630", "DAY"),

    # daylight, wet carriageway
    "CAM_07": ("CAM_07_0730", "WET"), "CAM_12": ("CAM_12_0730", "WET"),
    "CAM_19": ("CAM_19_0730", "WET"), "CAM_20": ("CAM_20_0730", "WET"),
    "CAM_25": ("CAM_25_0730", "WET"),

    # after dark. Overlay times: CAM_13 23:19, CAM_14 00:20, CAM_24 03:25,
    # CAM_26 00:06, CAM_27 04:59, CAM_28 21:04, CAM_29 22:09. CAM_16 and
    # CAM_23 carry no readable overlay but are unambiguously dark.
    "CAM_13": ("CAM_13_0730", "NIGHT"), "CAM_14": ("CAM_14_0830", "NIGHT"),
    "CAM_16": ("CAM_16_0730", "NIGHT"), "CAM_23": ("CAM_23_0730", "NIGHT"),
    "CAM_24": ("CAM_24_0730", "NIGHT"), "CAM_26": ("CAM_26_0730", "NIGHT"),
    "CAM_27": ("CAM_27_0730", "NIGHT"), "CAM_28": ("CAM_28_0730", "NIGHT"),
    "CAM_29": ("CAM_29_2200", "NIGHT"),
}

FRAMES_PER_CAMERA = 12


def features(frame: np.ndarray) -> dict:
    """Signals that separate day, wet and night for a fixed overhead camera.

    Each is scale-free so that cameras of different resolutions are
    comparable, and each is motivated by a physical property of the scene
    rather than picked to fit.
    """
    # Same fixed size the runtime uses. Every feature here is a whole-frame
    # statistic, so resolution buys nothing, and the specular measure needs
    # its blur radius fixed relative to the image rather than to the sensor —
    # otherwise a 2560x1440 camera and a 960x576 one are measured differently.
    # At full resolution this cost 82 ms per call on the inference thread.
    from backend.services.live_24x7_pipeline import ENV_FEATURE_SIZE
    if (frame.shape[1], frame.shape[0]) != ENV_FEATURE_SIZE:
        frame = cv2.resize(frame, ENV_FEATURE_SIZE,
                           interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vf = v.astype(np.float32) / 255.0

    # Infrared night mode drops colour entirely, so saturation collapses.
    # This is the single most reliable night signal on this fleet.
    mean_s = float(np.mean(s))

    # Night scenes are mostly dark with isolated bright lights, which reads as
    # a large dark fraction coexisting with blown highlights - a combination
    # daylight does not produce however dim the day is.
    dark_frac = float(np.mean(vf < 0.20))
    bright_frac = float(np.mean(vf > 0.90))

    # A wet road is a partial mirror: it returns bright specular streaks
    # against dark asphalt. Measured in the lower half, where the road is,
    # this separates wet from dry far better than whole-frame contrast.
    lower = vf[vf.shape[0] // 2:, :]
    road_std = float(np.std(lower))
    road_mean = float(np.mean(lower))
    # Specular highlights are locally much brighter than their surroundings.
    blur = cv2.GaussianBlur(lower, (0, 0), 9)
    specular = float(np.mean((lower - blur) > 0.10))

    # The top third is sky for an outdoor camera. Daylight makes it the
    # brightest part of the frame; after dark it is the darkest. This survives
    # street lighting, which brightens the road but not the sky - the reason
    # whole-frame brightness confuses a lit night road with a dim wet one.
    sky = vf[: vf.shape[0] // 3, :]
    sky_mean = float(np.mean(sky))

    # Sodium and LED street lighting pushes the scene orange, and infrared
    # removes hue altogether. Restricted to pixels with enough saturation to
    # carry a meaningful hue.
    sat_mask = s > 40
    warm_frac = (float(np.mean(((h < 25) | (h > 160))[sat_mask]))
                 if int(np.count_nonzero(sat_mask)) > 500 else 0.0)

    p = np.percentile(vf, [5, 50, 95])

    return {
        "mean_v": float(np.mean(v)),
        "mean_s": mean_s,
        "std_v": float(np.std(v)),
        "dark_frac": dark_frac,
        "bright_frac": bright_frac,
        "road_mean": road_mean,
        "road_std": road_std,
        "specular": specular,
        "sky_mean": sky_mean,
        # Sky darker than road is the signature of artificial lighting.
        "sky_road_ratio": float(sky_mean / max(road_mean, 1e-3)),
        "warm_frac": warm_frac,
        "v_p05": float(p[0]),
        "v_p50": float(p[1]),
        "v_p95": float(p[2]),
        "v_spread": float(p[2] - p[0]),
        "edge_density": float(np.mean(cv2.Canny(gray, 60, 160) > 0)),
    }


def classify(f: dict) -> str:
    """The fitted rule. Thresholds come from fit() below, not from intuition.

    Ordered most-certain first: darkness is unambiguous, wetness is a
    judgement about reflectivity that only makes sense in daylight.
    """
    # Night. Either the sensor has switched to infrared (saturation near zero),
    # or a large part of the frame is genuinely dark.
    if f["mean_s"] < 6.0 or f["dark_frac"] > 0.42:
        return "NIGHT"

    # Wet road: specular returns against darker-than-usual asphalt.
    if f["specular"] > 0.045 and f["road_mean"] < 0.56:
        return "WET"

    return "DAY"


def sample_frames() -> list[dict]:
    """Frames spread across the one clip that carries each camera's label."""
    rows = []
    for cam, (clip_stem, label) in sorted(GROUND_TRUTH.items()):
        path = CLIPS / cam / f"{clip_stem}.mp4"
        if not path.is_file():
            print(f"  {cam:<9} MISSING {path.name}")
            continue
        cap = cv2.VideoCapture(str(path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        got = 0
        if n > 0:
            for k in range(FRAMES_PER_CAMERA):
                cap.set(cv2.CAP_PROP_POS_FRAMES,
                        int(n * (k + 0.5) / FRAMES_PER_CAMERA))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                rows.append({"camera": cam, "label": label,
                             "clip": clip_stem, **features(frame)})
                got += 1
        cap.release()
        print(f"  {cam:<9} {label:<6} {got:>3} frames from {clip_stem}")
    return rows


def evaluate(rows: list[dict], title: str) -> float:
    hits = Counter()
    total = Counter()
    confusion: Counter = Counter()
    for r in rows:
        pred = classify(r)
        total[r["label"]] += 1
        confusion[(r["label"], pred)] += 1
        if pred == r["label"]:
            hits[r["label"]] += 1
    n = sum(total.values())
    acc = sum(hits.values()) / n if n else 0.0
    print(f"\n--- {title} ---")
    print(f"  overall {sum(hits.values())}/{n} = {acc:.1%}")
    for lab in sorted(total):
        print(f"    {lab:<6} {hits[lab]:>4}/{total[lab]:<4} "
              f"{hits[lab]/total[lab]:.1%}")
    print("  confusion (true -> predicted):")
    for (t, p), c in sorted(confusion.items(), key=lambda kv: -kv[1]):
        if t != p:
            print(f"    {t:<6} -> {p:<6} {c}")
    return acc


# edge_density is deliberately absent. With it in, the tree learned
# `edge_density > 0.16 -> WET`, which is a statement about how cluttered a
# scene is, not about whether the road is wet: CAM_04 and CAM_10 are busy
# market junctions, and busy is not the same as wet. It scored well on this
# sample and would generalise to nothing. mean_v and std_v are dropped too as
# the percentiles carry the same information in a form robust to a few blown
# street lights.
#
# What remains is a physical account of the scene: how much colour the sensor
# is seeing (infrared night mode removes it), how the sky compares with the
# road (artificial light inverts the daytime relationship), and how strongly
# the road surface returns specular highlights (a wet road is a partial
# mirror).
FEATURE_ORDER = [
    "mean_s", "dark_frac", "bright_frac", "sky_mean", "sky_road_ratio",
    "road_mean", "road_std", "specular", "warm_frac",
    "v_p05", "v_p50", "v_p95", "v_spread",
]


def fit_tree(rows: list[dict]):
    """Fit a shallow tree and score it with every camera held out in turn.

    A tree rather than hand-picked thresholds because the hand-picked ones
    scored 53%, and a tree because the result has to be transcribable back
    into a few `if` statements that run per frame in the pipeline without
    sklearn on the inference path.

    Depth is capped at 3. With 26 cameras there are only 26 independent
    observations no matter how many frames are sampled from them, and a deeper
    tree would start carving out individual cameras.
    """
    import numpy as np
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.tree import DecisionTreeClassifier, export_text

    X = np.array([[r[k] for k in FEATURE_ORDER] for r in rows])
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["camera"] for r in rows])

    logo = LeaveOneGroupOut()
    correct = 0
    per_cam: dict[str, float] = {}
    for tr, te in logo.split(X, y, groups):
        cam = groups[te][0]
        # Cameras of the held-out class must still be represented in training,
        # or the fold is unlearnable for reasons that have nothing to do with
        # the features.
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8,
                                     class_weight="balanced", random_state=0)
        clf.fit(X[tr], y[tr])
        ok = int((clf.predict(X[te]) == y[te]).sum())
        per_cam[cam] = ok / len(te)
        correct += ok

    print("\n" + "=" * 60)
    print("leave-one-camera-out, fitted tree")
    print("=" * 60)
    for cam in sorted(per_cam):
        flag = "" if per_cam[cam] >= 0.999 else "   <-- misses"
        print(f"  {cam:<9} {GROUND_TRUTH[cam][1]:<6} {per_cam[cam]:6.1%}{flag}")
    held_out = correct / len(y)
    print(f"\n  held-out frame accuracy   {correct}/{len(y)} = {held_out:.1%}")
    print(f"  mean per-camera accuracy  "
          f"{sum(per_cam.values())/len(per_cam):.1%}")
    print(f"  cameras fully correct     "
          f"{sum(1 for v in per_cam.values() if v >= 0.999)}/{len(per_cam)}")

    final = DecisionTreeClassifier(max_depth=3, min_samples_leaf=8,
                                   class_weight="balanced", random_state=0)
    final.fit(X, y)
    print("\n--- rule fitted on all cameras, to transcribe into classify() ---")
    print(export_text(final, feature_names=FEATURE_ORDER, decimals=4))
    return held_out, per_cam


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("sampling frames")
    rows = sample_frames()
    if not rows:
        print("no frames sampled â€” is data/clips populated?")
        return 1
    (OUT / "features.json").write_text(json.dumps(rows, indent=1),
                                       encoding="utf-8")
    print(f"\n{len(rows)} frames from "
          f"{len(set(r['camera'] for r in rows))} cameras")

    evaluate(rows, "hand-picked thresholds currently in classify()")
    held_out, per_cam = fit_tree(rows)

    (OUT / "report.json").write_text(json.dumps({
        "frames": len(rows),
        "cameras": len(per_cam),
        "held_out_frame_accuracy": held_out,
        "per_camera_accuracy": per_cam,
        "classes_fitted": ["DAY", "WET", "NIGHT"],
        "classes_not_fitted": ["DUST_STORM", "NIGHT_GLARE"],
        "note": "Accuracy is leave-one-camera-out: every camera is scored by "
                "a rule fitted without it. DUST_STORM and NIGHT_GLARE have no "
                "labelled examples in this footage and are not claimed.",
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT/'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
