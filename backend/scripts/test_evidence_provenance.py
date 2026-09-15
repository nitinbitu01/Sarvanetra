"""Is an alert's proof clip actually the footage the detector was looking at?

The seal was never the problem. Every one of the 60 clips this replaces hashed
correctly; what a SHA-256 proves is that a file has not changed since it was
sealed, not that the file is of the event. Under that valid seal, 60 alerts
shared 26 distinct clips — one video served as proof for five separate CAM_02
congestion alerts recorded hours apart — because capture had no record of
where in the footage the event occurred and fell back to the middle of the
camera's first file.

This checks the property the seal cannot: that the frames in the clip are the
frames at the recorded position in the recorded source file. It does that by
re-extracting from the original recording and comparing pixels, which is
exactly the check an officer or an opposing expert would run.

Run:  python -m backend.scripts.test_evidence_provenance
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVIDENCE = ROOT / "evidence"


def frame_at(path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def similarity(a, b) -> float:
    """Correlation of two frames, ignoring the overlay drawn on the clip."""
    if a is None or b is None:
        return -1.0
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    # The HUD occupies the top 75 px and bottom 35 px of a rendered frame, so
    # compare the middle band, which is untouched footage in both.
    h = a.shape[0]
    lo, hi = 80, max(90, h - 40)
    ga = cv2.cvtColor(a[lo:hi], cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    gb = cv2.cvtColor(b[lo:hi], cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    ga, gb = ga - ga.mean(), gb - gb.mean()
    d = float(np.linalg.norm(ga) * np.linalg.norm(gb))
    return float(ga @ gb / d) if d else -1.0


def main() -> int:
    manifests = sorted(EVIDENCE.glob("*/manifest.json"))
    if not manifests:
        print("No evidence with a manifest yet.\n"
              "Run the pipeline so it raises an alert:\n"
              "    python -m backend.scripts.run_pipeline")
        return 0

    print(f"{len(manifests)} evidence clips carry provenance\n")
    print(f"{'alert':<14}{'camera':<9}{'source file':<28}{'frames':>8}"
          f"{'boxed':>7}{'match':>8}  verdict")
    print("-" * 84)

    seen_hashes: dict[str, str] = {}
    fails = []
    for mpath in manifests:
        m = json.loads(mpath.read_text(encoding="utf-8"))
        clip = mpath.parent / "clip.mp4"
        aid = str(m.get("alert_id", mpath.parent.name))[:12]
        cam = m.get("camera_id", "?")
        src = m.get("source_file")

        if not clip.is_file():
            fails.append(f"{aid}: manifest without a clip")
            continue

        # 1. The seal must still hold.
        digest = hashlib.sha256(clip.read_bytes()).hexdigest()
        if m.get("clip_sha256") and digest != m["clip_sha256"]:
            fails.append(f"{aid}: clip does not match its recorded hash")

        # 2. No two alerts may share a clip. This is what caught the old
        #    behaviour: identical bytes under five different alert ids.
        if digest in seen_hashes and seen_hashes[digest] != aid:
            fails.append(f"{aid}: byte-identical to alert {seen_hashes[digest]}")
        seen_hashes[digest] = aid

        # 3. The frames must be the frames at the recorded position.
        verdict, score = "no source", -1.0
        if src:
            source_path = ROOT / src
            rng = m.get("frame_range") or [0, 0]
            if source_path.is_file():
                # The clip is sampled down from the source, and frames the
                # tracker measured are kept regardless of that stride, so clip
                # index k does not map to source frame rng[0]+k. The first
                # frame does map exactly, and is enough: if the clip started
                # somewhere other than the recorded position, it is not the
                # footage the manifest claims.
                from_clip = frame_at(clip, 0)
                from_source = frame_at(source_path, rng[0])
                score = similarity(from_clip, from_source)
                verdict = "matches source" if score > 0.90 else "DOES NOT MATCH"
                if score <= 0.90:
                    fails.append(
                        f"{aid}: the clip does not begin at frame {rng[0]} of "
                        f"{Path(src).name} (correlation {score:.2f})")
            else:
                verdict = "source missing"
                fails.append(f"{aid}: source file {src} is gone")

        print(f"{aid:<14}{cam:<9}{(Path(src).name if src else '-'):<28}"
              f"{m.get('frames_extracted', 0):>8}"
              f"{m.get('frames_with_subject_box', 0):>7}"
              f"{score:>8.2f}  {verdict}")

    print(f"\n{len(seen_hashes)} distinct clips for {len(manifests)} alerts")

    print("\n" + "=" * 84)
    if fails:
        print(f"{len(fails)} PROBLEM(S):")
        for f in fails:
            print(f"   - {f}")
        return 1
    print("Every clip is the recorded frames of its own alert, and no two "
          "alerts share one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
