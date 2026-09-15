"""Cross-camera identity from appearance alone — no plate required.

WHY THIS EXISTS
  Every cross-camera link this system currently makes is a plate link. The
  plate is the identity: `live_24x7_pipeline` writes a JourneyEvent only when
  `vt.plate_text` exists, and sets `reid_id = plate`. That is a hard ceiling,
  because a plate is readable on a minority of tracks — the committed operating
  point covers 61.4% of vehicles and only above 80px of plate width. Every
  vehicle below that line is invisible to journey search no matter how many
  cameras saw it.

  The appearance channel already exists (OSNet-IBN, 512-d, fine-tuned on
  Market1501) and is already used to *rank* candidates. It was never used to
  *assign* an identity. This does that: it links tracks across cameras by
  appearance, gives each cluster a global id, and writes the same JourneyEvent
  rows the plate path writes — so the existing Journeys page renders the route
  with no change to the frontend.

HOW THE THRESHOLD IS SET — this is the part that decides whether the number
means anything.
  A similarity threshold picked by hand, or tuned until the demo clips link, is
  not a measurement; it is a wish. So it is derived instead, from vehicles that
  are definitely NOT the same vehicle: several hundred crops sampled from this
  project's own CCTV, from different clips. Every cross-clip pair among those is
  an impostor pair, which gives an empirical impostor similarity distribution.
  The threshold is the quantile of that distribution corresponding to a chosen
  false-accept rate (default 0.1%). Nothing about the subjects influences it.

HOW ACCURACY IS MEASURED — likewise not asserted.
  The same run reads plates the ordinary way (the real ANPR engine, the real
  temporal voter, the same gates CAM_09 uses). Wherever two tracks both carry a
  confirmed plate, the plate says whether they are the same vehicle, and the
  appearance decision can be scored against it:

      precision = of the cross-camera links made between two plated tracks,
                  how many joined two tracks with the SAME plate
      recall    = of the cross-camera pairs that share a plate,
                  how many appearance actually linked

  Plates are the ground truth and appearance is the thing under test, so the
  two channels are genuinely independent here. Where there are too few plated
  pairs to support a percentage, the script says so and prints the count
  instead of a rate — a precision of "1/1 = 100%" is not a result.

WHAT THIS IS NOT
  It is not a change to the live pipeline. It does not run on CAM_09, does not
  touch the running capture, and holds no lock the pipeline needs; it opens its
  own short write transaction only under --commit. It reads clips off disk.

  python -m backend.scripts.appearance_linker --cameras CAM_M1,CAM_M2,CAM_M3,CAM_M4
  python -m backend.scripts.appearance_linker --cameras ... --commit
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402

CLIPS_DIR = ROOT / "data" / "clips"
REID_DIR = ROOT / "data" / "reid_test"
OUT_DIR = ROOT / "output" / "appearance_linker"

VEHICLE_CLASSES = [2, 3, 5, 7]                  # car, motorcycle, bus, truck
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# A track below this is too small for either channel to say anything.
MIN_BOX_W, MIN_BOX_H = 40, 48


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── data ────────────────────────────────────────────────────────────────────
# eq=False on purpose: the generated __eq__ would compare `emb`, and comparing
# two numpy arrays returns an array, so any `track in [tracks]` test raises
# "truth value of an array is ambiguous". Identity is what is wanted anyway —
# `key` is the value-level identifier.
@dataclass(eq=False)
class Track:
    camera: str
    local_id: int
    cls_id: int
    n_frames: int = 0
    first_frame: int = 0
    last_frame: int = 0
    fps: float = 25.0
    # Which frames this track was visible in. Two tracks sharing a frame are
    # necessarily different vehicles, which is the only source of labelled
    # negatives available in this footage — see in_domain_impostors().
    frames: set = field(default_factory=set)
    crops: list = field(default_factory=list)       # (sharpness, crop)
    emb: Optional[np.ndarray] = None
    best_crop: Optional[np.ndarray] = None
    plate: Optional[str] = None
    plate_conf: Optional[float] = None

    @property
    def key(self) -> str:
        return f"{self.camera}#{self.local_id}"

    @property
    def cls_name(self) -> str:
        return CLASS_NAMES.get(self.cls_id, "vehicle")

    def seconds(self, frame_idx: int) -> float:
        return frame_idx / max(self.fps, 1.0)


def _sharpness(crop: np.ndarray) -> float:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def _clip_for(camera: str) -> Optional[Path]:
    """The re-ID test clips live in their own tree; the fleet's live in clips/."""
    for base in (REID_DIR / camera, CLIPS_DIR / camera):
        vids = sorted(base.glob("*.mp4"))
        if vids:
            return vids[0]
    return None


# ── stage 1: every track in one clip, embedded, with a plate if readable ────
def scan_clip(model, anpr, embedder, camera: str, path: Path, *,
              imgsz: int, stride: int, topk: int, read_plates: bool,
              tid_base: int) -> list[Track]:
    """Detect, track, read and embed EVERY vehicle in one clip.

    Deliberately not "find the subject". The value of this run is that it makes
    no decision about which vehicle matters — a linker that only works on a
    vehicle somebody pointed at has not been tested.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log(f"  {camera}: cannot open {path.name}")
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)

    model.predictor = None                       # fresh tracker state per clip
    tracks: dict[int, Track] = {}
    frame_i = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1
        if stride > 1 and frame_i % stride:
            continue

        r = model.track(frame, persist=True, classes=VEHICLE_CLASSES, conf=0.35,
                        tracker="botsort.yaml", imgsz=imgsz, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue

        for box, tid, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                 r.boxes.id.cpu().numpy().astype(int),
                                 r.boxes.cls.cpu().numpy().astype(int)):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            t = tracks.get(int(tid))
            if t is None:
                t = Track(camera=camera, local_id=int(tid), cls_id=int(cls),
                          first_frame=frame_i, fps=fps)
                tracks[int(tid)] = t
            t.n_frames += 1
            t.last_frame = frame_i
            t.frames.add(frame_i)

            # Keep the sharpest few views rather than the last one. A single
            # crop is a lottery: the frame a track happens to end on is often
            # the one where it is half out of shot.
            t.crops.append((_sharpness(crop) * (crop.shape[0] * crop.shape[1]) ** 0.5,
                            crop.copy()))
            if len(t.crops) > topk * 3:
                t.crops.sort(key=lambda c: -c[0])
                del t.crops[topk:]

            if read_plates:
                # The real engine, the real per-track voter, the same gates
                # CAM_09 runs. track_id is offset so two cameras cannot share
                # the engine's per-track state — sharing it makes the engine
                # return the first track's locked plate for every later
                # vehicle, which looks like a spectacular read rate and is one
                # plate repeated.
                try:
                    res = anpr.process_vehicle_track(frame, [x1, y1, x2, y2],
                                                     int(cls),
                                                     tid_base + int(tid))
                except Exception:
                    res = None
                # Keep the running result. finalize_track() below pops the
                # voter and returns ONLY the voter's verdict, discarding
                # _confirmed_plates — so a plate that locked mid-track but
                # whose final vote falls under min_confidence is lost entirely
                # if it was not captured here.
                if res and res.get("plate"):
                    c = float(res.get("confidence") or 0.0)
                    if t.plate is None or c > (t.plate_conf or 0.0):
                        t.plate = str(res["plate"]).strip().upper()
                        t.plate_conf = c
    cap.release()

    out: list[Track] = []
    for tid, t in tracks.items():
        if not t.crops:
            continue
        t.crops.sort(key=lambda c: -c[0])
        del t.crops[topk:]
        t.best_crop = t.crops[0][1]

        if read_plates:
            # Voter consensus outranks any single frame's read, so it wins when
            # it produced one; otherwise the best in-track read above stands.
            try:
                res = anpr.finalize_track(tid_base + tid)
            except Exception:
                res = None
            if res and res.get("plate"):
                t.plate = str(res["plate"]).strip().upper()
                t.plate_conf = float(res.get("confidence") or 0.0)
        out.append(t)

    # One vector per track: the mean of its sharpest views, re-normalised.
    # Averaging before comparing is what stops a long track outvoting a short
    # one, and it is measurably steadier than any single crop.
    if out:
        flat, spans = [], []
        for t in out:
            spans.append((len(flat), len(t.crops)))
            flat.extend(c for _, c in t.crops)
        embs = np.asarray(embedder.extract_batch(flat), dtype=np.float32)
        for t, (off, n) in zip(out, spans):
            v = embs[off:off + n].mean(axis=0)
            norm = float(np.linalg.norm(v))
            t.emb = (v / norm).astype(np.float32) if norm > 1e-9 else None
        out = [t for t in out if t.emb is not None]

    plated = sum(1 for t in out if t.plate)
    log(f"  {camera}: {len(out)} tracks, {plated} with a confirmed plate "
        f"({path.name})")
    return out


# ── stage 2: the threshold, from vehicles that are definitely different ─────
def sample_impostors(model, embedder, exclude: set[str], *, n_target: int,
                     imgsz: int, seed: int):
    """Vehicle crops from fleet cameras outside the run, tagged by camera.

    Only CROSS-CAMERA pairs among these are treated as impostors: two crops
    from one camera may genuinely be the same vehicle twice, and letting those
    into the impostor set would inflate its tail and raise the threshold for
    the wrong reason.
    """
    rng = random.Random(seed)
    clips = [p for p in sorted(CLIPS_DIR.glob("CAM_*/*.mp4"))
             if p.parent.name not in exclude]
    rng.shuffle(clips)

    crops: list[np.ndarray] = []
    origin: list[str] = []
    for clip in clips:
        if len(crops) >= n_target:
            break
        cap = cv2.VideoCapture(str(clip))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 30:
            cap.release()
            continue
        for idx in np.linspace(total * 0.15, total * 0.85, 12).astype(int):
            if len(crops) >= n_target:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if not ok or f is None:
                continue
            r = model.predict(f, classes=VEHICLE_CLASSES, conf=0.4,
                              imgsz=imgsz, verbose=False)[0]
            if r.boxes is None:
                continue
            for box in r.boxes.xyxy.cpu().numpy():
                if len(crops) >= n_target:
                    break
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(f.shape[1] - 1, x2), min(f.shape[0] - 1, y2)
                if x2 - x1 < MIN_BOX_W or y2 - y1 < MIN_BOX_H:
                    continue
                c = f[y1:y2, x1:x2]
                if c.size:
                    crops.append(c.copy())
                    origin.append(clip.parent.name)
        cap.release()

    if len(crops) < 60:
        return None, None, {}

    embs = np.asarray(embedder.extract_batch(crops), dtype=np.float32)
    org = np.asarray(origin)
    sims = embs @ embs.T
    iu = np.triu_indices(len(embs), k=1)
    cross = org[iu[0]] != org[iu[1]]
    impostor = sims[iu][cross]
    if impostor.size < 500:
        return None, None, {}

    stats = {
        "crops": len(crops),
        "cameras": int(len(set(origin))),
        "pairs": int(impostor.size),
        "mean": float(impostor.mean()),
        "p50": float(np.quantile(impostor, 0.50)),
        "p99": float(np.quantile(impostor, 0.99)),
        "p999": float(np.quantile(impostor, 0.999)),
        "max": float(impostor.max()),
    }
    return embs, org, stats


def in_domain_impostors(by_cam: dict[str, list[Track]]):
    """Similarity between vehicles that are certainly NOT the same vehicle,
    measured inside this run's own footage.

    WHY THE FLEET-SAMPLED IMPOSTORS WERE NOT ENOUGH — measured, on the proof
    sheet, not in theory.
      Calibrating against crops from other fleet cameras said a threshold of
      0.7175 would produce 0.36 false links across the whole run. It then
      produced at least four false clusters, and the contact sheet showed what
      they were: every black vehicle in one group, every white one in another,
      four different motorcycles in a third. Grouped by colour and body type,
      not by identity.

      The sampled impostors came from different cameras, hours and lighting,
      so they were easy to tell apart and the tail of that distribution sat far
      too low. Vehicles on one road in one afternoon share illumination, camera
      angle, resolution and background. Unrelated ones look much more alike
      than any cross-camera sample suggests, and that is precisely the
      condition the threshold has to survive.

    THE LABEL-FREE NEGATIVE
      Two tracks visible in the SAME FRAME cannot be the same vehicle. No
      annotation is required and no assumption is made: it is a geometric fact.
      Those pairs are drawn from exactly the domain being tested, so their
      similarity distribution is the right one to set a threshold against.
    """
    sims: list[float] = []
    per_cam: dict[str, int] = {}
    for cam, ts in by_cam.items():
        n = 0
        for a, b in itertools.combinations(ts, 2):
            if a.frames & b.frames:              # on screen together
                sims.append(float(a.emb @ b.emb))
                n += 1
        per_cam[cam] = n
    if not sims:
        return None, {}
    arr = np.asarray(sims, dtype=np.float32)
    return arr, {
        "pairs": int(arr.size),
        "per_camera": per_cam,
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(arr.max()),
    }


def threshold_from_in_domain(imp: np.ndarray, real_pairs: int, budget: float):
    """Raise the threshold until this run is expected to make < budget false
    links, using the in-domain impostor rate.

    The rate is applied to every comparison the run makes. Mutual-best
    filtering will reject some of those, so this OVERSTATES the false links
    slightly — the safe direction for a system that names vehicles to police.
    """
    order = np.sort(imp)[::-1]
    trace = []
    chosen = None
    # Candidate thresholds walk down the impostor tail; the first (highest)
    # value whose implied count is under budget wins.
    qs = [1.0, 0.9999, 0.999, 0.995, 0.99, 0.95, 0.9]
    cands = sorted({float(np.quantile(imp, q)) for q in qs}
                   | {float(order[0]) + 0.01, float(order[0]) + 0.03},
                   reverse=True)
    for thr in cands:
        rate = float((imp >= thr).mean())
        expected = rate * real_pairs
        trace.append((thr, rate, expected))
        if expected <= budget:
            chosen = thr
    # `chosen` ends as the LOWEST threshold still under budget, which is the
    # most permissive safe one.
    return chosen, trace


def mutual_best_pairs(groups: dict[str, np.ndarray], threshold: float):
    """Mutual best match above a threshold, between every pair of groups.

    Mutual is what stops one photogenic track collecting the whole fleet: A in
    group X may only claim B in group Y if B, looking back across all of X,
    also picks A. A one-way nearest neighbour would link every unmatched
    vehicle to whatever happened to be closest.
    """
    out = []
    names = [n for n in groups if len(groups[n])]
    for cx, cy in itertools.combinations(names, 2):
        A, B = groups[cx], groups[cy]
        S = A @ B.T
        best_a = S.argmax(axis=1)
        best_b = S.argmax(axis=0)
        for i, j in enumerate(best_a):
            if best_b[j] != i:
                continue
            s = float(S[i, j])
            if s >= threshold:
                out.append((cx, int(i), cy, int(j), s))
    return out


def _cross_pair_count(sizes) -> int:
    return sum(a * b for a, b in itertools.combinations(list(sizes), 2))


def calibrate_threshold(d_embs, d_cams, real_pairs: int, budget: float,
                        stats: dict):
    """Set the threshold so the WHOLE RUN is expected to make < `budget`
    false links — not so that a single comparison is 99.9% safe.

    WHY A PER-PAIR RATE IS THE WRONG KNOB, measured
      The first version of this script set the threshold at the 99.9th
      percentile of the impostor distribution and called that a 0.1%
      false-accept rate. It is — per comparison. But a run comparing 131 tracks
      on one camera against 88 on another makes 11,528 comparisons, and
      0.1% of 11,528 is about 11 expected false links. That run returned
      exactly 12 links, every one of them scoring BELOW the highest similarity
      seen between two vehicles known to be different. In other words the
      entire result was consistent with noise, and it looked like a finding.

    WHAT IS DONE INSTEAD
      The identical matching procedure — same mutual-best rule, same threshold
      — is run over the impostor crops, grouped by their own camera. Every link
      it produces there is false by construction. That measures the false-link
      rate of the actual algorithm rather than of a single comparison, and it
      needs no assumption about independence, which mutual-best violates
      anyway. The rate is then scaled to the number of comparisons the real run
      makes, and the threshold is raised until the expected count falls under
      budget.
    """
    groups: dict[str, np.ndarray] = {}
    for cam in sorted(set(d_cams.tolist())):
        groups[cam] = d_embs[d_cams == cam]
    groups = {k: v for k, v in groups.items() if len(v) >= 2}
    if len(groups) < 2:
        return None, {}

    null_pairs = _cross_pair_count(len(v) for v in groups.values())
    if null_pairs < 200:
        return None, {}

    # Candidates from the impostor tail upward. Above the observed maximum the
    # null run produces nothing by definition, so a threshold always exists —
    # the question the output must answer honestly is whether anything real
    # still clears it.
    hi = stats["max"]
    candidates = sorted({stats["p99"], stats["p999"],
                         hi - 0.02, hi - 0.01, hi,
                         hi + 0.005, hi + 0.01, hi + 0.02, hi + 0.04})
    trace = []
    chosen = None
    for thr in candidates:
        if thr <= 0:
            continue
        n_false = len(mutual_best_pairs(groups, thr))
        rate = n_false / null_pairs
        expected = rate * real_pairs
        trace.append((thr, n_false, expected))
        if chosen is None and expected <= budget:
            chosen = thr
    info = {
        "null_groups": len(groups),
        "null_crops": int(sum(len(v) for v in groups.values())),
        "null_pairs": null_pairs,
        "real_pairs": real_pairs,
        "budget": budget,
        "trace": trace,
        "threshold": chosen,
    }
    return chosen, info


# ── stage 3: link ───────────────────────────────────────────────────────────
class _Union:
    def __init__(self, keys):
        self.p = {k: k for k in keys}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def link_tracks(by_cam: dict[str, list[Track]], threshold: float):
    """Link tracks across cameras, then take the transitive closure.

    Uses exactly the same matcher the null calibration was run through, so the
    measured false-link rate applies to this call and not to an approximation
    of it.
    """
    groups = {cam: np.vstack([t.emb for t in ts])
              for cam, ts in by_cam.items() if ts}
    links = [(by_cam[cx][i], by_cam[cy][j], s)
             for cx, i, cy, j, s in mutual_best_pairs(groups, threshold)]

    everything = [t for ts in by_cam.values() for t in ts]
    uf = _Union([t.key for t in everything])
    for a, b, _ in links:
        uf.union(a.key, b.key)

    groups: dict[str, list[Track]] = {}
    for t in everything:
        groups.setdefault(uf.find(t.key), []).append(t)

    clusters = []
    for members in groups.values():
        if len({m.camera for m in members}) < 2:
            continue                           # seen by one camera: no journey
        members.sort(key=lambda m: (m.camera, m.local_id))
        # A stable id, so re-running does not renumber the same vehicle.
        h = hashlib.sha1("|".join(m.key for m in members).encode()).hexdigest()
        member_keys = {m.key for m in members}
        scores = [s for a, b, s in links
                  if a.key in member_keys and b.key in member_keys]
        clusters.append({
            "global_id": f"APP_{h[:10].upper()}",
            "members": members,
            "score": float(np.mean(scores)) if scores else float("nan"),
            "min_score": float(np.min(scores)) if scores else float("nan"),
        })
    clusters.sort(key=lambda c: -len(c["members"]))
    return clusters, links


# ── stage 4: score it against the plates ────────────────────────────────────
def score_against_plates(by_cam: dict[str, list[Track]], links, threshold):
    """Plates are ground truth; appearance is what is being tested."""
    plated = {t.key: t.plate for ts in by_cam.values() for t in ts if t.plate}

    tp = fp = 0
    wrong = []
    for a, b, s in links:
        pa, pb = plated.get(a.key), plated.get(b.key)
        if not pa or not pb:
            continue                           # no ground truth for this link
        if pa == pb:
            tp += 1
        else:
            fp += 1
            wrong.append((a.key, pa, b.key, pb, s))

    # Every cross-camera pair that the plates say IS the same vehicle.
    truth_pairs = []
    cams = list(by_cam)
    for cx, cy in itertools.combinations(cams, 2):
        for ta in by_cam[cx]:
            for tb in by_cam[cy]:
                if ta.plate and tb.plate and ta.plate == tb.plate:
                    truth_pairs.append((ta, tb))

    linked_pairs = {(a.key, b.key) for a, b, _ in links}
    linked_pairs |= {(b, a) for a, b in linked_pairs}
    found = sum(1 for ta, tb in truth_pairs
                if (ta.key, tb.key) in linked_pairs)

    # What the missed ones scored, so a recall miss can be told apart from a
    # threshold that is simply set high.
    missed_scores = []
    for ta, tb in truth_pairs:
        if (ta.key, tb.key) not in linked_pairs:
            missed_scores.append(float(ta.emb @ tb.emb))

    return {
        "plated_tracks": len(plated),
        "scored_links": tp + fp,
        "correct": tp,
        "incorrect": fp,
        "wrong_examples": wrong[:5],
        "truth_pairs": len(truth_pairs),
        "truth_pairs_found": found,
        "missed_scores": missed_scores,
        "threshold": threshold,
    }


def plate_pair_audit(by_cam: dict[str, list[Track]], threshold: float):
    """Score the appearance channel on every plated pair, not just the links.

    WHY THIS IS A SEPARATE MEASUREMENT
      The cross-camera scoring above can only judge a link the linker chose to
      make, and on the handheld clips it judges nothing at all, because no plate
      is readable in them. That is a real limitation of those clips, not of the
      method — so the method is measured here instead, on this fleet's own CCTV,
      where plates do read.

      Every pair of tracks carrying a confirmed plate is a labelled example:
      same plate means same vehicle, different plate means different vehicle.
      That gives true/false positives and negatives at the derived threshold —
      the ordinary way a matcher is scored, on ground truth the appearance model
      never saw.

      The positives are not easy ones. A vehicle appears as two separate tracks
      mainly when the tracker switched identity on it — a different stretch of
      road, a different pose, often a partial occlusion in between. Re-matching
      those is exactly the job.
    """
    tracks = [t for ts in by_cam.values() for t in ts if t.plate]
    if len(tracks) < 4:
        return None

    embs = np.vstack([t.emb for t in tracks])
    sims = embs @ embs.T
    iu = np.triu_indices(len(tracks), k=1)
    s = sims[iu]
    same = np.array([tracks[i].plate == tracks[j].plate
                     for i, j in zip(*iu)])
    cross_cam = np.array([tracks[i].camera != tracks[j].camera
                          for i, j in zip(*iu)])

    if same.sum() == 0:
        return {"positives": 0, "negatives": int((~same).sum())}

    pred = s >= threshold
    tp = int((pred & same).sum())
    fn = int((~pred & same).sum())
    fp = int((pred & ~same).sum())
    tn = int((~pred & ~same).sum())

    # The threshold that would admit no false positive at all, and what recall
    # survives there — the operating point a police deployment would want.
    neg_max = float(s[~same].max()) if (~same).any() else 0.0
    recall_at_zero_fp = float((s[same] > neg_max).mean())

    return {
        "tracks": len(tracks),
        "positives": int(same.sum()),
        "negatives": int((~same).sum()),
        "cross_camera_positives": int((same & cross_cam).sum()),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "pos_mean": float(s[same].mean()),
        "pos_min": float(s[same].min()),
        "neg_mean": float(s[~same].mean()),
        "neg_max": neg_max,
        "zero_fp_threshold": neg_max,
        "recall_at_zero_fp": recall_at_zero_fp,
        "threshold": threshold,
    }


# ── stage 5: persist ────────────────────────────────────────────────────────
def commit(clusters, cameras: list[str], base_time: datetime, purge: bool) -> int:
    """Write the clusters as JourneyEvents, so the existing Journeys page
    renders them with no frontend change.

    match_type is 'appearance', which is what distinguishes these from the
    plate path's 'plate_confirmed'. subtype and visual_score are set, so these
    rows pass the seeded-row provenance test the read side applies (see
    routers/v1/journeys.py::_real_events_only) — they are observations, and
    they must not be mistaken for generated rows.
    """
    from backend.db.models import Camera, JourneyEvent
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        if purge:
            n = (db.query(JourneyEvent)
                 .filter(JourneyEvent.match_type == "appearance")
                 .filter(JourneyEvent.camera_id.in_(cameras))
                 .delete(synchronize_session=False))
            log(f"  purged {n} previous appearance event(s)")

        cams = {c.id: c for c in db.query(Camera)
                .filter(Camera.id.in_(cameras)).all()}
        written = 0
        for cl in clusters:
            for m in cl["members"]:
                cam = cams.get(m.camera)
                db.add(JourneyEvent(
                    reid_id=cl["global_id"],
                    camera_id=m.camera,
                    lat=getattr(cam, "lat", None),
                    lon=getattr(cam, "lon", None),
                    zone=getattr(cam, "zone", None),
                    district=getattr(cam, "district", None),
                    object_class="vehicle",
                    subtype=m.cls_name,
                    plate_text=m.plate,
                    visual_score=cl["score"] if cl["score"] == cl["score"] else 0.0,
                    final_score=cl["score"] if cl["score"] == cl["score"] else 0.0,
                    match_type="appearance",
                    timestamp=base_time + timedelta(
                        seconds=m.seconds(m.first_frame)
                        + 60.0 * cameras.index(m.camera)),
                ))
                written += 1
        db.commit()
        return written
    finally:
        db.close()


# ── proof sheet ─────────────────────────────────────────────────────────────
def contact_sheet(clusters, path: Path, limit: int = 6) -> None:
    rows = []
    for cl in clusters[:limit]:
        tiles = []
        for m in cl["members"][:6]:
            t = cv2.resize(m.best_crop, (140, 240))
            cv2.rectangle(t, (0, 0), (139, 239), (90, 210, 120), 3)
            cv2.putText(t, m.camera, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 210, 120), 2, cv2.LINE_AA)
            label = m.plate or "no plate"
            cv2.putText(t, label, (6, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (235, 235, 235), 1, cv2.LINE_AA)
            tiles.append(t)
        strip = np.hstack(tiles)
        bar = np.full((30, strip.shape[1], 3), 28, np.uint8)
        cv2.putText(bar, f"{cl['global_id']}   mean sim {cl['score']:.3f}",
                    (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 255), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack([bar, strip]))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 12), (0, w - r.shape[1]), (0, 0)), constant_values=28)
            for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"  proof sheet -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cameras", default="CAM_M1,CAM_M2,CAM_M3,CAM_M4")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=3,
                    help="process every Nth frame")
    ap.add_argument("--topk", type=int, default=5,
                    help="sharpest crops averaged into one track vector")
    ap.add_argument("--distractors", type=int, default=400)
    ap.add_argument("--false-links", type=float, default=0.5,
                    help="how many false links the WHOLE run may be expected "
                         "to make; the threshold is raised until the null run "
                         "predicts fewer than this. Not a per-comparison rate")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-plates", action="store_true",
                    help="skip ANPR (faster, but then nothing scores the links)")
    ap.add_argument("--commit", action="store_true",
                    help="write the clusters to journey_events")
    ap.add_argument("--no-purge", action="store_true",
                    help="keep previous appearance events for these cameras")
    args = ap.parse_args()

    cameras = [c.strip().upper() for c in args.cameras.split(",") if c.strip()]
    if len(cameras) < 2:
        log("FATAL: cross-camera linking needs at least two cameras.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from backend.services.anpr_engine import get_anpr_engine
    from backend.services.reid_embedder import get_embedder

    log("loading models ...")
    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()

    def _val(obj, name):                       # is_stub is a property, not a method
        v = getattr(obj, name, None)
        return v() if callable(v) else v

    if _val(embedder, "is_stub"):
        log("FATAL: the Re-ID model did not load. Every embedding would be "
            "noise and every link meaningless. Refusing to run.")
        return 2
    log(f"  Re-ID checkpoint: {_val(embedder, 'checkpoint_applied')}")

    anpr = None if args.no_plates else get_anpr_engine()

    # ── scan ────────────────────────────────────────────────────────────────
    log("\nscanning clips (every vehicle, not a chosen one) ...")
    by_cam: dict[str, list[Track]] = {}
    for i, cam in enumerate(cameras):
        clip = _clip_for(cam)
        if clip is None:
            log(f"  {cam}: no clip found under data/reid_test or data/clips — skipped")
            continue
        by_cam[cam] = scan_clip(model, anpr, embedder, cam, clip,
                                imgsz=args.imgsz, stride=args.stride,
                                topk=args.topk, read_plates=not args.no_plates,
                                tid_base=1_000_000 * (i + 1))
    by_cam = {k: v for k, v in by_cam.items() if v}
    if len(by_cam) < 2:
        log("FATAL: fewer than two cameras produced tracks.")
        return 2
    total = sum(len(v) for v in by_cam.values())
    log(f"  {total} tracks across {len(by_cam)} cameras")

    # ── threshold ───────────────────────────────────────────────────────────
    log("\nsampling impostors (vehicles known to be different) ...")
    d_embs, d_cams, stats = sample_impostors(
        model, embedder, set(by_cam), n_target=args.distractors,
        imgsz=args.imgsz, seed=args.seed)
    if d_embs is None:
        log("FATAL: not enough impostor pairs to calibrate. Refusing to "
            "invent a threshold.")
        return 2
    log(f"  {stats['crops']} crops from {stats['cameras']} other fleet cameras"
        f" -> {stats['pairs']:,} cross-camera impostor pairs")
    log(f"  impostor similarity: mean {stats['mean']:.3f}  median "
        f"{stats['p50']:.3f}  p99 {stats['p99']:.3f}  "
        f"p99.9 {stats['p999']:.3f}  MAX {stats['max']:.3f}")

    real_pairs = _cross_pair_count(len(v) for v in by_cam.values())

    # The decisive calibration: vehicles on screen together in this run's own
    # footage. The fleet sample above is kept only as a reference point,
    # because it is measurably too optimistic here.
    imp, idom = in_domain_impostors(by_cam)
    if imp is None or idom["pairs"] < 20:
        log("\nFATAL: fewer than 20 pairs of vehicles appear on screen "
            "together, so there is no in-domain way to tell how often "
            "different vehicles look alike here. Refusing to pick a threshold "
            "from cross-camera crops alone — that was measured to be far too "
            "permissive.")
        return 2

    log(f"\nin-domain impostors (two vehicles in the SAME FRAME = certainly "
        f"different):")
    log(f"  {idom['pairs']:,} such pairs   "
        + "  ".join(f"{c}:{n}" for c, n in idom["per_camera"].items()))
    log(f"  similarity: mean {idom['mean']:.3f}  median {idom['p50']:.3f}  "
        f"p95 {idom['p95']:.3f}  p99 {idom['p99']:.3f}  MAX {idom['max']:.3f}")
    log(f"  compare the fleet-sampled max of {stats['max']:.3f} — different "
        f"cameras make impostors look easier than they are here.")

    log(f"\ncalibrating ({real_pairs:,} comparisons this run, budget "
        f"{args.false_links} false link(s)) ...")
    thr, trace = threshold_from_in_domain(imp, real_pairs, args.false_links)
    log("     threshold   in-domain false rate   expected false links")
    for t, r, e in trace:
        log(f"      {t:8.4f}   {r*100:19.3f}%   {e:20.2f}")
    if thr is None:
        log("\nRESULT: no threshold keeps the expected false links under "
            f"{args.false_links} on this footage.")
        log("Appearance alone cannot assign identity here at an acceptable "
            "error rate, and the honest output is no links rather than links "
            "that cannot be told from coincidence.")
        return 1
    log(f"\n  THRESHOLD = {thr:.4f}  — set so that, at the rate different "
        f"vehicles in this footage")
    log(f"  actually resemble each other, {real_pairs:,} comparisons yield "
        f"under {args.false_links} false link(s).")

    # ── link ────────────────────────────────────────────────────────────────
    clusters, links = link_tracks(by_cam, thr)
    log(f"\n{len(links)} cross-camera link(s) above threshold -> "
        f"{len(clusters)} multi-camera vehicle(s)")
    for cl in clusters[:12]:
        route = " -> ".join(m.camera for m in cl["members"])
        plates = {m.plate for m in cl["members"] if m.plate}
        tag = ("plate " + "/".join(sorted(plates))) if plates else "NO PLATE"
        log(f"  {cl['global_id']}  {len(cl['members'])} sightings  "
            f"mean {cl['score']:.3f}  {route}   [{tag}]")

    no_plate = [c for c in clusters if not any(m.plate for m in c["members"])]
    log(f"\n  {len(no_plate)} of {len(clusters)} were linked with NO readable "
        f"plate on any camera — journeys the plate channel cannot produce at all.")

    # ── score ───────────────────────────────────────────────────────────────
    if not args.no_plates:
        sc = score_against_plates(by_cam, links, thr)
        log("\n" + "=" * 66)
        log("ACCURACY, scored against plates (independent channel)")
        log("=" * 66)
        log(f"  tracks carrying a confirmed plate : {sc['plated_tracks']}")
        if sc["scored_links"] == 0:
            log("  precision : NOT MEASURABLE — no link joined two tracks that")
            log("              both had a readable plate, so nothing here can")
            log("              confirm or refute a link. This is the honest")
            log("              outcome on clips where plates are unreadable.")
        else:
            p = sc["correct"] / sc["scored_links"]
            log(f"  links with plates on both ends    : {sc['scored_links']}")
            log(f"  precision                         : {sc['correct']}/"
                f"{sc['scored_links']} = {p*100:.1f}%")
            for a, pa, b, pb, s in sc["wrong_examples"]:
                log(f"      WRONG {a} ({pa}) -> {b} ({pb}) at {s:.3f}")
        if sc["truth_pairs"] == 0:
            log("  recall    : NOT MEASURABLE — no plate was read on two")
            log("              different cameras, so there is no known pair to")
            log("              have found.")
        else:
            r = sc["truth_pairs_found"] / sc["truth_pairs"]
            log(f"  same-plate cross-camera pairs     : {sc['truth_pairs']}")
            log(f"  recall                            : {sc['truth_pairs_found']}/"
                f"{sc['truth_pairs']} = {r*100:.1f}%")
            if sc["missed_scores"]:
                log(f"      missed pairs scored: "
                    + ", ".join(f"{m:.3f}" for m in sorted(sc['missed_scores'])[-6:])
                    + f"  (threshold {thr:.3f})")
        if sc["scored_links"] and sc["scored_links"] < 10:
            log("\n  NOTE: too few plated links for that percentage to be a")
            log("  rate. Read it as a count, not as accuracy.")
        log("=" * 66)

        au = plate_pair_audit(by_cam, thr)
        if au and au.get("positives"):
            log("\n" + "=" * 66)
            log("APPEARANCE MATCHING, scored on every plated pair")
            log("=" * 66)
            log(f"  plated tracks                     : {au['tracks']}")
            log(f"  labelled pairs                    : {au['positives']} same "
                f"vehicle, {au['negatives']:,} different")
            log(f"      of the same-vehicle pairs, {au['cross_camera_positives']} "
                f"span two cameras")
            log(f"  similarity, same vehicle          : mean {au['pos_mean']:.3f}"
                f"  worst {au['pos_min']:.3f}")
            log(f"  similarity, different vehicles    : mean {au['neg_mean']:.3f}"
                f"  worst {au['neg_max']:.3f}")
            log(f"  at the derived threshold {au['threshold']:.4f}:")
            log(f"      precision                     : {au['precision']*100:.1f}%"
                f"   ({au['tp']} correct, {au['fp']} wrong)")
            log(f"      recall                        : {au['recall']*100:.1f}%"
                f"   ({au['tp']} of {au['tp']+au['fn']})")
            log(f"  at zero false positives (thr {au['zero_fp_threshold']:.4f}):")
            log(f"      recall                        : "
                f"{au['recall_at_zero_fp']*100:.1f}%")
            if au["positives"] < 10:
                log("\n  NOTE: fewer than 10 same-vehicle pairs. These are counts,")
                log("  not rates — do not quote them as accuracy.")
            log("=" * 66)
        elif au is not None:
            log("\n  plated-pair audit: no two tracks shared a plate, so there is")
            log("  no positive example to score. Run over more cameras or a")
            log("  camera whose plates read.")

    contact_sheet(clusters, OUT_DIR / "appearance_clusters.jpg")

    with open(OUT_DIR / "appearance_linker_report.txt", "w", encoding="utf-8") as f:
        f.write("Cross-camera identity from appearance alone\n")
        f.write(f"cameras      : {', '.join(by_cam)}\n")
        f.write(f"tracks       : {total}\n")
        f.write(f"threshold    : {thr:.4f}, set from {idom['pairs']:,} pairs of "
                f"vehicles seen in the same frame (certainly different) so "
                f"that this run's {real_pairs:,} comparisons yield under "
                f"{args.false_links} false link(s)\n")
        f.write(f"in-domain impostor max : {idom['max']:.4f}   "
                f"(fleet-sampled max {stats['max']:.4f} — too optimistic)\n")
        f.write(f"links        : {len(links)}\n")
        f.write(f"vehicles     : {len(clusters)} seen by 2+ cameras "
                f"({len(no_plate)} with no readable plate)\n")
        for cl in clusters:
            f.write(f"  {cl['global_id']}  mean {cl['score']:.3f}  "
                    + " -> ".join(m.camera for m in cl["members"]) + "\n")
    log(f"  report -> {OUT_DIR / 'appearance_linker_report.txt'}")

    if args.commit:
        log("\nwriting journey_events ...")
        base = datetime.utcnow().replace(microsecond=0) - timedelta(minutes=30)
        n = commit(clusters, list(by_cam), base, purge=not args.no_purge)
        log(f"  {n} event(s) written with match_type='appearance'.")
        log("  Open the Journeys page and search one of the APP_ ids above.")
    else:
        log("\n(dry run — pass --commit to write these to journey_events)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
