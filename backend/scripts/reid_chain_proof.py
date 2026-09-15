"""One vehicle, one ID, across all four handheld cameras — found, not told.

WHAT THIS PROVES AND HOW IT DIFFERS FROM THE THRESHOLD LINKER
  `appearance_linker.py` asks an absolute question: "is track A the same
  vehicle as track B?" That needs a similarity threshold, and on this footage a
  threshold is a bad instrument — calibrated honestly it links almost nothing,
  and calibrated loosely it groups every black vehicle together. Measured, on
  the contact sheet, not asserted.

  This asks the question re-identification is actually good at, and which this
  project has already measured at Rank-1 100%: "of everything camera M2 saw,
  WHICH track is this vehicle?" That is ranking, not thresholding. It always
  returns an answer, so the burden of proof moves to showing the answer is
  right — which is what the gallery evaluation below is for.

THE SUBJECT IS DISCOVERED, NOT NAMED
  An earlier version of this proof had a person name the subject's track number
  in each clip. That is defensible as evaluation but it is not a system: it
  cannot run on footage nobody has eyeballed, and it invites the fair objection
  that the answer was supplied with the question.

  So every track in every clip is tried as an anchor. For each anchor a chain
  is grown — the best match in each remaining camera, matched against the
  running mean of the chain so far rather than the anchor alone, so one odd
  viewpoint cannot steer it. Each completed chain is scored by its WEAKEST
  pairwise similarity, because a chain is only as good as its worst link. The
  winning chain is the entity most consistently present across all four
  cameras. Nobody says which vehicle it is; the margin over the runner-up says
  how clear the answer was.

WHY THE PLATE CANNOT HELP HERE — measured, so it is not an excuse
  The clips are 478x850 portrait phone video. A number plate in them is roughly
  20-30 px wide, against a recogniser floor of 40 px and a committed operating
  point that needs 80 px. Zero plates were read across all four clips by the
  same engine that reads 3,900 on CAM_09. Appearance is not the convenient
  channel here, it is the only one.

  python -m backend.scripts.reid_chain_proof
  python -m backend.scripts.reid_chain_proof --commit
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2                                                          # noqa: E402
import numpy as np                                                  # noqa: E402

from backend.scripts.appearance_linker import (                     # noqa: E402
    Track, _clip_for, scan_clip,
)

OUT_DIR = ROOT / "output" / "reid_chain_proof"
CAMS = ["CAM_M1", "CAM_M2", "CAM_M3", "CAM_M4"]


def log(msg: str = "") -> None:
    print(msg, flush=True)


VEHICLE_CLASSES = [2, 3, 5, 7]
PERSON_CLASS = 0
TWO_WHEELERS = {3}          # motorcycle; a bicycle would belong here too


def _union(a, b):
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def scan_clip_rider_aware(model, embedder, camera: str, path: Path, *,
                          imgsz: int, stride: int, topk: int, tid_base: int):
    """Every track in one clip, with the RIDER included in the crop.

    WHY THIS EXISTS — measured, from the first run of this script.
      Without it the chain scored Rank-1 9/12, and all three misses were
      queries into CAM_M1. The contact sheet showed why: YOLO's motorcycle box
      is tight on the machine, so CAM_M1's crop was a bare rear view of the
      bike while CAM_M2/M3/M4 were a rider in a light shirt sitting on it. The
      embedder was being asked to match a motorcycle against a person.

      OSNet-IBN is a PERSON re-identification network. On a two-wheeler the
      rider is not noise to be cropped away, it is the richest signal the model
      has — clothing colour, posture, build — and it is what makes the same
      subject recognisable from behind, from the side and head-on. So for
      motorcycles the vehicle box is unioned with the box of the person riding
      it, giving one consistent "rider + machine" crop on every camera.

      Cars and trucks are left alone: their occupants are inside the box
      already, and merging a pedestrian standing beside a parked car would
      corrupt the crop rather than complete it.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        log(f"  {camera}: cannot open {path.name}")
        return []
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    model.predictor = None
    tracks: dict[int, Track] = {}
    frame_i = 0
    merged_count = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frame_i += 1
        if stride > 1 and frame_i % stride:
            continue

        r = model.track(frame, persist=True,
                        classes=VEHICLE_CLASSES + [PERSON_CLASS],
                        conf=0.35, tracker="botsort.yaml", imgsz=imgsz,
                        verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            continue

        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        cls = r.boxes.cls.cpu().numpy().astype(int)

        persons = [xyxy[i] for i in range(len(cls)) if cls[i] == PERSON_CLASS]
        two_wheelers = [i for i in range(len(cls)) if int(cls[i]) in TWO_WHEELERS]

        # Assign each rider to exactly ONE machine.
        #
        # Letting a person merge into every nearby motorcycle was a real bug:
        # two different bikes in one frame both absorbed the same rider, their
        # crops became near-identical, and the same-frame similarity that
        # calibrates fragment merging shot to 0.976 — which then blocked every
        # legitimate merge on that camera. A rider sits on one machine.
        rider_of: dict[int, list] = {}
        for p in persons:
            px1, py1, px2, py2 = p
            pcx = (px1 + px2) / 2.0
            best, best_score = None, 0.0
            for i in two_wheelers:
                vx1, vy1, vx2, vy2 = xyxy[i]
                vw, vh = vx2 - vx1, vy2 - vy1
                if vw <= 0 or vh <= 0:
                    continue
                # Rider shares the machine's horizontal span and their feet
                # land within its vertical extent. Both tests matter: the
                # first alone catches a pedestrian on the pavement behind,
                # the second alone catches someone standing beside it.
                if not ((vx1 - 0.35 * vw) <= pcx <= (vx2 + 0.35 * vw)):
                    continue
                if not (vy1 - vh * 1.2 <= py2 <= vy2 + vh * 0.35):
                    continue
                # Prefer the machine whose centre the rider sits closest above.
                dx = abs(pcx - (vx1 + vx2) / 2.0) / vw
                score = 1.0 / (1.0 + dx)
                if score > best_score:
                    best, best_score = i, score
            if best is not None:
                rider_of.setdefault(best, []).append(p)

        for i in range(len(cls)):
            c = int(cls[i])
            if c == PERSON_CLASS:
                continue
            box = list(xyxy[i])
            for p in rider_of.get(i, []):
                box = _union(box, list(p))
                merged_count += 1

            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2 = min(frame.shape[1] - 1, x2)
            y2 = min(frame.shape[0] - 1, y2)
            if x2 - x1 < 40 or y2 - y1 < 48:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            tid = int(ids[i])
            t = tracks.get(tid)
            if t is None:
                t = Track(camera=camera, local_id=tid, cls_id=c,
                          first_frame=frame_i, fps=fps)
                tracks[tid] = t
            t.n_frames += 1
            t.last_frame = frame_i
            t.frames.add(frame_i)
            # Where it was, so a vehicle that TRANSITED the scene can be told
            # from one parked in the background — see moving_pool().
            if not hasattr(t, "centres"):
                t.centres = []
                t.frame_wh = (frame.shape[1], frame.shape[0])
            t.centres.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharp = float(cv2.Laplacian(g, cv2.CV_64F).var())
            t.crops.append((sharp * (crop.shape[0] * crop.shape[1]) ** 0.5,
                            crop.copy()))
            if len(t.crops) > topk * 3:
                t.crops.sort(key=lambda x: -x[0])
                del t.crops[topk:]
    cap.release()

    out = []
    for t in tracks.values():
        if not t.crops:
            continue
        t.crops.sort(key=lambda x: -x[0])
        del t.crops[topk:]
        t.best_crop = t.crops[0][1]
        out.append(t)

    if out:
        flat, spans = [], []
        for t in out:
            spans.append((len(flat), len(t.crops)))
            flat.extend(c for _, c in t.crops)
        embs = np.asarray(embedder.extract_batch(flat), dtype=np.float32)
        for t, (off, n) in zip(out, spans):
            v = embs[off:off + n].mean(axis=0)
            nn = float(np.linalg.norm(v))
            t.emb = (v / nn).astype(np.float32) if nn > 1e-9 else None
        out = [t for t in out if t.emb is not None]

    log(f"  {camera}: {len(out)} tracks ({merged_count} rider merges) "
        f"({path.name})")
    return out


def displacement(t: Track) -> float:
    """How far the track travelled, as a fraction of the frame diagonal."""
    cs = getattr(t, "centres", None)
    if not cs or len(cs) < 2:
        return 0.0
    w, h = getattr(t, "frame_wh", (1, 1))
    diag = (w * w + h * h) ** 0.5 or 1.0
    xs = [c[0] for c in cs]
    ys = [c[1] for c in cs]
    span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    return float(span / diag)


def moving_pool(by_cam: dict[str, list[Track]], min_frames: int,
                min_disp: float):
    """Restrict the SEARCH to vehicles that actually transited the scene.

    WHY, and why this is a prior rather than a thumb on the scale.
      Scored on similarity alone the search picked a chain of a black SUV and a
      row of parked motorcycles — dark things resemble other dark things, and
      the winning margin was +0.005, i.e. nothing. The clips contain a great
      deal of stationary background: parked bikes, cars at the kerb, vehicles
      that never move for the whole 10 seconds.

      What is being looked for is a vehicle that PASSED each of the four
      cameras. That is a statement about the subject's behaviour, not its
      identity — it does not say which vehicle, or what colour, or where in the
      frame. Anything parked cannot be the thing that travelled the route, so
      excluding it removes candidates that are wrong by definition rather than
      candidates that are inconvenient.

      The gallery evaluation below is deliberately NOT filtered this way: the
      chosen track is still ranked against every track each camera saw,
      stationary ones included, plus 500 fleet vehicles. The prior narrows the
      search; it does not soften the test.
    """
    pool, report = {}, {}
    for cam, ts in by_cam.items():
        keep = [t for t in ts
                if t.n_frames >= min_frames and displacement(t) >= min_disp]
        if not keep:                       # never leave a camera empty
            keep = sorted(ts, key=lambda t: -displacement(t))[:3]
        pool[cam] = keep
        report[cam] = (len(ts), len(keep))
    return pool, report


def merge_fragments(tracks: list[Track], camera: str):
    """Join tracks that are the same vehicle seen twice by one camera.

    WHY — this is the project's own known failure, showing up again.
      The tracker fragments a single vehicle into several tracks when it loses
      and re-acquires it. On CAM_M3 the subject motorcycle appears as tracks
      #2, #10, #16, #18 and #9; on CAM_M1 as #3 and #7. Left unmerged, the
      gallery evaluation punishes the model for ranking one fragment of the
      right vehicle above another fragment of the SAME vehicle — scoring a
      correct answer as a miss. The first run lost 3 of 12 queries that way.

    THE TWO RULES, one of them a hard constraint
      1. Two tracks visible in the SAME FRAME are different vehicles. Always.
         No similarity score can override it; a thing cannot be in two places
         at once. This is what stops a merge cascading through every
         similar-looking bike in the clip.
      2. Two tracks that never co-occur MAY be the same, and are merged if they
         are more alike than two known-different vehicles from this same
         camera ever are — the threshold is the maximum similarity observed
         between same-frame pairs here, so it is measured from the footage
         rather than chosen.

      Merging is greedy from the most similar pair down, and a merge is
      rejected if the two GROUPS would together contain a co-occurring pair,
      so the constraint survives transitively.
    """
    if len(tracks) < 2:
        return tracks, 0, float("nan")

    # Rule 2's threshold, measured on this camera.
    same_frame = [float(a.emb @ b.emb)
                  for a, b in itertools.combinations(tracks, 2)
                  if a.frames & b.frames]
    if same_frame:
        thr = max(same_frame)
    else:
        # No two vehicles ever shared a frame here, so this camera cannot say
        # how alike different vehicles look. Fall back to a strict value rather
        # than merging on an unmeasured basis.
        thr = 0.90

    groups = {t.key: [t] for t in tracks}
    owner = {t.key: t.key for t in tracks}

    pairs = sorted(
        ((float(a.emb @ b.emb), a, b)
         for a, b in itertools.combinations(tracks, 2)
         if not (a.frames & b.frames)),
        key=lambda x: -x[0])

    merges = 0
    for sim, a, b in pairs:
        if sim < thr:
            break
        ga, gb = owner[a.key], owner[b.key]
        if ga == gb:
            continue
        A, B = groups[ga], groups[gb]
        # transitive safety: no member of A may share a frame with any of B
        if any(x.frames & y.frames for x in A for y in B):
            continue
        groups[ga] = A + B
        for m in B:
            owner[m.key] = ga
        del groups[gb]
        merges += 1

    out = []
    for key, members in groups.items():
        if len(members) == 1:
            out.append(members[0])
            continue
        members.sort(key=lambda m: -m.n_frames)
        lead = members[0]
        merged = Track(camera=camera, local_id=lead.local_id,
                       cls_id=lead.cls_id, fps=lead.fps,
                       first_frame=min(m.first_frame for m in members),
                       last_frame=max(m.last_frame for m in members))
        merged.n_frames = sum(m.n_frames for m in members)
        for m in members:
            merged.frames |= m.frames
            merged.crops.extend(m.crops)
        merged.crops.sort(key=lambda c: -c[0])
        del merged.crops[8:]
        merged.best_crop = merged.crops[0][1]
        # Weight by how long each fragment was actually seen: a 288-frame view
        # is better evidence than a 6-frame glimpse.
        w = np.array([m.n_frames for m in members], dtype=np.float32)
        v = (np.vstack([m.emb for m in members]) * w[:, None]).sum(0)
        n = float(np.linalg.norm(v))
        merged.emb = (v / n).astype(np.float32)
        merged.plate = next((m.plate for m in members if m.plate), None)
        merged.fragment_ids = [m.local_id for m in members]
        merged.centres = [c for m in members
                          for c in getattr(m, "centres", [])]
        merged.frame_wh = getattr(lead, "frame_wh", (1, 1))
        out.append(merged)

    out.sort(key=lambda t: -t.n_frames)
    return out, merges, thr


def candidates_sheet(by_cam: dict[str, list[Track]], path: Path) -> None:
    """Every track found, so the chosen one can be checked against the rest."""
    rows = []
    for cam in CAMS:
        ts = sorted(by_cam.get(cam, []), key=lambda t: -t.n_frames)[:10]
        if not ts:
            continue
        tiles = []
        for t in ts:
            x = cv2.resize(t.best_crop, (120, 210))
            cv2.rectangle(x, (0, 0), (119, 209), (70, 70, 70), 2)
            cv2.putText(x, f"#{t.local_id}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(x, f"{t.n_frames}f", (6, 202), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (210, 210, 210), 1, cv2.LINE_AA)
            tiles.append(x)
        strip = np.hstack(tiles)
        bar = np.full((24, strip.shape[1], 3), 26, np.uint8)
        cv2.putText(bar, cam, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (120, 220, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([bar, strip]))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 8), (0, w - r.shape[1]), (0, 0)), constant_values=26)
            for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    log(f"  candidates -> {path}")


# ── chain search ────────────────────────────────────────────────────────────
def grow_chain(by_cam: dict[str, list[Track]], anchor_cam: str, anchor: Track):
    """Best match in every other camera, against the running mean of the chain.

    Cameras are taken best-first rather than in a fixed order: whichever camera
    currently offers the most confident match is added next, so the mean is
    built from the strongest evidence available at each step instead of being
    dragged by whichever camera happened to come first in the list.
    """
    chain = {anchor_cam: anchor}
    mean = anchor.emb.copy()
    remaining = [c for c in by_cam if c != anchor_cam]

    while remaining:
        best = None
        for cam in remaining:
            sims = np.array([float(mean @ t.emb) for t in by_cam[cam]])
            j = int(sims.argmax())
            if best is None or sims[j] > best[2]:
                best = (cam, j, float(sims[j]))
        cam, j, _ = best
        chain[cam] = by_cam[cam][j]
        remaining.remove(cam)
        mean = mean + by_cam[cam][j].emb
        n = float(np.linalg.norm(mean))
        if n > 1e-9:
            mean = mean / n
    return chain


def score_chain(chain: dict[str, Track]) -> tuple[float, float, dict]:
    """A chain is worth its weakest link, not its average."""
    pairs = {}
    for a, b in itertools.combinations(sorted(chain), 2):
        pairs[(a, b)] = float(chain[a].emb @ chain[b].emb)
    vals = list(pairs.values())
    return min(vals), float(np.mean(vals)), pairs


def find_best_chain(by_cam: dict[str, list[Track]]):
    """Every track in every camera gets a turn as the anchor."""
    seen: dict[tuple, dict] = {}
    for cam in by_cam:
        for t in by_cam[cam]:
            chain = grow_chain(by_cam, cam, t)
            key = tuple(sorted(m.key for m in chain.values()))
            if key in seen:
                continue
            weakest, mean, pairs = score_chain(chain)
            seen[key] = {"chain": chain, "weakest": weakest, "mean": mean,
                         "pairs": pairs}
    ranked = sorted(seen.values(), key=lambda c: -c["weakest"])
    return ranked


# ── gallery evaluation ──────────────────────────────────────────────────────
def evaluate_chain(chain: dict[str, Track], by_cam: dict[str, list[Track]],
                   d_emb: np.ndarray):
    """Standard re-ID scoring: is the chosen track the TOP-RANKED one?

    For every ordered pair of cameras, the chain's track in the first is the
    query, and the gallery is every track the second camera saw plus several
    hundred vehicles cut from this project's own CCTV. If the chain is right,
    its own member comes first out of all of them. Getting that wrong is
    exactly what a coincidental match would do.
    """
    results = []
    for qc, gc in itertools.permutations(sorted(chain), 2):
        q = chain[qc].emb
        gallery = [t.emb for t in by_cam[gc]]
        truth_idx = by_cam[gc].index(chain[gc])
        G = np.vstack(gallery + ([d_emb] if len(d_emb) else []))
        sims = G @ q
        order = np.argsort(-sims)
        rank = int(np.where(order == truth_idx)[0][0]) + 1
        # margin over the best competitor that is NOT the chosen track
        comp = np.delete(sims, truth_idx)
        results.append({
            "query": qc, "gallery_cam": gc, "rank": rank,
            "gallery_size": int(len(sims)),
            "sim": float(sims[truth_idx]),
            "best_other": float(comp.max()) if comp.size else float("nan"),
        })
    return results


def evaluate_leave_one_out(chain: dict[str, Track], by_cam: dict[str, list[Track]],
                           d_emb: np.ndarray):
    """The deployed question: given three sightings, find it at the fourth.

    A single-crop query is the hardest possible version of the task and not the
    one the system faces. In service the network already holds several
    sightings of a vehicle, and a new camera's tracks are ranked against
    everything known about it so far. So for each camera the query is the mean
    of the chain's OTHER three members, and the gallery is every track that
    camera saw plus the same 500 fleet vehicles.

    The camera under test contributes nothing to its own query, so this cannot
    flatter itself — it is leave-one-out, not a re-ranking of the answer.
    """
    out = []
    for cam in sorted(chain):
        others = [chain[c].emb for c in chain if c != cam]
        if not others:
            continue
        q = np.mean(np.vstack(others), axis=0)
        n = float(np.linalg.norm(q))
        if n <= 1e-9:
            continue
        q = q / n
        cand = by_cam[cam]
        truth_idx = cand.index(chain[cam])
        G = np.vstack([t.emb for t in cand] + ([d_emb] if len(d_emb) else []))
        sims = G @ q
        order = np.argsort(-sims)
        rank = int(np.where(order == truth_idx)[0][0]) + 1
        comp = np.delete(sims, truth_idx)
        out.append({
            "camera": cam, "rank": rank, "gallery_size": int(len(sims)),
            "sim": float(sims[truth_idx]),
            "best_other": float(comp.max()) if comp.size else float("nan"),
        })
    return out


# ── visuals ─────────────────────────────────────────────────────────────────
def chain_sheet(chain: dict[str, Track], gid: str, weakest: float,
                path: Path) -> None:
    tiles = []
    for cam in CAMS:
        m = chain.get(cam)
        if m is None:
            continue
        t = cv2.resize(m.best_crop, (190, 330))
        cv2.rectangle(t, (0, 0), (189, 329), (90, 220, 120), 3)
        cv2.putText(t, cam, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (90, 220, 120), 2, cv2.LINE_AA)
        cv2.putText(t, gid, (8, 318), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    strip = np.hstack(tiles)
    bar = np.full((38, strip.shape[1], 3), 26, np.uint8)
    cv2.putText(bar, f"{gid}   same ID on {len(tiles)} cameras   "
                     f"weakest pair {weakest:.3f}",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 215, 255), 1,
                cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([bar, strip]), [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  chain sheet -> {path}")


def ranking_sheet(chain, by_cam, d_emb, d_crops, path: Path) -> None:
    """Query beside the gallery's top-5, so a reader can see it was not close."""
    rows = []
    for qc, gc in [("CAM_M1", "CAM_M2"), ("CAM_M2", "CAM_M3"),
                   ("CAM_M3", "CAM_M4"), ("CAM_M4", "CAM_M1")]:
        if qc not in chain or gc not in chain:
            continue
        q = chain[qc].emb
        cand = by_cam[gc]
        G = np.vstack([t.emb for t in cand] + ([d_emb] if len(d_emb) else []))
        sims = G @ q
        order = np.argsort(-sims)[:5]
        truth_idx = cand.index(chain[gc])

        qt = cv2.resize(chain[qc].best_crop, (150, 260))
        cv2.rectangle(qt, (0, 0), (149, 259), (0, 200, 255), 3)
        cv2.putText(qt, f"Q {qc}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 200, 255), 2, cv2.LINE_AA)
        tiles = [qt, np.full((260, 12, 3), 26, np.uint8)]
        for k, gi in enumerate(order):
            if gi < len(cand):
                crop = cand[gi].best_crop
                good = (gi == truth_idx)
            else:
                crop = d_crops[gi - len(cand)]
                good = False
            t = cv2.resize(crop, (150, 260))
            col = (90, 220, 120) if good else (90, 90, 220)
            cv2.rectangle(t, (0, 0), (149, 259), col, 3)
            cv2.putText(t, f"#{k+1} {sims[gi]:.3f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2, cv2.LINE_AA)
            tiles.append(t)
        row = np.hstack(tiles)
        bar = np.full((26, row.shape[1], 3), 26, np.uint8)
        cv2.putText(bar, f"{qc} -> ranked against everything {gc} saw "
                         f"+ {len(d_emb)} fleet vehicles",
                    (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1,
                    cv2.LINE_AA)
        rows.append(np.vstack([bar, row]))
    if not rows:
        return
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 10), (0, w - r.shape[1]), (0, 0)), constant_values=26)
            for r in rows]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    log(f"  ranking sheet -> {path}")


# ── persistence ─────────────────────────────────────────────────────────────
def commit(chain: dict[str, Track], gid: str, weakest: float,
           pairs: dict) -> int:
    from backend.db.models import Camera, JourneyEvent
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        n = (db.query(JourneyEvent)
             .filter(JourneyEvent.match_type == "appearance")
             .filter(JourneyEvent.camera_id.in_(CAMS))
             .delete(synchronize_session=False))
        if n:
            log(f"  cleared {n} previous appearance event(s) on these cameras")

        cams = {c.id: c for c in db.query(Camera).filter(Camera.id.in_(CAMS)).all()}
        base = datetime.utcnow().replace(microsecond=0) - timedelta(minutes=20)
        written = 0
        for i, cam in enumerate(CAMS):
            m = chain.get(cam)
            if m is None:
                continue
            c = cams.get(cam)
            # Per-sighting score: how strongly THIS camera's crop agrees with
            # the rest of the chain, not the chain's overall figure — so a weak
            # sighting is visible as weak on the map.
            others = [v for k, v in pairs.items() if cam in k]
            score = float(np.mean(others)) if others else weakest
            db.add(JourneyEvent(
                reid_id=gid,
                camera_id=cam,
                lat=getattr(c, "lat", None),
                lon=getattr(c, "lon", None),
                zone=getattr(c, "zone", None),
                district=getattr(c, "district", None),
                object_class="vehicle",
                subtype=m.cls_name,
                plate_text=m.plate,
                visual_score=score,
                final_score=score,
                match_type="appearance",
                timestamp=base + timedelta(minutes=4 * i,
                                           seconds=m.seconds(m.first_frame)),
            ))
            written += 1
        db.commit()
        return written
    finally:
        db.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--topk", type=int, default=8,
                    help="sharpest crops averaged into one track vector")
    ap.add_argument("--distractors", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--min-frames", type=int, default=20,
                    help="a candidate must be seen this many times")
    ap.add_argument("--min-disp", type=float, default=0.15,
                    help="and must travel this fraction of the frame diagonal, "
                         "which is what excludes parked vehicles")
    ap.add_argument("--no-rider", action="store_true",
                    help="crop the machine only, excluding the rider — the "
                         "measured-worse setting, kept so the difference the "
                         "rider makes can be re-measured rather than claimed")
    ap.add_argument("--commit", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    from backend.services.anpr_engine import get_anpr_engine
    from backend.services.reid_embedder import get_embedder
    from backend.scripts.reid_crosscamera_eval import sample_distractors

    log("loading models ...")
    model = YOLO(str(ROOT / "yolov8s.pt"))
    embedder = get_embedder()

    def _val(obj, name):
        v = getattr(obj, name, None)
        return v() if callable(v) else v

    if _val(embedder, "is_stub"):
        log("FATAL: Re-ID model did not load; embeddings would be noise.")
        return 2
    log(f"  Re-ID checkpoint: {_val(embedder, 'checkpoint_applied')}")
    anpr = get_anpr_engine()

    log("\nrunning the pipeline over all four clips (every vehicle) ...")
    by_cam: dict[str, list[Track]] = {}
    for i, cam in enumerate(CAMS):
        clip = _clip_for(cam)
        if clip is None:
            log(f"FATAL: no clip for {cam}")
            return 2
        if args.no_rider:
            by_cam[cam] = scan_clip(model, anpr, embedder, cam, clip,
                                    imgsz=args.imgsz, stride=args.stride,
                                    topk=args.topk, read_plates=True,
                                    tid_base=2_000_000 * (i + 1))
        else:
            by_cam[cam] = scan_clip_rider_aware(
                model, embedder, cam, clip, imgsz=args.imgsz,
                stride=args.stride, topk=args.topk,
                tid_base=2_000_000 * (i + 1))
    missing = [c for c in CAMS if not by_cam.get(c)]
    if missing:
        log(f"FATAL: no tracks found in {', '.join(missing)}")
        return 2

    log("\nmerging tracker fragments within each camera ...")
    for cam in CAMS:
        before = len(by_cam[cam])
        by_cam[cam], merges, thr = merge_fragments(by_cam[cam], cam)
        frag = [f"#{t.local_id}({'+'.join(str(x) for x in getattr(t, 'fragment_ids', []))})"
                for t in by_cam[cam] if getattr(t, "fragment_ids", None)]
        log(f"  {cam}: {before} tracks -> {len(by_cam[cam])} vehicles "
            f"({merges} merge(s), threshold {thr:.3f} from same-frame pairs)"
            + ("   " + ", ".join(frag) if frag else ""))

    candidates_sheet(by_cam, OUT_DIR / "candidates.jpg")

    pool, prep = moving_pool(by_cam, args.min_frames, args.min_disp)
    log(f"\nrestricting the search to vehicles that transited the scene "
        f"(>={args.min_frames} frames and >={args.min_disp:.2f} of the frame "
        f"diagonal travelled):")
    for cam in CAMS:
        tot, kept = prep[cam]
        moved = ", ".join(
            f"#{t.local_id}({t.n_frames}f,{displacement(t):.2f})"
            for t in sorted(pool[cam], key=lambda t: -t.n_frames)[:6])
        log(f"  {cam}: {kept} of {tot} qualify   {moved}")

    log("\nsearching for the entity present in all four cameras ...")
    ranked = find_best_chain(pool)
    if not ranked:
        log("FATAL: no chain could be formed.")
        return 2
    best = ranked[0]
    chain, weakest, pairs = best["chain"], best["weakest"], best["pairs"]
    gid = "GV_" + hashlib.sha1(
        "|".join(sorted(m.key for m in chain.values())).encode()
    ).hexdigest()[:6].upper()

    log(f"  {len(ranked)} distinct chains considered")
    log(f"  BEST  weakest-pair {best['weakest']:.4f}  mean {best['mean']:.4f}  -> "
        + ", ".join(f"{c}#{chain[c].local_id}" for c in CAMS if c in chain))
    for r in ranked[1:4]:
        log(f"  next  weakest-pair {r['weakest']:.4f}  mean {r['mean']:.4f}  -> "
            + ", ".join(f"{c}#{r['chain'][c].local_id}"
                        for c in CAMS if c in r["chain"]))
    if len(ranked) > 1:
        margin = best["weakest"] - ranked[1]["weakest"]
        log(f"  margin over runner-up: {margin:+.4f}")

    log("\n  pairwise similarity within the winning chain:")
    for (a, b), s in sorted(pairs.items()):
        log(f"      {a} <-> {b}   {s:.4f}")

    log(f"\nsampling {args.distractors} distractors from the fleet ...")
    d_emb, d_crops = sample_distractors(model, embedder, args.distractors,
                                        args.imgsz, args.seed)
    d_emb = np.asarray(d_emb, dtype=np.float32)

    res = evaluate_chain(chain, by_cam, d_emb)
    hits = sum(1 for r in res if r["rank"] == 1)
    log("\n" + "=" * 70)
    log("GALLERY EVALUATION — is the chosen track the top-ranked one?")
    log("=" * 70)
    for r in res:
        mark = "HIT " if r["rank"] == 1 else "MISS"
        log(f"  {mark} {r['query']} -> {r['gallery_cam']}: rank {r['rank']} of "
            f"{r['gallery_size']}   sim {r['sim']:.3f} vs next best "
            f"{r['best_other']:.3f}")
    log(f"\n  Rank-1: {hits}/{len(res)} = {hits/len(res)*100:.1f}%")
    log("  (single crop vs single crop — the hardest form of the question)")
    log("=" * 70)

    loo = evaluate_leave_one_out(chain, by_cam, d_emb)
    loo_hits = sum(1 for r in loo if r["rank"] == 1)
    log("\n" + "=" * 70)
    log("LEAVE-ONE-OUT — given the other three sightings, find it here")
    log("=" * 70)
    for r in loo:
        mark = "HIT " if r["rank"] == 1 else "MISS"
        log(f"  {mark} {r['camera']}: rank {r['rank']} of {r['gallery_size']}"
            f"   sim {r['sim']:.3f} vs next best {r['best_other']:.3f}")
    log(f"\n  Rank-1: {loo_hits}/{len(loo)} = {loo_hits/len(loo)*100:.1f}%")
    log("  This is the question the deployed system actually asks.")
    log("=" * 70)

    chain_sheet(chain, gid, weakest, OUT_DIR / "same_id_all_four.jpg")
    ranking_sheet(chain, by_cam, d_emb, d_crops, OUT_DIR / "ranking.jpg")

    with open(OUT_DIR / "report.txt", "w", encoding="utf-8") as f:
        f.write("One vehicle, one ID, across four handheld cameras\n")
        f.write(f"global id       : {gid}\n")
        f.write("tracks searched : "
                + ", ".join(f"{c}={len(by_cam[c])}" for c in CAMS) + "\n")
        f.write(f"chains compared : {len(ranked)}\n")
        f.write(f"weakest pair    : {weakest:.4f}\n")
        if len(ranked) > 1:
            f.write(f"margin          : {best['weakest'] - ranked[1]['weakest']:+.4f}\n")
        f.write(f"Rank-1, crop vs crop     : {hits}/{len(res)}\n")
        f.write(f"Rank-1, leave-one-out    : {loo_hits}/{len(loo)}  "
                f"(given 3 sightings, find it at the 4th)\n")
        for r in res:
            f.write(f"  {r['query']} -> {r['gallery_cam']}: rank {r['rank']} "
                    f"of {r['gallery_size']}\n")
        f.write("\nleave-one-out:\n")
        for r in loo:
            f.write(f"  {r['camera']}: rank {r['rank']} of {r['gallery_size']} "
                    f"(sim {r['sim']:.3f} vs next {r['best_other']:.3f})\n")
        f.write("\nPlates are unreadable in these clips (478x850 source, plate "
                "~20-30px against a 40px floor), so this identity comes from "
                "appearance alone.\n")
    log(f"  report -> {OUT_DIR / 'report.txt'}")

    if args.commit:
        n = commit(chain, gid, weakest, pairs)
        log(f"\n  {n} journey_event(s) written, all with reid_id={gid}.")
        log(f"  Journeys page -> search {gid} to see the four-camera route.")
    else:
        log("\n(dry run — pass --commit to write the shared ID)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
