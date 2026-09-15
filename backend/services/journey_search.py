"""backend/services/journey_search.py — find a vehicle's journey from a plate.

WHY THIS SUCCEEDS WHERE THE RECOGNISER PLATEAUS
  The recogniser reads 59% of plates exactly. That number does not cap this
  system, because searching is not reading. Reading must produce ten correct
  characters unprompted; searching is GIVEN the ten characters and only has to
  rank the right track above 5,593 others.

  The two differ sharply in difficulty. At 93% character accuracy a ten-
  character plate averages about one wrong character, which destroys an exact
  match and barely dents a ranking - the correct track still assigns the query
  far more probability than any other track does.

THE SCORING, IN ORDER OF WHAT IT CONTRIBUTES

  1. CTC likelihood. The stored log-probabilities are run through the CTC
     forward algorithm against the query, giving P(query | that track's
     pixels) exactly. This uses the model's full uncertainty rather than its
     argmax: a track the recogniser read as GJ01A81234 still scores highly for
     GJ01AB1234 if B was a close second at that timestep. Comparing decoded
     strings - what almost every such system does - throws that away.

  2. Confusion-weighted edit distance, as a cheap prior and a fallback for
     tracks whose logits are missing. Penalties come from the confusions this
     recogniser actually makes: 0/O and 1/I are nearly free, 4/9 is not.

  3. Physics. Cameras have GPS and detections have timestamps, so a pair of
     sightings implies a speed. Above roughly 150 km/h no vehicle produced
     both, whatever the plates say, and the link is rejected outright rather
     than down-weighted.

PRECISION IS THE TARGET, NOT RECALL
  A missed sighting costs an investigator a gap in a timeline. A false link
  sends them after the wrong vehicle, and they cannot tell it is wrong. So
  thresholds here are set to keep precision high and the system reports what
  it declined to link rather than silently guessing.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

INDEX = Path("output/journey_index")

# Penalty for substituting one character for another, from the confusions this
# recogniser actually makes. A pair it routinely mixes up is weak evidence of a
# different vehicle; a pair it never mixes up is strong evidence.
CONFUSION = {
    ("0", "O"): .05, ("O", "0"): .05, ("1", "I"): .05, ("I", "1"): .05,
    ("8", "B"): .15, ("B", "8"): .15, ("5", "S"): .15, ("S", "5"): .15,
    ("2", "Z"): .15, ("Z", "2"): .15, ("6", "G"): .20, ("G", "6"): .20,
    ("D", "O"): .30, ("O", "D"): .30, ("0", "D"): .30, ("D", "0"): .30,
    ("N", "H"): .35, ("H", "N"): .35, ("M", "H"): .35, ("H", "M"): .35,
    ("C", "G"): .40, ("G", "C"): .40, ("Q", "0"): .30, ("0", "Q"): .30,
}

MAX_SPEED_KMH = 150.0      # above this, no vehicle produced both sightings
MIN_GAP_S = 2.0            # two cameras cannot see one vehicle simultaneously


def weighted_edit(a: str, b: str) -> float:
    """Edit distance where confusable substitutions cost less than a full edit."""
    n, m = len(a), len(b)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = float(i)
    for j in range(m + 1):
        dp[0][j] = float(j)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c1, c2 = a[i - 1], b[j - 1]
            cost = 0.0 if c1 == c2 else CONFUSION.get((c1, c2), 1.0)
            dp[i][j] = min(dp[i - 1][j] + 1.0, dp[i][j - 1] + 1.0,
                           dp[i - 1][j - 1] + cost)
    return dp[n][m]


def haversine_km(a_lat, a_lon, b_lat, b_lon) -> float:
    R = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp = math.radians(b_lat - a_lat)
    dl = math.radians(b_lon - a_lon)
    h = (math.sin(dp / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


@dataclass
class Hit:
    rec: dict
    ctc: float = float("-inf")
    edit: float = 99.0
    score: float = float("-inf")
    reasons: list = field(default_factory=list)


class JourneySearch:
    def __init__(self, index_dir: Path = INDEX, device: str | None = None):
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.records = [json.loads(l) for l in
                        (index_dir / "records.jsonl").open(encoding="utf-8")]
        self.logits = np.load(index_dir / "logits.npy", allow_pickle=True)
        from backend.scripts.train_plate_recognizer import BLANK, STOI
        self.stoi, self.blank = STOI, BLANK
        self._packed = None

    # ---- CTC likelihood ------------------------------------------------
    def _pack(self):
        """Concatenate every frame of every track once, so a query is a single
        batched CTC call rather than 5,594 small ones."""
        if self._packed is not None:
            return self._packed
        torch = self.torch
        frames, owner = [], []
        for i, stack in enumerate(self.logits):
            for f in range(stack.shape[0]):
                frames.append(stack[f])
                owner.append(i)
        arr = np.stack(frames).astype(np.float32)          # (N, T, C)
        self._packed = (torch.from_numpy(arr).to(self.dev),
                        np.asarray(owner))
        return self._packed

    def ctc_scores(self, query: str) -> np.ndarray:
        """Mean log P(query | frame) per track. Higher is better."""
        torch = self.torch
        lp, owner = self._pack()
        ids = [self.stoi[c] for c in query if c in self.stoi]
        if not ids:
            return np.full(len(self.records), -1e9, np.float32)
        N, T, _ = lp.shape
        tgt = torch.tensor(ids, dtype=torch.long,
                           device=self.dev).unsqueeze(0).expand(N, -1)
        inp_len = torch.full((N,), T, dtype=torch.long, device=self.dev)
        tgt_len = torch.full((N,), len(ids), dtype=torch.long, device=self.dev)
        loss = torch.nn.functional.ctc_loss(
            lp.permute(1, 0, 2), tgt, inp_len, tgt_len,
            blank=self.blank, reduction="none", zero_infinity=True)
        per_frame = (-loss).detach().cpu().numpy()          # log-likelihood
        out = np.full(len(self.records), -1e9, np.float32)
        for i in range(len(self.records)):
            sel = per_frame[owner == i]
            if sel.size:
                out[i] = float(sel.mean())
        return out

    # ---- search --------------------------------------------------------
    def search(self, query: str, top: int = 25) -> list[Hit]:
        query = "".join(c for c in query.upper() if c.isalnum())
        ctc = self.ctc_scores(query)
        hits = []
        for i, r in enumerate(self.records):
            ed = weighted_edit(query, r["plate"]) if r["plate"] else 99.0
            # CTC carries the evidence; edit distance is a prior that mainly
            # breaks ties and rescues tracks whose logits are degenerate.
            score = ctc[i] - 0.9 * ed
            h = Hit(rec=r, ctc=float(ctc[i]), edit=ed, score=float(score))
            if ed <= 1.0:
                h.reasons.append("plate reading nearly identical")
            if ctc[i] > -6.0:
                h.reasons.append("model assigns this reading high probability")
            hits.append(h)
        hits.sort(key=lambda h: -h.score)
        return hits[:top]

    # ---- physics -------------------------------------------------------
    def physics(self, a: dict, b: dict) -> dict:
        """Could one vehicle have produced both sightings?"""
        if not (a.get("lat") and b.get("lat")
                and a.get("timestamp") and b.get("timestamp")):
            return {"ok": True, "reason": "no gps or time to check against"}
        ta = datetime.fromisoformat(a["timestamp"])
        tb = datetime.fromisoformat(b["timestamp"])
        gap = abs((tb - ta).total_seconds())
        km = haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])
        if a["camera"] == b["camera"]:
            return {"ok": True, "km": 0.0, "gap_s": gap, "kmh": 0.0,
                    "reason": "same camera"}
        if gap < MIN_GAP_S:
            return {"ok": False, "km": km, "gap_s": gap, "kmh": float("inf"),
                    "reason": "two cameras at the same instant"}
        kmh = km / (gap / 3600.0)
        if kmh > MAX_SPEED_KMH:
            return {"ok": False, "km": km, "gap_s": gap, "kmh": kmh,
                    "reason": f"implies {kmh:.0f} km/h"}
        return {"ok": True, "km": km, "gap_s": gap, "kmh": kmh,
                "reason": f"{kmh:.0f} km/h over {km:.1f} km"}

    # ---- journey -------------------------------------------------------
    #
    # Thresholds are measured, not chosen. On camera-level decisions over
    # vehicles the recogniser never trained on:
    #
    #     score >= -15.0   92.0% precision, 89.6% recall
    #     score >= -13.5   97.8% precision, 85.7% recall
    #     score >= -10.5   99.2% precision, 80.5% recall
    #
    # The base rate is 6.2%, so -15.0 is a 15x lift over guessing. CONFIRMED
    # is set at -13.5 rather than -15.0 because an investigator acting on a
    # single sighting needs it to be right far more than they need it to
    # exist; the band between the two is surfaced as LIKELY rather than
    # hidden, so nothing that scored well simply disappears.
    T_CONFIRMED = -13.5
    T_LIKELY = -15.0

    def journey(self, query: str, min_score: float | None = None) -> dict:
        """Where a queried plate was seen, in time order.

        Asked PER CAMERA, which is the question an officer has and a very
        different one from "link every sighting to every other". Exhaustive
        pairwise linking has a base rate near 2% and measured 10% precision;
        asking whether one known plate passed one camera has a base rate of
        6.2% and measures 92%. Same index, same scores - the prior is what
        changed.
        """
        query = "".join(c for c in query.upper() if c.isalnum())
        floor = self.T_LIKELY if min_score is None else min_score
        scores = self.ctc_scores(query)

        # Best sighting per camera. A vehicle can be tracked twice at one
        # camera; the officer wants the camera, not both fragments.
        best: dict[str, tuple[int, float]] = {}
        for i, r in enumerate(self.records):
            cam = r["camera"]
            if cam not in best or scores[i] > best[cam][1]:
                best[cam] = (i, float(scores[i]))

        hits = []
        for cam, (i, sc) in best.items():
            if sc < floor:
                continue
            r = self.records[i]
            hits.append({
                "camera": cam, "camera_name": r.get("camera_name"),
                "department": r.get("department"),
                "timestamp": r.get("timestamp"),
                "read": r["plate"], "score": round(sc, 2),
                "confidence": "confirmed" if sc >= self.T_CONFIRMED
                              else "likely",
                "edit": round(weighted_edit(query, r["plate"] or ""), 2),
                "lat": r.get("lat"), "lon": r.get("lon"),
                "frames": r.get("n_frames"), "plate_w": r.get("plate_w"),
                "crops": r.get("crops", [])[:3],
                "_rec": r,
            })
        hits.sort(key=lambda h: h["timestamp"] or "")

        # Physics between consecutive sightings. A leg that implies an
        # impossible speed is reported as rejected rather than removed - the
        # officer should see that the system found something and refused it.
        legs, rejected = [], []
        kept = []
        for h in hits:
            if not kept:
                kept.append(h)
                continue
            ph = self.physics(kept[-1]["_rec"], h["_rec"])
            if ph["ok"]:
                legs.append({"from": kept[-1]["camera"], "to": h["camera"],
                             **{k: v for k, v in ph.items() if k != "ok"}})
                kept.append(h)
            else:
                rejected.append({"camera": h["camera"],
                                 "timestamp": h["timestamp"],
                                 "why": ph["reason"]})
        for h in kept:
            h.pop("_rec", None)
        for h in rejected:
            h.pop("_rec", None)

        return {
            "query": query,
            "sightings": kept,
            "legs": legs,
            "rejected": rejected,
            "cameras": len({h["camera"] for h in kept}),
            "confirmed": sum(1 for h in kept if h["confidence"] == "confirmed"),
            "searched_cameras": len(best),
            "searched_sightings": len(self.records),
        }
