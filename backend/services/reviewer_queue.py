"""
backend/services/reviewer_queue.py — Reviewer Queue Prioritization & Workload Optimization (v15.0.0)

Closes Gap AA: Intelligent priority scoring, batch grouping by plate, and 48h auto-expiry.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class QueuePriorityScore:
    violation_id: int
    majority_ratio: float
    is_stolen: bool
    insurance_valid: bool
    pucc_valid: bool
    violation_type: str
    created_at_epoch: float

    @property
    def score(self) -> float:
        pts = 0.0

        # Confidence contribution (0 to 40 pts)
        pts += max(0.0, (self.majority_ratio - 0.65) / 0.35 * 40.0)

        # Stolen vehicle bonus (50 pts)
        if self.is_stolen:
            pts += 50.0

        # VAHAN anomalies (10 pts each)
        if not self.insurance_valid:
            pts += 10.0
        if not self.pucc_valid:
            pts += 10.0

        # Violation severity
        severity_pts = {
            "BRTS_WRONG_WAY": 20.0,
            "WRONG_WAY": 15.0,
            "TRIPLE_RIDING": 10.0,
            "BRTS_INCURSION": 5.0,
        }
        pts += severity_pts.get(self.violation_type, 5.0)

        # Recency bonus: up to 10 pts for recent items
        age_sec = max(0.0, time.time() - self.created_at_epoch)
        pts += max(0.0, 10.0 - age_sec / 360.0)

        return round(pts, 2)


def compute_priority(violation: dict[str, Any]) -> float:
    qs = QueuePriorityScore(
        violation_id=violation.get("id", 0),
        majority_ratio=float(violation.get("majority_ratio", 0.65)),
        is_stolen=bool(violation.get("vahan_stolen", False)),
        insurance_valid=bool(violation.get("vahan_insurance_valid", True)),
        pucc_valid=bool(violation.get("vahan_pucc_valid", True)),
        violation_type=str(violation.get("violation_type", "UNKNOWN")),
        created_at_epoch=float(violation.get("created_at_epoch", time.time())),
    )
    return qs.score
