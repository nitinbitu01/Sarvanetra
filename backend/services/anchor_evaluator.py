"""
backend/services/anchor_evaluator.py — Anchor-Set Regression Gate for Camera Thresholds.
Evaluates proposed thresholds against frozen ground-truth anchor sets to prevent
performance regression during automated periodic calibration.
"""
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import select, and_

from backend.db.models import VaultEntry, CameraCalibrationProfile
from backend.core.vault_config import MIN_ANCHOR_GROUPS, VaultCompartment

logger = logging.getLogger("sentinel.anchor_eval")


class AnchorEvaluator:
    """
    Evaluates proposed ReID distance thresholds and HSV color envelopes against
    the frozen anchor set before any camera profile is activated.
    """

    def __init__(self, db: Session):
        self.db = db

    def evaluate_profile(
        self,
        camera_id: str,
        proposed_thresholds: Dict[str, Any],
        lighting_condition: str = "day",
    ) -> Dict[str, Any]:
        """
        Runs anchor evaluation for proposed thresholds.
        Returns:
            {
                "passed": bool,
                "anchor_groups_count": int,
                "top1_accuracy": float,
                "false_positive_rate": float,
                "baseline_top1": float,
                "reason": str,
            }
        """
        # Fetch anchor entries
        anchors = (
            self.db.query(VaultEntry)
            .filter(
                VaultEntry.compartment == VaultCompartment.ANCHOR.value,
                VaultEntry.anchor_group_id.isnot(None),
            )
            .all()
        )

        groups = {}
        for a in anchors:
            gid = a.anchor_group_id
            if gid not in groups:
                groups[gid] = []
            vec = a.embedding_vector
            arr = None
            if isinstance(vec, np.ndarray):
                arr = vec.astype(np.float32)
            elif isinstance(vec, list):
                arr = np.array(vec, dtype=np.float32)
            elif isinstance(vec, str):
                s = vec.strip()
                if s.startswith("[") and s.endswith("]"):
                    s = s[1:-1]
                try:
                    parts = [float(x.strip()) for x in s.split(",") if x.strip()]
                    arr = np.array(parts, dtype=np.float32)
                except Exception:
                    arr = None

            if arr is not None and len(arr) > 0:
                norm = float(np.linalg.norm(arr))
                if norm > 0:
                    groups[gid].append(arr / norm)

        # Filter groups that have at least 2 crops for probe-gallery matching
        valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}

        if len(valid_groups) < MIN_ANCHOR_GROUPS:
            logger.warning(
                f"[ANCHOR EVAL] Camera {camera_id}: Insufficient anchor groups "
                f"({len(valid_groups)} < {MIN_ANCHOR_GROUPS}). Flagging for supervisor review."
            )
            return {
                "passed": False,
                "anchor_groups_count": len(valid_groups),
                "top1_accuracy": 0.0,
                "false_positive_rate": 0.0,
                "baseline_top1": 0.95,
                "reason": f"Insufficient anchor groups ({len(valid_groups)}/{MIN_ANCHOR_GROUPS} required)",
            }

        # Proposed vehicle threshold
        proposed_thresh = float(
            proposed_thresholds.get(lighting_condition, {}).get("vehicle", 0.35)
        )

        # Run all-vs-all anchor validation
        correct_top1 = 0
        total_queries = 0
        false_positives = 0
        total_cross_comparisons = 0

        group_keys = list(valid_groups.keys())
        for g_idx, gid in enumerate(group_keys):
            vecs = valid_groups[gid]
            probe = vecs[0]
            gallery_same = vecs[1:]

            for same_vec in gallery_same:
                total_queries += 1
                dist_same = 1.0 - float(np.dot(probe, same_vec))
                if dist_same <= proposed_thresh:
                    correct_top1 += 1

            # Cross-group comparison for false positive rate
            for other_gid in group_keys:
                if other_gid == gid:
                    continue
                for diff_vec in valid_groups[other_gid]:
                    total_cross_comparisons += 1
                    dist_diff = 1.0 - float(np.dot(probe, diff_vec))
                    if dist_diff <= proposed_thresh:
                        false_positives += 1

        top1_acc = correct_top1 / max(1, total_queries)
        fp_rate = false_positives / max(1, total_cross_comparisons)

        # Baseline criteria: Top-1 Accuracy >= 85% and False Positive Rate <= 3.0%
        passed = (top1_acc >= 0.85) and (fp_rate <= 0.03)

        report = {
            "passed": passed,
            "anchor_groups_count": len(valid_groups),
            "top1_accuracy": round(top1_acc, 4),
            "false_positive_rate": round(fp_rate, 4),
            "baseline_top1": 0.85,
            "tested_threshold": proposed_thresh,
            "lighting_condition": lighting_condition,
            "reason": "Passed safety gate" if passed else f"Degraded metrics: Top-1={top1_acc:.2%}, FP={fp_rate:.2%}",
        }

        logger.info(f"[ANCHOR EVAL] Camera {camera_id} [{lighting_condition}]: {report}")
        return report
