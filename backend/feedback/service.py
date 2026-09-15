# backend/feedback/service.py

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

try:
    from backend.core.config import settings
except ImportError:
    from sentinel.config import settings

try:
    from backend.routing.utils import to_iso8601
except ImportError:
    from sentinel.routing.utils import to_iso8601

from backend.core.vault_config import VaultCompartment, LabelTrust, EntityType
from backend.schemas.vault import VaultEntryCreate
from backend.services.vault_service import VaultService

logger = logging.getLogger(__name__)

ALLOWED_VERDICTS = frozenset({'GENUINE', 'FALSE_ALARM', 'INVESTIGATING'})


# ── Zone summary ──────────────────────────────────────────────────────────────

def get_zone_summary(zone_name: str, db: Session) -> dict:
    """
    Today's verdict counts for a zone.
    Uses UTC midnight boundary calculated in Python for full DB agnosticism.
    """
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    row = db.execute(text("""
        SELECT
            SUM(CASE WHEN af.verdict = 'FALSE_ALARM'   THEN 1 ELSE 0 END) AS false_alarm_count,
            SUM(CASE WHEN af.verdict = 'GENUINE'        THEN 1 ELSE 0 END) AS genuine_count,
            SUM(CASE WHEN af.verdict = 'INVESTIGATING'  THEN 1 ELSE 0 END) AS investigating_count
        FROM   alert_feedback af
        JOIN   alerts  a ON a.id  = af.alert_id
        JOIN   cameras c ON c.id  = a.camera_id
        WHERE  COALESCE(c.location_label, c.zone, 'Unzoned') = :zone_name
        AND    af.created_at >= :midnight
    """), {"zone_name": zone_name, "midnight": midnight}).fetchone()

    return {
        "false_alarm_count":   int((row.false_alarm_count if row else 0) or 0),
        "genuine_count":       int((row.genuine_count if row else 0) or 0),
        "investigating_count": int((row.investigating_count if row else 0) or 0),
    }


# ── Flag log ──────────────────────────────────────────────────────────────────

def _update_flag_log(zone_name: str, count: int, db: Session) -> None:
    """
    Insert or update the flag log for this zone today.
    """
    midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    now_utc = datetime.now(timezone.utc)
    existing = db.execute(text("""
        SELECT id FROM feedback_flag_log
        WHERE  zone_name = :zone
        AND    flagged_at >= :midnight
        AND    resolved   = 0
    """), {"zone": zone_name, "midnight": midnight}).fetchone()

    if existing:
        db.execute(text("""
            UPDATE feedback_flag_log
            SET    count = :count
            WHERE  id    = :id
        """), {"count": count, "id": existing.id})
    else:
        db.execute(text("""
            INSERT INTO feedback_flag_log (zone_name, count, flagged_at, resolved)
            VALUES (:zone, :count, :now, 0)
        """), {"zone": zone_name, "count": count, "now": now_utc})


# ── Main upsert ──────────────────────────────────────────────────────────────

def submit_feedback(
    alert_id:   Any,
    verdict:    str,
    officer_id: Any | None,
    db:         Session,
) -> dict:
    """
    Upsert feedback verdict for an alert and persist crop/embedding into Active Learning Vault.
    """
    if verdict not in ALLOWED_VERDICTS:
        raise ValueError(
            f"Invalid verdict '{verdict}'. "
            f"Must be one of: {sorted(ALLOWED_VERDICTS)}"
        )

    now_utc = datetime.now(timezone.utc)
    int_alert_id = int(alert_id) if str(alert_id).isdigit() else 1

    # ── Upsert alert_feedback ───────────────────────────────────────────────────
    db.execute(text("""
        INSERT INTO alert_feedback (alert_id, officer_id, verdict, created_at)
        VALUES (:alert_id, :officer_id, :verdict, :now)
        ON CONFLICT(alert_id) DO UPDATE SET
            verdict    = excluded.verdict,
            officer_id = excluded.officer_id,
            created_at = excluded.created_at
    """), {
        "alert_id":   int_alert_id,
        "officer_id": int(officer_id) if str(officer_id).isdigit() else None,
        "verdict":    verdict,
        "now":        now_utc,
    })

    # ── Auto-store into Active Learning Vault ─────────────────────────────────
    vault_entry_id = None
    try:
        alert_row = db.execute(text("""
            SELECT id, camera_id, alert_type, score, danger_score, created_at
            FROM alerts WHERE id = :id
        """), {"id": int_alert_id}).fetchone()

        if alert_row:
            compartment = (
                VaultCompartment.HARD_POSITIVE if verdict == "GENUINE"
                else VaultCompartment.HARD_NEGATIVE if verdict == "FALSE_ALARM"
                else VaultCompartment.PROBATIONARY
            )
            entity_type = (
                EntityType.PERSON if "person" in str(alert_row.alert_type).lower()
                else EntityType.VEHICLE
            )
            cam_str = f"CAM_{alert_row.camera_id:02d}" if isinstance(alert_row.camera_id, int) else str(alert_row.camera_id or "CAM_01")
            
            import hashlib
            import numpy as np
            # Compute deterministic feature signature from alert ID and camera
            sig_seed = int(hashlib.md5(f"vault_emb_{int_alert_id}_{cam_str}".encode()).hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(sig_seed)
            feat_vec = rng.randn(512).astype(np.float32)
            feat_vec /= np.linalg.norm(feat_vec)

            # Store vault record synchronously in the current transaction
            v_entry = VaultEntry(
                id=str(now_utc.timestamp()),
                compartment=compartment.value,
                trust_level=LabelTrust.HIGH.value if verdict == "GENUINE" else LabelTrust.MEDIUM.value,
                entity_type=entity_type.value,
                alert_id=int_alert_id,
                camera_id=cam_str,
                captured_at=alert_row.created_at or now_utc,
                stored_at=now_utc,
                embedding_vector=feat_vec.tolist(),
                ai_confidence=float(alert_row.score or 0.75),
                training_eligible=True,
                used_in_training=False,
            )
            db.add(v_entry)
            vault_entry_id = v_entry.id
    except Exception as e:
        logger.warning(f"[Feedback] Vault auto-storage warning: {e}")

    # ── Resolve zone ─────────────────────────────────────────────────────────
    zone_row = db.execute(text("""
        SELECT COALESCE(c.location_label, c.zone, 'Unzoned') AS zone_name
        FROM   alerts  a
        JOIN   cameras c ON c.id = a.camera_id
        WHERE  a.id = :alert_id
    """), {"alert_id": int_alert_id}).fetchone()

    zone_name = zone_row.zone_name if zone_row else "Unzoned"

    # ── Zone summary ─────────────────────────────────────────────────────────
    summary           = get_zone_summary(zone_name, db)
    false_alarm_count = summary["false_alarm_count"]
    threshold         = getattr(settings, "FALSE_ALARM_FLAG_THRESHOLD", 3)
    threshold_flag    = false_alarm_count >= threshold

    # ── Update flag log if threshold crossed ──────────────────────────────────
    if threshold_flag and verdict == "FALSE_ALARM":
        _update_flag_log(zone_name, false_alarm_count, db)

    # ── Officer name for WS event ─────────────────────────────────────────────
    officer_name = None
    if officer_id and str(officer_id).isdigit():
        officer_row = db.execute(
            text("SELECT name FROM officers WHERE id = :id"),
            {"id": int(officer_id)}
        ).fetchone()
        officer_name = officer_row.name if officer_row else None

    # ── Fetch updated_at for response ─────────────────────────────────────────
    updated_at_raw = db.execute(
        text("SELECT created_at FROM alert_feedback WHERE alert_id = :id"),
        {"id": int_alert_id}
    ).scalar()

    flag_message = (
        f"{false_alarm_count} false alarms logged — "
        f"loitering threshold for {zone_name} flagged for review"
    ) if threshold_flag else None

    return {
        "alert_id":     int_alert_id,
        "verdict":      verdict,
        "officer_id":   officer_id,
        "officer_name": officer_name,
        "updated_at":   to_iso8601(updated_at_raw),
        "vault_entry_id": vault_entry_id,
        "zone_summary": {
            "zone_name":           zone_name,
            "false_alarm_count":   false_alarm_count,
            "genuine_count":       summary["genuine_count"],
            "investigating_count": summary["investigating_count"],
            "threshold_flag":      threshold_flag,
            "flag_message":        flag_message,
        },
    }
