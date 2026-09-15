"""
backend/routers/v1/shadow_admin.py — Protected Shadow Mode Metrics & Admin REST API (v15.0.0)

Closes Gap CC: Requires JWT authentication with admin/supervisor role before exposing shadow metrics.
"""
import sqlite3
from fastapi import APIRouter, Depends, HTTPException
from backend.auth import get_current_reviewer, ReviewerToken

router = APIRouter(prefix="/v1/admin/shadow", tags=["shadow_admin"])


def require_admin_or_supervisor(reviewer: ReviewerToken = Depends(get_current_reviewer)) -> ReviewerToken:
    if not any(r in ("admin", "supervisor", "lead") for r in reviewer.roles):
        raise HTTPException(status_code=403, detail="Forbidden: Requires admin or supervisor role")
    return reviewer


@router.get("/metrics")
async def get_shadow_metrics(
    user: ReviewerToken = Depends(require_admin_or_supervisor),
    db_path: str = "sentinel.db",
):
    try:
        conn = sqlite3.connect(db_path)
        total_shadow = conn.execute("SELECT COUNT(*) FROM shadow_violations").fetchone()[0]
        rows = conn.execute("""
            SELECT violation_type, COUNT(*), AVG(majority_ratio), AVG(speed_kmh)
            FROM shadow_violations
            GROUP BY violation_type
        """).fetchall()
        conn.close()

        breakdown = [
            {
                "violation_type": r[0],
                "count": r[1],
                "avg_majority_ratio": round(r[2], 3) if r[2] else None,
                "avg_speed_kmh": round(r[3], 1) if r[3] else None,
            }
            for r in rows
        ]

        return {
            "status": "ok",
            "total_shadow_violations": total_shadow,
            "breakdown": breakdown,
            "authorized_user": user.username,
        }
    except Exception as e:
        return {
            "status": "ok",
            "total_shadow_violations": 0,
            "breakdown": [],
            "note": str(e),
        }
