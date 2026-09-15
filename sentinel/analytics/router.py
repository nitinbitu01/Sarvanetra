# sentinel/analytics/router.py
"""
sentinel/analytics/router.py — Analytics endpoints powering the production Analytics Dashboard.
"""

from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.auth.dependencies import AnyOfficer
from sentinel.analytics.service import (
    get_live_overview,
    get_zone_performance,
    get_officer_performance,
    get_trend,
)
from sentinel.db import get_db

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/overview")
async def analytics_overview(
    officer: AnyOfficer,
    db: AsyncSession = Depends(get_db),
):
    """
    Get aggregated overview of alerts, severity counts, and response performance.
    """
    return await get_live_overview(db)


@router.get("/zones")
async def analytics_zones(
    officer: AnyOfficer,
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """
    Get zone-by-zone performance and false alarm distributions.
    """
    return await get_zone_performance(days, db)


@router.get("/officers")
async def analytics_officers(
    officer: AnyOfficer,
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """
    Get officer response times and acknowledgement rates.
    """
    return await get_officer_performance(days, db)


@router.get("/trend")
async def analytics_trend(
    officer: AnyOfficer,
    days: int = Query(7, ge=1, le=90),
    db: AsyncSession = Depends(get_db),
):
    """
    Get multi-day trend bars for alerts over time.
    """
    return await get_trend(days, db)
