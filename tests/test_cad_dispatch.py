"""
tests/test_cad_dispatch.py — Dial-112 CAD Patrol Dispatch Tests (v15.0.0).
"""
import time
import pytest

from backend.services.cad_dispatch import (
    CADDispatcher,
    PCRUnit,
    UnitStatus,
)


def test_nearest_patrol_allocation_and_eta():
    u1 = PCRUnit("PCR-A", "Alpha", "Ahmedabad", "Officer A", "+91-1", 23.0200, 72.5700, UnitStatus.AVAILABLE)
    u2 = PCRUnit("PCR-B", "Bravo", "Ahmedabad", "Officer B", "+91-2", 23.0800, 72.5700, UnitStatus.AVAILABLE)
    disp = CADDispatcher([u1, u2])

    # Incident at (23.0210, 72.5710) — much closer to PCR-A
    res = disp.dispatch_nearest(23.0210, 72.5710, severity="CRITICAL", incident_id="INC-101")
    assert res is not None
    assert res.unit_id == "PCR-A"
    assert res.distance_km < 1.0
    assert res.eta_minutes < 2.0
    assert u1.status == UnitStatus.DISPATCHED


def test_stale_gps_unit_skipped():
    # Unit with stale GPS (> 60s ago)
    u_stale = PCRUnit("PCR-S", "Stale", "Ahmedabad", "Officer S", "+91-3", 23.0200, 72.5700, UnitStatus.AVAILABLE)
    u_stale.last_gps_update = time.monotonic() - 120.0  # 2 minutes ago!

    u_fresh = PCRUnit("PCR-F", "Fresh", "Ahmedabad", "Officer F", "+91-4", 23.0500, 72.5700, UnitStatus.AVAILABLE)
    u_fresh.last_gps_update = time.monotonic()  # right now

    disp = CADDispatcher([u_stale, u_fresh])
    res = disp.dispatch_nearest(23.0210, 72.5710, severity="HIGH", incident_id="INC-102")

    # PCR-S is closer, but skipped because its GPS is stale!
    assert res is not None
    assert res.unit_id == "PCR-F"


def test_unit_status_lifecycle():
    u = PCRUnit("PCR-01", "A1", "Ahmedabad", "Officer", "+91-0", 23.0, 72.0, UnitStatus.AVAILABLE)
    disp = CADDispatcher([u])

    assert disp.update_unit_status("PCR-01", UnitStatus.DISPATCHED, "INC-01") is True
    assert u.status == UnitStatus.DISPATCHED

    assert disp.update_unit_status("PCR-01", UnitStatus.ON_SCENE) is True
    assert u.status == UnitStatus.ON_SCENE

    assert disp.update_unit_status("PCR-01", UnitStatus.AVAILABLE) is True
    assert u.status == UnitStatus.AVAILABLE
